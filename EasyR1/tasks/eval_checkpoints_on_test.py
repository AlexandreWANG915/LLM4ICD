"""Evaluate multiple checkpoints on the held-out test set with PLM-ICD metrics.

Reports β-invariant metrics so we can compare across rounds that were trained
with different F_β values. Metric formulas match
${MED_CODING_REPRO_ROOT}/src/metrics.py:464-508 exactly.

Usage:
    python tasks/eval_checkpoints_on_test.py \
        --base_model ${SFT_MODEL_PATH:-${REPO_ROOT}/models/qwen2.5-1.5b-icd-sft-top50} \
        --test_parquet ${REPO_ROOT}/data/icd_grpo_top50_hint/test.parquet \
        --output_dir ${REPO_ROOT}/EasyR1/tasks/logs/phase9_eval \
        --num_gpus 4 \
        --adapter_paths NONE /path/to/adapter_A /path/to/adapter_B ...

Each positional `adapter_paths` value is either a LoRA adapter directory
(containing adapter_config.json + adapter_model.safetensors) or the literal
string "NONE" to evaluate the base model alone.
"""

import argparse
import gc
import hashlib
import json
import multiprocessing
import os
import re
import sys
from collections import Counter
from pathlib import Path

# vLLM v1 needs spawn; must be set before any CUDA init.
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks.icd_label_space import filter_allowed_codes, load_allowed_codes, parse_answer_codes
from verl.utils.note_truncate import chat_template_overhead, truncate_for_chat_prompt


CODE_BLOCK_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)


def parse_codes(
    response: str,
    allowed_codes: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    m = CODE_BLOCK_RE.search(response)
    if not m:
        return [], []
    codes, seen = [], set()
    for part in re.split(r"[,;\n]", m.group(1)):
        c = part.strip().rstrip(".").upper()
        if c and c not in seen:
            seen.add(c)
            codes.append(c)
    return filter_allowed_codes(codes, allowed_codes)


def parse_gt(answer: str) -> list[str]:
    return parse_answer_codes(answer)


def compute_corpus_metrics(
    records: list[dict],
    label_space: set[str] | None = None,
) -> dict:
    """PLM-ICD compliant micro + macro P/R/F1.

    Formulas (from medical-coding-reproducibility/src/metrics.py):
      micro_f1 = 2*TP.sum() / (2*TP.sum() + FP.sum() + FN.sum())
      macro_f1 = mean_c(2*TP_c / (2*TP_c + FP_c + FN_c))
      micro_p  = TP.sum() / (TP.sum() + FP.sum())
      macro_p  = mean_c(TP_c / (TP_c + FP_c))
      micro_r  = TP.sum() / (TP.sum() + FN.sum())
      macro_r  = mean_c(TP_c / (TP_c + FN_c))
    """
    tp_per_code: Counter[str] = Counter()
    fp_per_code: Counter[str] = Counter()
    fn_per_code: Counter[str] = Counter()
    tp_total = fp_total = fn_total = 0
    n = len(records)
    mastered = 0
    total_missed = 0
    total_pred = 0
    total_gt = 0

    for r in records:
        pred = set(r["pred_codes"])
        gt = set(r["gt_codes"])
        tp = pred & gt
        fp = pred - gt
        fn = gt - pred
        tp_per_code.update(tp)
        fp_per_code.update(fp)
        fn_per_code.update(fn)
        tp_total += len(tp)
        fp_total += len(fp)
        fn_total += len(fn)
        if not r["missed"]:
            mastered += 1
        total_missed += len(r["missed"])
        total_pred += len(pred)
        total_gt += len(gt)

    # Micro (pooled)
    if tp_total + fp_total + fn_total == 0:
        micro_p = micro_r = micro_f1 = 1.0
    else:
        micro_p = tp_total / max(tp_total + fp_total, 1)
        micro_r = tp_total / max(tp_total + fn_total, 1)
        micro_f1 = 2 * tp_total / max(2 * tp_total + fp_total + fn_total, 1)

    # Macro (per-class then mean)
    seen_codes = set(tp_per_code) | set(fp_per_code) | set(fn_per_code)
    codes = set(label_space) if label_space is not None else seen_codes
    p_list, r_list, f1_list = [], [], []
    per_code_stats = {}
    for c in codes:
        tp = tp_per_code[c]
        fp = fp_per_code[c]
        fn = fn_per_code[c]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        p_list.append(p)
        r_list.append(r)
        f1_list.append(f1)
        per_code_stats[c] = {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f1}

    if codes:
        macro_p = sum(p_list) / len(p_list)
        macro_r = sum(r_list) / len(r_list)
        macro_f1 = sum(f1_list) / len(f1_list)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    return {
        "n_samples": n,
        "mastered": mastered,
        "mastered_pct": 100 * mastered / max(n, 1),
        "avg_missed": total_missed / max(n, 1),
        "avg_pred": total_pred / max(n, 1),
        "avg_gt": total_gt / max(n, 1),
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total,
        "micro_p": micro_p,
        "micro_r": micro_r,
        "micro_f1": micro_f1,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "n_codes_seen": len(codes),
        "n_codes_with_activity": len(seen_codes),
        "per_code": per_code_stats,
    }


def run_inference(
    base_model: str,
    adapter_path: str | None,
    prompts: list[str],
    num_gpus: int,
    max_model_len: int,
    gpu_memory_utilization: float = 0.80,
    temperature: float = 0.0,
    max_tokens: int = 196,
) -> list[str]:
    """Spin up vLLM, generate, tear down. Return list of response strings."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm_kwargs = dict(
        model=base_model,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=num_gpus,
        disable_custom_all_reduce=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        disable_log_stats=True,
        enforce_eager=bool(adapter_path),
    )
    lora_req = None
    if adapter_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64
        lora_req = LoRARequest("eval_adapter", 1, adapter_path)

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=temperature,
        top_p=1.0 if temperature == 0.0 else 0.9,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(prompts=prompts, sampling_params=sampling, lora_request=lora_req)
    responses = [o.outputs[0].text if o.outputs else "" for o in outputs]

    # Try to release vLLM resources — next checkpoint eval needs clean GPU.
    import torch
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return responses


def render_prompts(
    df, tokenizer, max_prompt_length: int, template_overhead: int
) -> tuple[list[str], list[int]]:
    """Truncate overlong notes + apply chat template. Return (prompts, kept_indices)."""
    from tqdm import tqdm
    prompts, kept = [], []
    n_truncated = n_failed = 0
    first_failure_logged = False
    for i, problem in enumerate(tqdm(df["problem"].tolist(), desc="render")):
        try:
            p = truncate_for_chat_prompt(
                problem, tokenizer,
                max_prompt_length=max_prompt_length,
                template_overhead=template_overhead,
                enable_thinking=False,
            )
            if p is not problem:
                n_truncated += 1
        except ValueError as e:
            n_failed += 1
            if not first_failure_logged:
                print(f"  sample {i}: truncation failed ({e}); skipping")
                first_failure_logged = True
            continue
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        prompts.append(text)
        kept.append(i)
    print(f"  Rendered {len(prompts)}/{len(df)} prompts "
          f"({n_truncated} truncated, {n_failed} failed)")
    return prompts, kept


def evaluate_one(
    tag: str,
    base_model: str,
    adapter_path: str | None,
    df,
    prompts: list[str],
    kept_indices: list[int],
    num_gpus: int,
    max_model_len: int,
    max_tokens: int,
    output_dir: str,
    allowed_codes: set[str] | None,
) -> dict:
    print(f"\n{'=' * 70}")
    print(f"[{tag}] Running inference")
    print(f"  base_model: {base_model}")
    print(f"  adapter:    {adapter_path if adapter_path else '<none, base only>'}")
    print('=' * 70)

    responses = run_inference(
        base_model, adapter_path, prompts, num_gpus, max_model_len,
        max_tokens=max_tokens,
    )

    records = []
    total_off_label = 0
    for kept_idx, resp in zip(kept_indices, responses):
        row = df.iloc[kept_idx]
        pred, off_label = parse_codes(resp, allowed_codes=allowed_codes)
        pred = sorted(set(pred))
        gt = sorted(set(parse_gt(row["answer"])))
        missed = sorted(set(gt) - set(pred))
        total_off_label += len(off_label)
        records.append({
            "sample_id": hashlib.sha1(row["problem"].encode("utf-8")).hexdigest()[:12],
            "pred_codes": pred,
            "off_label_codes": off_label,
            "gt_codes": gt,
            "missed": missed,
            "n_pred": len(pred),
            "n_off_label": len(off_label),
            "n_gt": len(gt),
            "n_missed": len(missed),
            "raw_response": resp[:200],  # truncate for size
        })

    # Dump JSONL
    jsonl_path = os.path.join(output_dir, f"eval_{tag}.jsonl")
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Compute corpus metrics
    metrics = compute_corpus_metrics(records, label_space=allowed_codes)
    metrics["tag"] = tag
    metrics["adapter_path"] = adapter_path or "<base>"
    metrics["off_label_total"] = total_off_label
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--test_parquet", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=5120)
    parser.add_argument("--max_tokens", type=int, default=196)
    parser.add_argument("--allowed_codes_file", default=None,
                        help="Optional newline/JSON code list for closed-set "
                             "filtering. If omitted, derive from "
                             "--allowed_codes_parquet or --test_parquet.")
    parser.add_argument("--allowed_codes_parquet", default=None,
                        help="Optional parquet whose answer column defines the "
                             "closed label space. Defaults to --test_parquet.")
    parser.add_argument("--disable_code_filter", action="store_true",
                        help="Disable closed-set filtering of predicted codes.")
    parser.add_argument(
        "--adapter_paths", nargs="+", required=True,
        help="List of adapter paths. Use 'NONE' for base-only. "
             "Each produces one evaluation.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    import pandas as pd
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    overhead = chat_template_overhead(tokenizer, enable_thinking=False)
    print(f"Chat-template overhead: {overhead} tokens")

    print(f"Loading test parquet: {args.test_parquet}")
    df = pd.read_parquet(args.test_parquet)
    print(f"Test samples: {len(df)}")

    allowed_codes = None
    if not args.disable_code_filter:
        using_default_label_space = (
            args.allowed_codes_file is None and args.allowed_codes_parquet is None
        )
        allowed_codes = load_allowed_codes(
            allowed_codes_file=args.allowed_codes_file,
            allowed_codes_parquet=args.allowed_codes_parquet or args.test_parquet,
        )
        if allowed_codes is not None:
            print(f"Closed-set code filter enabled: {len(allowed_codes)} allowed codes")
            if using_default_label_space:
                print(
                    "Default behavior matches Edin et al. (2023): closed-set "
                    "label space is derived from the test split. To override, "
                    "pass an explicit code list via --allowed_codes_parquet or "
                    "--allowed_codes_file."
                )
    else:
        print("Closed-set code filter disabled")

    prompts, kept = render_prompts(df, tokenizer, args.max_prompt_length, overhead)

    max_model_len = args.max_prompt_length + args.max_tokens

    all_metrics = []
    for adapter in args.adapter_paths:
        tag = "sft_base" if adapter.upper() == "NONE" else (
            os.path.basename(os.path.dirname(os.path.dirname(adapter.rstrip("/"))))
            or os.path.basename(adapter.rstrip("/"))
        )
        adapter_arg = None if adapter.upper() == "NONE" else adapter
        metrics = evaluate_one(
            tag, args.base_model, adapter_arg, df, prompts, kept,
            args.num_gpus, max_model_len, args.max_tokens, args.output_dir,
            allowed_codes,
        )
        all_metrics.append(metrics)

        # Print per-ckpt summary
        print(f"\n[{tag}] metrics:")
        for k in ("mastered", "mastered_pct", "avg_missed", "avg_pred",
                  "micro_p", "micro_r", "micro_f1",
                  "macro_p", "macro_r", "macro_f1", "off_label_total"):
            v = metrics[k]
            if isinstance(v, float):
                print(f"    {k:<15}: {v:.4f}")
            else:
                print(f"    {k:<15}: {v}")

    # ── Comparison table ─────────────────────────────────────────────
    print("\n\n" + "=" * 110)
    print("COMPARISON TABLE")
    print("=" * 110)
    cols = ["tag", "mastered", "mastered_pct", "avg_missed",
            "micro_p", "micro_r", "micro_f1",
            "macro_p", "macro_r", "macro_f1"]
    header = "  ".join(f"{c:>13}" for c in cols)
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        row = []
        for c in cols:
            v = m.get(c, "")
            if isinstance(v, float):
                row.append(f"{v:>13.4f}")
            elif isinstance(v, int):
                row.append(f"{v:>13d}")
            else:
                row.append(f"{str(v):>13}")
        print("  ".join(row))

    # Dump summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        # Strip per_code stats from summary (too big)
        lean = [{k: v for k, v in m.items() if k != "per_code"} for m in all_metrics]
        json.dump(lean, f, indent=2)
    print(f"\nSummary written to {summary_path}")

    # Also dump per-code stats separately
    per_code_path = os.path.join(args.output_dir, "per_code_stats.json")
    with open(per_code_path, "w") as f:
        json.dump({m["tag"]: m["per_code"] for m in all_metrics}, f, indent=2)
    print(f"Per-code stats written to {per_code_path}")


if __name__ == "__main__":
    main()
