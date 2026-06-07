"""Inference on the training parquet to discover each sample's missed ICD codes.

Used by Phase 2 iterative hint refresh: after each epoch, run this to find
out what the current checkpoint fails to predict, and `update_hint_pool.py`
writes those missed codes back into the parquet's `hint_pool` column so the
next epoch's DataLoader can dynamically inject them as hints.

Important: this script does NOT inject hints itself (we want to see model's
independent capability). The parquet's existing `hint_pool` column is
ignored; only `problem` + `answer` are read.

Example:
  python tasks/inference_on_train.py \
      --model_path ${SFT_MODEL_PATH_QWEN3:-${REPO_ROOT}/models/qwen3-4b-icd-sft-top50} \
      --train_parquet ${REPO_ROOT}/data/icd_grpo_top50_hint/train.parquet \
      --output_jsonl tasks/logs/inference_round0.jsonl \
      --num_gpus 4
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import sys
from pathlib import Path

# Ensure verl package is importable when the script is run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks.icd_label_space import filter_allowed_codes, load_allowed_codes, parse_answer_codes
from verl.utils.note_truncate import chat_template_overhead, truncate_for_chat_prompt

# vLLM v1 executor fork()s workers. If the parent ever initializes CUDA
# (transformers/torch might during tokenizer load), the children can't
# re-init CUDA. Forcing spawn avoids "Cannot re-initialize CUDA in forked
# subprocess". Must be done before any CUDA touch.
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import pandas as pd
from tqdm import tqdm


CODE_BLOCK_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)


def sample_id(problem_text: str) -> str:
    """Stable 12-char hex id for each sample's problem text."""
    return hashlib.sha1(problem_text.encode("utf-8")).hexdigest()[:12]


def parse_codes(
    response: str,
    allowed_codes: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Extract comma-separated codes from a <code>...</code> block.
    Falls back to nothing if the tag is missing — such samples are treated
    as "model predicted no codes", so everything in GT counts as missed."""
    m = CODE_BLOCK_RE.search(response)
    if not m:
        return [], []
    codes = []
    seen: set[str] = set()
    for part in re.split(r"[,;\n]", m.group(1)):
        c = part.strip().rstrip(".").upper()
        if c and c not in seen:
            seen.add(c)
            codes.append(c)
    return filter_allowed_codes(codes, allowed_codes)


def parse_gt(answer: str) -> list[str]:
    return parse_answer_codes(answer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Merged model path (for Round 0 SFT) or base model if using --adapter_path")
    parser.add_argument("--adapter_path", default=None,
                        help="Optional LoRA adapter path (from EasyR1 checkpoint global_step_*/actor/)")
    parser.add_argument("--train_parquet", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=5120,
                        help="Prompt budget used for train-time inference. "
                             "Keep this aligned with the training "
                             "data.max_prompt_length unless you intentionally "
                             "want a different truncation regime.")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Generation budget for the response. "
                             "Use a smaller explicit override for top-50 "
                             "experiments if desired.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Greedy decoding by default for determinism")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit samples for debugging; 0 = all")
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768,
                        help="vLLM continuous batching budget per step. "
                             "Higher = more concurrent prompts = faster, "
                             "but more GPU KV cache memory. Default 32768 "
                             "can fit ~5 prompts @ 6144 tokens.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--allowed_codes_file", default=None,
                        help="Optional newline/JSON code list for closed-set "
                             "filtering. If omitted, derive from "
                             "--allowed_codes_parquet or --train_parquet.")
    parser.add_argument("--allowed_codes_parquet", default=None,
                        help="Optional parquet whose answer column defines the "
                             "closed label space. Defaults to --train_parquet.")
    parser.add_argument("--disable_code_filter", action="store_true",
                        help="Disable closed-set filtering of predicted codes.")
    args = parser.parse_args()

    # Lazy imports so the script can be syntax-checked without vLLM/torch.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"Loading parquet: {args.train_parquet}")
    df = pd.read_parquet(args.train_parquet)
    if args.max_samples > 0:
        df = df.head(args.max_samples).reset_index(drop=True)
    print(f"Total samples: {len(df)}")

    allowed_codes = None
    if not args.disable_code_filter:
        allowed_codes = load_allowed_codes(
            allowed_codes_file=args.allowed_codes_file,
            allowed_codes_parquet=args.allowed_codes_parquet or args.train_parquet,
        )
        if allowed_codes is not None:
            print(f"Closed-set code filter enabled: {len(allowed_codes)} allowed codes")
    else:
        print("Closed-set code filter disabled")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Measure chat-template overhead once so truncate_for_chat_prompt gets
    # the correct budget for the problem text itself.
    template_overhead = chat_template_overhead(tokenizer, enable_thinking=False)
    print(f"Chat-template overhead: {template_overhead} tokens; "
          f"problem budget: {args.max_prompt_length - template_overhead} tokens.")

    print("Rendering chat templates (truncating overlong notes instead of skipping)...")
    prompts: list[str] = []
    kept_indices: list[int] = []
    n_truncated = 0
    n_failed = 0
    first_failure_logged = False
    for i, problem in enumerate(tqdm(df["problem"].tolist())):
        try:
            truncated_problem = truncate_for_chat_prompt(
                problem,
                tokenizer,
                max_prompt_length=args.max_prompt_length,
                template_overhead=template_overhead,
                enable_thinking=False,
            )
            if truncated_problem is not problem:
                n_truncated += 1
        except ValueError as e:
            # Boilerplate exceeds budget or drift too large — unfixable. Skip.
            n_failed += 1
            if not first_failure_logged:
                tqdm.write(f"  sample {i}: truncation failed ({e}); skipping")
                first_failure_logged = True
            continue

        messages = [{"role": "user", "content": truncated_problem}]
        # enable_thinking=False: defensive — for Qwen3 thinking variants the
        # chat template would otherwise prefix <think>, wrecking our short
        # max_tokens budget. Harmless for Qwen3-*-Instruct-2507 whose
        # template ignores this kwarg.
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        prompts.append(text)
        kept_indices.append(i)
    print(f"Kept {len(prompts)}/{len(df)} prompts "
          f"({n_truncated} truncated, {n_failed} failed).")

    print(f"Building vLLM (tp={args.num_gpus})...")
    llm_kwargs = dict(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.num_gpus,
        disable_custom_all_reduce=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_length + args.max_tokens,
        max_num_batched_tokens=args.max_num_batched_tokens,
        disable_log_stats=True,
        # LoRA + CUDA graphs is flaky on some vLLM versions — eager is safer.
        enforce_eager=bool(args.adapter_path),
    )
    lora_req = None
    if args.adapter_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64
        lora_req = LoRARequest("round_adapter", 1, args.adapter_path)
        print(f"LoRA adapter: {args.adapter_path}")

    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=1.0 if args.temperature == 0.0 else 0.9,
        max_tokens=args.max_tokens,
    )

    print("Generating...")
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling,
        lora_request=lora_req,
    )

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    print(f"Writing {args.output_jsonl}")

    total_missed_empty = 0
    total_missed_codes = 0
    total_off_label = 0
    with open(args.output_jsonl, "w") as f:
        for kept_idx, out in zip(kept_indices, outputs):
            row = df.iloc[kept_idx]
            problem = row["problem"]
            response = out.outputs[0].text if out.outputs else ""
            pred, off_label = parse_codes(response, allowed_codes=allowed_codes)
            gt = parse_gt(row["answer"])
            pred_set = set(pred)
            gt_set = set(gt)
            missed = sorted(gt_set - pred_set)
            if not missed:
                total_missed_empty += 1
            total_missed_codes += len(missed)
            total_off_label += len(off_label)
            record = {
                "sample_id": sample_id(problem),
                "pred_codes": sorted(pred_set),
                "off_label_codes": off_label,
                "gt_codes": sorted(gt_set),
                "missed": missed,
                "n_pred": len(pred_set),
                "n_off_label": len(off_label),
                "n_gt": len(gt_set),
                "n_missed": len(missed),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"Done. {total_missed_empty}/{len(df)} samples fully correct "
        f"(missed=[]). Total missed codes across corpus: {total_missed_codes}. "
        f"Filtered off-label predictions: {total_off_label}."
    )


if __name__ == "__main__":
    main()
