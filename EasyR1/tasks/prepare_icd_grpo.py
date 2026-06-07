"""Convert LlamaFactory SFT JSON (conversations format) into GRPO parquet
with two columns: problem (the FULL rendered prompt) and answer (comma-separated codes).

The prompt written into `problem` contains the entire system instruction, the
discharge summary, and an explicit "Output format:" block. The paired jinja
template should be a passthrough (see examples/format_prompt/icd_passthrough.jinja).

Four parameterization axes control the prompt:
- --icd_version (icd9 | icd10)
- --code_scope  (top50 | full)
- --think / --no_think  (whether to ask for <think>...</think> reasoning)

Example:
  python tasks/prepare_icd_grpo.py \
    --src_dir ${REPO_ROOT}/data/icd_sft_top50 \
    --out_dir ${REPO_ROOT}/data/icd_grpo_mimic3_top50 \
    --icd_version icd9 --code_scope top50 --think
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import pandas as pd

CODE_BLOCK_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)

# SFT boilerplate anchors used to extract the clinical note body.
SUMMARY_MARKER = "Discharge Summary:"
TAIL_MARKER = "Output ONLY the applicable codes"

# Marker used by prepare_hint_data.py to locate where to insert the hint.
FORMAT_BLOCK_HEADER = "Output format:"


def parse_assistant_codes(text: str) -> list[str]:
    """Extract codes from assistant reply like '<code>A, B, C</code>'."""
    m = CODE_BLOCK_RE.search(text)
    payload = m.group(1) if m else text
    codes = [c.strip() for c in re.split(r"[,;\n]", payload) if c.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def strip_sft_boilerplate(text: str) -> str:
    """Keep only the clinical note, dropping the SFT-provided system header and
    the tail "Output ONLY..." instruction."""
    start = text.find(SUMMARY_MARKER)
    body = text[start + len(SUMMARY_MARKER):] if start != -1 else text
    end = body.find(TAIL_MARKER)
    if end != -1:
        body = body[:end]
    return body.strip()


def build_system_line(icd_version: str, code_scope: str) -> str:
    if icd_version not in ("icd9", "icd10"):
        raise ValueError(f"bad icd_version={icd_version}")
    if code_scope not in ("top50", "full"):
        raise ValueError(f"bad code_scope={code_scope}")

    icd_name = "ICD-9" if icd_version == "icd9" else "ICD-10"
    scope_suffix = " from the top-50 most common codes" if code_scope == "top50" else ""
    return (
        f"You are a medical coding specialist. Given the following discharge summary, "
        f"identify all applicable {icd_name} diagnosis and procedure codes{scope_suffix}."
    )


def build_format_block(think: bool) -> str:
    """The explicit final format spec. The header is used as the hint-insertion anchor."""
    if think:
        return (
            f"{FORMAT_BLOCK_HEADER}\n"
            f"<think>your step-by-step reasoning here</think>\n"
            f"<code>CODE1, CODE2, ...</code>"
        )
    return (
        f"{FORMAT_BLOCK_HEADER}\n"
        f"<code>CODE1, CODE2, ...</code>"
    )


def build_prompt(note: str, icd_version: str, code_scope: str, think: bool) -> str:
    return (
        f"{build_system_line(icd_version, code_scope)}\n\n"
        f"Discharge Summary:\n{note}\n\n"
        f"{build_format_block(think)}"
    )


def convert(src_json: str, out_parquet: str, icd_version: str, code_scope: str, think: bool) -> int:
    with open(src_json, "r") as f:
        data = json.load(f)

    rows = []
    for item in data:
        convs = item.get("conversations", [])
        human = next((c["value"] for c in convs if c["from"] == "human"), None)
        gpt = next((c["value"] for c in convs if c["from"] == "gpt"), None)
        if not human or not gpt:
            continue

        codes = parse_assistant_codes(gpt)
        if not codes:
            continue

        note = strip_sft_boilerplate(human)
        prompt = build_prompt(note, icd_version, code_scope, think)
        rows.append({"problem": prompt, "answer": ", ".join(codes)})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    return len(df)


def make_val_small(full_val_parquet: str, small_parquet: str, n: int, seed: int) -> int:
    df = pd.read_parquet(full_val_parquet)
    n = min(n, len(df))
    rng = random.Random(seed)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    sub = df.iloc[idx[:n]].reset_index(drop=True)
    sub.to_parquet(small_parquet, index=False)
    return len(sub)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--icd_version", choices=["icd9", "icd10"], required=True)
    parser.add_argument("--code_scope", choices=["top50", "full"], required=True)
    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument("--think", dest="think", action="store_true")
    think_group.add_argument("--no_think", dest="think", action="store_false")
    parser.set_defaults(think=True)
    parser.add_argument("--val_small_n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)

    print(f"icd_version={args.icd_version}  code_scope={args.code_scope}  think={args.think}")

    for split in ("train", "val", "test"):
        sj = src / f"{split}.json"
        if not sj.exists():
            print(f"skip {split} (not found)")
            continue
        op = out / f"{split}.parquet"
        n = convert(str(sj), str(op), args.icd_version, args.code_scope, args.think)
        print(f"{split}: {n} rows -> {op}")

    val_full = out / "val.parquet"
    if val_full.exists():
        n_small = make_val_small(
            str(val_full),
            str(out / "val_small.parquet"),
            args.val_small_n,
            args.seed,
        )
        print(f"val_small: {n_small} rows -> {out / 'val_small.parquet'}")


if __name__ == "__main__":
    main()
