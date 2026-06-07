"""Prepare naive-SFT JSON for ICD coding (LlamaFactory sharegpt format).

Supports four dataset variants (matching the yaml configs under
examples/icd_sft/):
  --dataset mimic3        → data/icd_naive_sft_mimic3_full/        (ICD-9, ~3000 codes, freq≥10)
  --dataset mimic4        → data/icd_naive_sft_mimic4_full/        (ICD-10, ~7000 codes)
  --dataset mimic3_top50  → data/icd_naive_sft_mimic3_top50/       (ICD-9, CAML/Mullenbach top-50)
  --dataset mimic4_top50  → data/icd_naive_sft_mimic4_top50/       (ICD-10, top-50)
  --dataset all

Each output directory gets train.json / val.json / test.json in sharegpt
format (`conversations` list with `human` prompt + `gpt` <code>...</code>
target).

Input dependency: pre-built feather files containing cleaned discharge
summaries with `text`, `target`, `split`, `_id`/`note_id` columns.
We don't ship the feathers (they are MIMIC-derived PHI). To produce them,
run the upstream prep pipelines:
  - MIMIC-III ICD-9: hdt's prepare_mimiciii_sections.py, or
                     medical-coding-reproducibility/prepare_data/prepare_mimiciii_clean.py
  - MIMIC-IV ICD-10: medical-coding-reproducibility/prepare_data/prepare_mimiciv.py
                     (output: mimiciv_icd10_with_sections_v4.feather)

Override input paths via env vars (defaults are repo-relative under
${REPO_ROOT}/data/raw/):

    MIMIC3_FULL_FEATHER  - MIMIC-III ICD-9 full-code feather
    MIMIC4_FEATHER       - MIMIC-IV ICD-10 feather
    MIMIC4_TOP50_CODES   - MIMIC-IV top-50 ALL_CODES.txt (50 lines, one ICD-10 code per line)

Usage:
    python LlamaFactory/tasks/prepare_naive_sft.py --dataset mimic3_top50
"""

import argparse
import json
import os

import pandas as pd

# Resolve repo root from this file: <root>/LlamaFactory/tasks/this.py
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Override these via env var if your feathers live elsewhere.
MIMIC3_FULL_FEATHER = os.environ.get(
    "MIMIC3_FULL_FEATHER",
    os.path.join(_REPO_ROOT, "data", "raw", "mimiciii_icd9_full_with_sections.feather"),
)
MIMIC4_FEATHER = os.environ.get(
    "MIMIC4_FEATHER",
    os.path.join(_REPO_ROOT, "data", "raw", "mimiciv_icd10_with_sections_v4.feather"),
)
MIMIC4_TOP50_CODES = os.environ.get(
    "MIMIC4_TOP50_CODES",
    os.path.join(_REPO_ROOT, "data", "icd10_top50", "ALL_CODES.txt"),
)

# Mullenbach et al. (CAML) fixed top-50 ICD-9 codes — same list used in
# data/icd9_top50/ALL_CODES.txt.
MIMIC3_MULLENBACH_TOP50 = {
    "401.9", "38.93", "428.0", "427.31", "414.01", "96.04", "96.6", "584.9",
    "250.00", "96.71", "272.4", "518.81", "99.04", "39.61", "599.0", "530.81",
    "96.72", "272.0", "285.9", "88.56", "244.9", "486", "38.91", "285.1",
    "36.15", "276.2", "496", "99.15", "995.92", "V58.61", "507.0", "038.9",
    "88.72", "585.9", "403.90", "311", "305.1", "37.22", "412", "33.24",
    "39.95", "287.5", "410.71", "276.1", "V45.81", "424.0", "45.13", "V15.82",
    "511.9", "37.23",
}

MIMIC3_PROMPT = (
    "You are a medical coding specialist. Given the following discharge summary, "
    "identify all applicable ICD-9 diagnosis and procedure codes.\n\n"
    "Discharge Summary:\n{text}\n\n"
    "Output ONLY the applicable codes in the format: <code>CODE1, CODE2, ...</code>"
)

MIMIC4_PROMPT = (
    "You are a medical coding specialist. Given the following discharge summary, "
    "identify all applicable ICD-10 diagnosis and procedure codes.\n\n"
    "Discharge Summary:\n{text}\n\n"
    "Output ONLY the applicable codes in the format: <code>CODE1, CODE2, ...</code>"
)


def generate_naive_sft(
    feather_path: str,
    prompt_template: str,
    output_dir: str,
    id_field: str,
    code_filter: set | None = None,
) -> None:
    """Generate naive SFT data from a sectioned-notes feather.

    Args:
        code_filter: optional set of codes to keep; if None, all codes are kept.
    """
    if not os.path.exists(feather_path):
        raise FileNotFoundError(
            f"feather not found at {feather_path}. Either set the appropriate "
            f"env var (MIMIC3_FULL_FEATHER / MIMIC4_FEATHER) or run the upstream "
            f"prep pipeline (see this script's module docstring)."
        )
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_feather(feather_path)
    print(f"Loaded {len(df)} rows from {feather_path}")

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        samples = []

        for _, row in split_df.iterrows():
            codes = list(row["target"])
            if code_filter is not None:
                codes = [c for c in codes if c in code_filter]
            if not codes:
                continue

            note_id = str(row[id_field])
            codes_str = ", ".join(sorted(codes))
            sample = {
                "conversations": [
                    {"from": "human", "value": prompt_template.format(text=row["text"])},
                    {"from": "gpt", "value": f"<code>{codes_str}</code>"},
                ],
                "_meta": {
                    "note_id": note_id,
                    "num_codes": len(codes),
                    "codes": sorted(codes),
                },
            }
            samples.append(sample)

        out_path = os.path.join(output_dir, f"{split}.json")
        with open(out_path, "w") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"  {split}: {len(samples):,} samples -> {out_path}")

    print(f"Done: {output_dir}\n")


def load_top50_icd10() -> set[str]:
    if not os.path.exists(MIMIC4_TOP50_CODES):
        raise FileNotFoundError(
            f"MIMIC4_TOP50_CODES file not found at {MIMIC4_TOP50_CODES}; set the "
            f"env var or place the 50-code text file at the default location."
        )
    with open(MIMIC4_TOP50_CODES) as f:
        return set(line.strip() for line in f if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mimic3", "mimic4", "mimic3_top50", "mimic4_top50", "all"],
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=os.path.join(_REPO_ROOT, "data"),
        help="Where to write the icd_naive_sft_*/ directories (default: <repo>/data/)",
    )
    args = parser.parse_args()

    if args.dataset in ("mimic3", "all"):
        print("=== MIMIC-III (ICD-9, full codes, freq>=10) ===")
        generate_naive_sft(
            feather_path=MIMIC3_FULL_FEATHER,
            prompt_template=MIMIC3_PROMPT,
            output_dir=os.path.join(args.out_root, "icd_naive_sft_mimic3_full"),
            id_field="_id",
        )

    if args.dataset in ("mimic4", "all"):
        print("=== MIMIC-IV (ICD-10, full codes) ===")
        generate_naive_sft(
            feather_path=MIMIC4_FEATHER,
            prompt_template=MIMIC4_PROMPT,
            output_dir=os.path.join(args.out_root, "icd_naive_sft_mimic4_full"),
            id_field="note_id",
        )

    if args.dataset in ("mimic3_top50", "all"):
        print("=== MIMIC-III (ICD-9, top-50 Mullenbach) ===")
        generate_naive_sft(
            feather_path=MIMIC3_FULL_FEATHER,
            prompt_template=MIMIC3_PROMPT,
            output_dir=os.path.join(args.out_root, "icd_naive_sft_mimic3_top50"),
            id_field="_id",
            code_filter=MIMIC3_MULLENBACH_TOP50,
        )

    if args.dataset in ("mimic4_top50", "all"):
        print("=== MIMIC-IV (ICD-10, top-50) ===")
        top50 = load_top50_icd10()
        generate_naive_sft(
            feather_path=MIMIC4_FEATHER,
            prompt_template=MIMIC4_PROMPT,
            output_dir=os.path.join(args.out_root, "icd_naive_sft_mimic4_top50"),
            id_field="note_id",
            code_filter=top50,
        )


if __name__ == "__main__":
    main()
