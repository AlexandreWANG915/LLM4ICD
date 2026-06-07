"""Parse CMS ICD-9-CM v32 (final) descriptions into a dotted-code lookup.

Source: https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/Downloads/ICD-9-CM-v32-master-descriptions.zip

CMS publishes 4 fixed-width text files in that zip; this script consumes the
two LONG variants (which match the wording the model is trained on).

Input files (in `--source-dir`, default /tmp/cms_icd9):
    CMS32_DESC_LONG_DX.txt   — diagnosis codes, ~14.5k entries
    CMS32_DESC_LONG_SG.txt   — surgical/procedure codes, ~3.9k entries

Output JSON file (`--out`):
    {"401.9": "Unspecified essential hypertension",
     "33.24": "Closed [endoscopic] biopsy of bronchus",
     ...}

We store the dotted form (e.g. `401.9`, not `4019`) because the reward
function and hint rendering normalize predictions to the dotted form. Both DX
and SG codes share one flat namespace — collisions are not possible because
DX codes are 3-5 chars (after dot insertion) and SG codes are 2-4 chars,
and they use different prefix patterns (V/E vs pure-numeric vs short-numeric).

Usage:
    python tasks/prepare_icd9_descriptions.py \
        --source-dir /tmp/cms_icd9 \
        --out ${ICD9_DESC_PATH:-${REPO_ROOT}/data/icd9_descriptions.json}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def insert_dx_dot(code: str) -> str:
    """Insert decimal point in an ICD-9-CM diagnosis code.

    Rules (per CMS / ICD-9-CM Tabular convention):
      - V codes:       dot after position 3 if length > 3   (V1582 → V15.82)
      - E codes:       dot after position 4 if length > 4   (E9109 → E910.9)
      - numeric:       dot after position 3 if length > 3   (4019  → 401.9 ;
                                                            25000 → 250.00 ;
                                                            412   → 412)

    Returns the code unchanged if it's at or below the no-dot threshold.
    """
    if not code:
        return code
    head = code[0].upper()
    if head == "V":
        return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"
    if head == "E":
        return code if len(code) <= 4 else f"{code[:4]}.{code[4:]}"
    # numeric (and any other unexpected prefix — keep behavior conservative)
    return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"


def insert_sg_dot(code: str) -> str:
    """Insert decimal point in an ICD-9-CM procedure code.

    Procedures are 2-4 chars; dot after position 2 if length > 2.
    Examples: 3324 → 33.24, 9904 → 99.04, 00 → 00.
    """
    if not code:
        return code
    return code if len(code) <= 2 else f"{code[:2]}.{code[2:]}"


# Lines look like:
#   "0010  Cholera due to vibrio cholerae"   (DX, two spaces after code)
#   "3324 Closed [endoscopic] biopsy of bronchus"   (SG, one space)
# Tolerate either by splitting on the first whitespace run.
_LINE_RE = re.compile(r"^([A-Z0-9]{2,5})\s+(.+?)\s*$")


def parse_cms_file(path: Path, dot_inserter) -> dict[str, str]:
    """Read a CMS DESC_LONG_*.txt and return {dotted_code: description}.

    Source files are ISO-8859 / ASCII; latin-1 decodes both losslessly.
    """
    out: dict[str, str] = {}
    with open(path, "r", encoding="latin-1") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            m = _LINE_RE.match(raw)
            if not m:
                # Malformed lines indicate a format change — fail loudly so
                # the JSON output isn't silently incomplete.
                raise ValueError(
                    f"{path.name}:{line_no} unrecognised format: {raw!r}"
                )
            raw_code, desc = m.group(1), m.group(2)
            dotted = dot_inserter(raw_code)
            if dotted in out and out[dotted] != desc:
                raise ValueError(
                    f"{path.name}:{line_no} duplicate code "
                    f"{dotted!r} with conflicting descriptions"
                )
            out[dotted] = desc
    return out


# Used to sanity-check the parser against the project's known top-50
# diagnosis + procedure code list. If any of these are missing from the
# parsed dict the dot-insertion logic almost certainly broke.
_TOP50_SAMPLE_CODES = [
    "401.9",   # Unspecified essential hypertension
    "250.00",  # Diabetes mellitus type II / unspec, no complication
    "428.0",   # Congestive heart failure, unspecified
    "412",     # Old myocardial infarction (3-digit, no dot)
    "V15.82",  # Personal history of tobacco use
    "V58.61",  # Long-term use of anticoagulants
    "33.24",   # Closed [endoscopic] biopsy of bronchus
    "38.93",   # Venous catheterization, NEC
    "99.04",   # Transfusion of packed cells
]


def main() -> None:
    # Default --out lands at <repo_root>/data/icd9_descriptions.json so the
    # generated file matches what verl/utils/icd_descriptions.py expects.
    repo_root = Path(__file__).resolve().parents[2]   # tasks/ → EasyR1/ → repo
    default_out = repo_root / "data" / "icd9_descriptions.json"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/tmp/cms_icd9"),
        help="directory containing CMS32_DESC_LONG_{DX,SG}.txt",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="destination JSON file (flat {code: description} dict)",
    )
    args = ap.parse_args()

    dx_path = args.source_dir / "CMS32_DESC_LONG_DX.txt"
    sg_path = args.source_dir / "CMS32_DESC_LONG_SG.txt"
    for p in (dx_path, sg_path):
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p}; download CMS ICD-9-CM v32 zip and unzip into "
                f"{args.source_dir}"
            )

    dx = parse_cms_file(dx_path, insert_dx_dot)
    sg = parse_cms_file(sg_path, insert_sg_dot)

    # Defensive: confirm DX and SG namespaces don't actually collide.
    overlap = set(dx) & set(sg)
    if overlap:
        raise ValueError(
            f"DX/SG code collision: {sorted(overlap)[:5]}... "
            f"(first 5 of {len(overlap)})"
        )

    merged: dict[str, str] = {}
    merged.update(dx)
    merged.update(sg)

    # Spot-check known top-50 codes — fast feedback if dot logic regresses.
    missing = [c for c in _TOP50_SAMPLE_CODES if c not in merged]
    if missing:
        raise RuntimeError(
            f"top-50 sanity codes missing from merged dict: {missing}. "
            f"Dot-insertion rules likely broke."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(merged, f, ensure_ascii=False, sort_keys=True, indent=2)

    print(
        f"Wrote {len(merged):,} ICD-9-CM descriptions to {args.out} "
        f"(DX: {len(dx):,}, SG: {len(sg):,})"
    )
    print("Spot-check (top-50 samples):")
    for c in _TOP50_SAMPLE_CODES:
        print(f"  {c:>8s}  {merged[c]}")


if __name__ == "__main__":
    main()
