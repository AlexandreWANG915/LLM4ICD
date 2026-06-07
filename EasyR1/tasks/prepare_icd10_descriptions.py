"""Merge two PLM-ICD ICD-10 description sources into a flat lookup JSON.

Produces ${REPO_ROOT}/data/icd10_descriptions.json with the same flat
{code: description} schema our verl/utils/icd_descriptions.py uses for
ICD-9. Covers both ICD-10-CM (diagnoses) and ICD-10-PCS (procedures).

Sources (both from the PLM-ICD project — public domain code descriptions
extracted from CMS / WHO releases):
  --all-json     dict {code: {desc, synonyms, full_text}}     ~46.9k codes (CM only)
  --mimic4-jsonl one JSON object per line, fields code/description/code_type
                                                              ~7.9k codes incl. PCS

The all/ source is used as the base; mimic4/ supplements it with codes
not present (typically PCS procedures + a few CM dotted variants).

Usage:
    python EasyR1/tasks/prepare_icd10_descriptions.py \\
        --all-json /path/to/all/code_descriptions.json \\
        --mimic4-jsonl /path/to/mimic4/icd10_code_descriptions.jsonl
    # Default --out lands at <repo>/data/icd10_descriptions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Sample top-10 codes used as a sanity check. Mix of ICD-10-CM diagnoses,
# trailing-dot variants, and one ICD-10-PCS procedure (02HV33Z).
_TOP10_SANITY = [
    "D62.",     # Acute posthemorrhagic anemia (trailing-dot variant)
    "D64.9",    # Anemia, unspecified
    "E03.9",    # Hypothyroidism, unspecified
    "E11.22",   # Type 2 diabetes mellitus with diabetic CKD
    "E11.9",    # Type 2 diabetes mellitus without complications
    "I10.",     # Essential hypertension (trailing-dot variant)
    "Z23.",     # Encounter for immunization (trailing-dot variant)
    "Z66.",     # Do not resuscitate (trailing-dot variant)
    "02HV33Z",  # Insertion of infusion device into superior vena cava (PCS)
    "5A1D70Z",  # Performance of urinary filtration (PCS)
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]   # tasks/ → EasyR1/ → repo
    default_out = repo_root / "data" / "icd10_descriptions.json"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--all-json",
        type=Path,
        required=True,
        help="PLM-ICD all/code_descriptions.json (dict {code: {desc, ...}}); "
             "comprehensive ICD-10-CM coverage (~46.9k entries).",
    )
    ap.add_argument(
        "--mimic4-jsonl",
        type=Path,
        required=True,
        help="PLM-ICD mimic4/icd10_code_descriptions.jsonl (one record per "
             "line); supplements all/ with PCS procedures and dotted variants.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="destination flat-dict JSON (default: <repo>/data/icd10_descriptions.json)",
    )
    args = ap.parse_args()

    if not args.all_json.exists():
        raise FileNotFoundError(f"--all-json not found: {args.all_json}")
    if not args.mimic4_jsonl.exists():
        raise FileNotFoundError(f"--mimic4-jsonl not found: {args.mimic4_jsonl}")

    src_all = json.loads(args.all_json.read_text())
    flat: dict[str, str] = {
        code: v["desc"] for code, v in src_all.items() if v.get("desc")
    }
    n_base = len(flat)

    n_added = 0
    with open(args.mimic4_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            code = r.get("code")
            desc = (r.get("description") or "").strip()
            if not (code and desc):
                continue
            # Don't overwrite all/ entries (they have curated synonyms /
            # canonical wording); only add what's missing.
            if code not in flat:
                flat[code] = desc
                n_added += 1

    # Spot-check sanity codes — fast feedback if a source format changes
    # upstream and our merge silently loses entries.
    missing = [c for c in _TOP10_SANITY if c not in flat]
    if missing:
        raise RuntimeError(
            f"Sanity-check codes missing from merged dict: {missing}. "
            f"Verify the input sources still have the expected schema."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(flat, f, ensure_ascii=False, sort_keys=True, indent=2)

    print(
        f"Wrote {len(flat):,} ICD-10 descriptions to {args.out} "
        f"(base from all/: {n_base:,}, +{n_added:,} from mimic4/)"
    )
    print("Spot-check:")
    for c in _TOP10_SANITY:
        print(f"  {c:>10s}  {flat[c]}")


if __name__ == "__main__":
    main()
