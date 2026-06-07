#!/usr/bin/env python3
"""Build per-code reward weights from ICD train parquet frequencies."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.icd_label_space import parse_answer_codes  # noqa: E402


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def _summarize(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    return {
        "min": values[0],
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p95": _quantile(values, 0.95),
        "max": values[-1],
    }


def build_weights(
    frequencies: Counter[str],
    *,
    alpha: float,
    min_weight: float,
    max_weight: float,
) -> dict[str, float]:
    if not frequencies:
        raise ValueError("No ICD codes found in train parquet")
    if alpha < 0:
        raise ValueError("--alpha must be non-negative")
    if min_weight <= 0 or max_weight <= 0 or min_weight > max_weight:
        raise ValueError("Require 0 < --min_weight <= --max_weight")

    sorted_freqs = sorted(frequencies.values())
    median_freq = _quantile([float(x) for x in sorted_freqs], 0.50)
    weights: dict[str, float] = {}
    for code, freq in frequencies.items():
        raw = (median_freq / freq) ** alpha if alpha > 0 else 1.0
        weights[code] = round(min(max(raw, min_weight), max_weight), 6)
    return dict(sorted(weights.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--answer_column", default="answer")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min_weight", type=float, default=0.5)
    parser.add_argument("--max_weight", type=float, default=5.0)
    parser.add_argument("--preview_k", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_parquet(args.train_parquet, columns=[args.answer_column])
    frequencies: Counter[str] = Counter()
    for answer in df[args.answer_column].tolist():
        frequencies.update(parse_answer_codes(answer))

    weights = build_weights(
        frequencies,
        alpha=args.alpha,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(weights, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    freq_stats = _summarize([float(v) for v in frequencies.values()])
    weight_stats = _summarize(list(weights.values()))
    print(f"Wrote {len(weights):,} code weights to {out}")
    print(f"Samples: {len(df):,}")
    print("Frequency stats:", json.dumps(freq_stats, sort_keys=True))
    print("Weight stats:", json.dumps(weight_stats, sort_keys=True))

    k = max(args.preview_k, 0)
    if k:
        rare = sorted(frequencies.items(), key=lambda x: (x[1], x[0]))[:k]
        head = sorted(frequencies.items(), key=lambda x: (-x[1], x[0]))[:k]
        print("Lowest-frequency preview:")
        for code, freq in rare:
            print(f"  {code}: freq={freq}, weight={weights[code]}")
        print("Highest-frequency preview:")
        for code, freq in head:
            print(f"  {code}: freq={freq}, weight={weights[code]}")


if __name__ == "__main__":
    main()
