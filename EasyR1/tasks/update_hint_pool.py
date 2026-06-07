"""Merge inference results back into the training parquet's `hint_pool` column.

Phase 2 loop:
  1. `inference_on_train.py` produces a JSONL with per-sample `missed` codes
  2. This script reads that JSONL and writes `hint_pool = missed` back into
     the source parquet (atomically).
  3. Optionally also writes a `code_weights.json` with per-code sampling
     weights (= 1/sqrt(gt_freq) * 1/max(recall, 0.05)) so the DataLoader can
     bias hint sampling toward rare-and-hard codes.
  4. DataLoader's dynamic hint injection then samples from the fresh
     hint_pool during the next training epoch.

Samples the model got fully correct (missed=[]) get hint_pool=[], so the
DataLoader (with skip_empty_hint_pool=true) will drop them.

Example:
  python tasks/update_hint_pool.py \
      --inference_jsonl tasks/logs/inference_round0.jsonl \
      --train_parquet ${REPO_ROOT}/data/icd_grpo_top50_hint/train.parquet \
      --code_weights_out tasks/logs/phase2/code_weights_round0.json
"""

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd


def sample_id(problem_text: str) -> str:
    return hashlib.sha1(problem_text.encode("utf-8")).hexdigest()[:12]


def compute_code_weights(records: list[dict], clip_multiplier: float = 3.0) -> dict[str, float]:
    """Per-code weight = 1/sqrt(gt_freq) * 1/max(recall, 0.05), then clip.

    - gt_freq: how often this code appears in any ground-truth across the corpus
    - recall: fraction of GT occurrences the model predicted correctly
    - The weight is proportional to "rare AND hard" — high for codes that are
      both underrepresented in the data and frequently missed by the model.
    - Floor on recall (0.05) avoids division-by-zero blowups.
    - Optional clipping at `clip_multiplier * median(raw_weights)` prevents one
      pathological code from dominating hint sampling.
    """
    gt_freq: Counter[str] = Counter()
    miss_freq: Counter[str] = Counter()
    for r in records:
        gt_freq.update(r.get("gt_codes", []))
        miss_freq.update(r.get("missed", []))

    weights: dict[str, float] = {}
    for code, gf in gt_freq.items():
        recall = (gf - miss_freq.get(code, 0)) / gf
        recall = max(recall, 0.05)
        weights[code] = (1.0 / math.sqrt(gf)) * (1.0 / recall)
    if not weights or clip_multiplier <= 0:
        return weights

    median_weight = statistics.median(weights.values())
    max_weight = clip_multiplier * median_weight
    for code, weight in list(weights.items()):
        weights[code] = min(weight, max_weight)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_jsonl", required=True)
    parser.add_argument("--train_parquet", required=True,
                        help="Path to the parquet; will be overwritten atomically")
    parser.add_argument("--code_weights_out", default=None,
                        help="If set, also dump per-code sampling weights "
                             "(rare × hard) to this JSON path")
    parser.add_argument("--weight_clip_multiplier", type=float, default=3.0,
                        help="Clip code weights to multiplier × median(raw_weights). "
                             "<= 0 disables clipping.")
    args = parser.parse_args()

    # Load inference results: {sample_id: missed_codes}; keep raw records too
    # for code-weight computation.
    records: list[dict] = []
    id_to_missed: dict[str, list[str]] = {}
    with open(args.inference_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
            id_to_missed[rec["sample_id"]] = rec["missed"]
    print(f"Inference records: {len(id_to_missed)}")

    # Compute per-code weights (frequency × recall) for DataLoader use.
    # Defer the write to after parquet update so a parquet-write failure
    # doesn't leave fresh weights paired with stale hint_pool data.
    code_weights = None
    if args.code_weights_out:
        code_weights = compute_code_weights(records, clip_multiplier=args.weight_clip_multiplier)
        # Print top-5 most weighted (rarest × hardest) for visibility.
        top = sorted(code_weights.items(), key=lambda kv: -kv[1])[:5]
        print(f"Computed code weights ({len(code_weights)} codes); "
              f"top-5 rare-and-hard (clip={args.weight_clip_multiplier}x median):")
        for c, w in top:
            print(f"    {c:<10} weight={w:.4f}")

    # Load parquet
    df = pd.read_parquet(args.train_parquet)
    print(f"Parquet rows: {len(df)}")

    # Recompute sample_ids from problem column and attach new hint_pool
    new_hint_pools = []
    missing_from_inference = 0
    old_hint_pools = (
        df["hint_pool"].tolist() if "hint_pool" in df.columns else [None] * len(df)
    )
    for problem, old_pool in zip(df["problem"].tolist(), old_hint_pools):
        sid = sample_id(problem)
        if sid in id_to_missed:
            new_hint_pools.append(id_to_missed[sid])
        else:
            # Parquet row had no matching inference record (e.g. inference was
            # sliced for debugging, or the prompt exceeded the length filter).
            # Keep the prior hint_pool so training doesn't lose signal.
            missing_from_inference += 1
            fallback = list(old_pool) if old_pool is not None else []
            new_hint_pools.append(fallback)

    if missing_from_inference:
        print(
            f"WARNING: {missing_from_inference} parquet rows had no matching "
            f"inference record; kept stale hint_pool for them."
        )

    # Stats on the update
    pool_sizes = [len(p) for p in new_hint_pools]
    size_dist = Counter(pool_sizes)
    n_empty = size_dist.get(0, 0)
    avg_size = sum(pool_sizes) / max(len(pool_sizes), 1)
    print(f"Rows with empty hint_pool (mastered): {n_empty}/{len(df)}")
    print(f"Avg hint_pool size: {avg_size:.2f}")
    print(f"Hint_pool size dist (top 10): "
          f"{sorted(size_dist.items())[:10]}")

    df["hint_pool"] = new_hint_pools

    # Atomic write: write to .tmp, then os.replace
    target = Path(args.train_parquet)
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, target)
    print(f"Wrote {target}")

    # Only write code weights AFTER parquet succeeded — otherwise next round
    # would mix new weights with stale hint_pool.
    if code_weights is not None:
        weights_target = Path(args.code_weights_out)
        weights_target.parent.mkdir(parents=True, exist_ok=True)
        weights_tmp = weights_target.with_suffix(weights_target.suffix + ".tmp")
        with open(weights_tmp, "w") as f:
            json.dump(code_weights, f, indent=2)
        os.replace(weights_tmp, weights_target)
        print(f"Wrote {weights_target}")


if __name__ == "__main__":
    main()
