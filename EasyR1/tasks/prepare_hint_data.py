"""Add a `hint_pool` column to a GRPO parquet so the EasyR1 DataLoader can
dynamically sample random hints on every `__getitem__` read.

For Phase 1: hint_pool = all GT codes parsed from `answer`.
For Phase 2 (iterative refresh): an external script (update_hint_pool.py)
overwrites this column with per-sample missed codes from inference.

Val / val_small / test parquets are copied without `hint_pool`, so the
DataLoader will skip hint injection for evaluation.

Example:
  python tasks/prepare_hint_data.py \
    --src_dir ${REPO_ROOT}/data/icd_grpo_top50_nothink \
    --out_dir ${REPO_ROOT}/data/icd_grpo_top50_hint
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def parse_gt(answer: str) -> list[str]:
    return [c.strip() for c in answer.split(",") if c.strip()]


def write_train_with_hint_pool(src_parquet: str, out_parquet: str) -> int:
    df = pd.read_parquet(src_parquet)
    df["hint_pool"] = df["answer"].map(parse_gt)
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    return len(df)


def copy_no_hint(src_parquet: str, out_parquet: str) -> int:
    df = pd.read_parquet(src_parquet)
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)

    train_src = src / "train.parquet"
    if not train_src.exists():
        raise FileNotFoundError(f"missing {train_src}")
    n = write_train_with_hint_pool(str(train_src), str(out / "train.parquet"))
    print(f"train: {n} rows (hint_pool column added) -> {out / 'train.parquet'}")

    for split in ("val", "val_small", "test"):
        sp = src / f"{split}.parquet"
        if sp.exists():
            m = copy_no_hint(str(sp), str(out / f"{split}.parquet"))
            print(f"{split}: {m} rows (no hint_pool) -> {out / f'{split}.parquet'}")


if __name__ == "__main__":
    main()
