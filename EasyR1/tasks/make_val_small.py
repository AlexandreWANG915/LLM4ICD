"""Create a smaller validation parquet by random sampling rows from a source parquet."""

from __future__ import annotations

import argparse
import os
import random

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_parquet", required=True)
    parser.add_argument("--out_parquet", required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.src_parquet)
    n = min(args.n, len(df))
    indices = list(range(len(df)))
    random.Random(args.seed).shuffle(indices)
    sub = df.iloc[indices[:n]].reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)
    sub.to_parquet(args.out_parquet, index=False)
    print(f"{len(sub)} rows -> {args.out_parquet}")


if __name__ == "__main__":
    main()
