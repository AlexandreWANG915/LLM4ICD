"""Helpers for closed-set ICD decoding/evaluation.

This module centralizes:
1. Loading the allowed label space from a parquet or explicit code file
2. Filtering predicted codes to that label space
3. Recording which predictions were dropped as off-label

Current use:
- top-50 closed-set eval / inference: derive allowed codes from the target
  parquet by default
- future full-code eval / inference: point the scripts at a full-code parquet,
  or pass an explicit code list file
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def normalize_code(code: str) -> str:
    """Normalize a code token assuming text-like input."""
    return code.strip().rstrip(".").upper()


def parse_answer_codes(answer) -> list[str]:
    if answer is None or pd.isna(answer):
        return []
    answer = str(answer)
    out: list[str] = []
    seen: set[str] = set()
    for part in answer.split(","):
        code = normalize_code(part)
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def load_allowed_codes(
    *,
    allowed_codes_file: str | None = None,
    allowed_codes_parquet: str | None = None,
    answer_column: str = "answer",
) -> set[str] | None:
    """Load the closed label space.

    Priority:
    1. `allowed_codes_file` if given
    2. `allowed_codes_parquet` if given
    3. None (filter disabled)
    """
    if allowed_codes_file:
        return _load_allowed_codes_file(allowed_codes_file)
    if allowed_codes_parquet:
        return _load_allowed_codes_parquet(allowed_codes_parquet, answer_column=answer_column)
    return None


def filter_allowed_codes(
    codes: Iterable[str],
    allowed_codes: set[str] | None,
) -> tuple[list[str], list[str]]:
    """Return (kept_codes, off_label_codes) in original order."""
    seen_kept: set[str] = set()
    seen_off: set[str] = set()
    kept: list[str] = []
    off_label: list[str] = []
    for code in codes:
        norm = normalize_code(code)
        if not norm:
            continue
        if allowed_codes is None or norm in allowed_codes:
            if norm not in seen_kept:
                seen_kept.add(norm)
                kept.append(norm)
        elif norm not in seen_off:
            seen_off.add(norm)
            off_label.append(norm)
    return kept, off_label


def _load_allowed_codes_parquet(parquet_path: str, answer_column: str = "answer") -> set[str]:
    df = pd.read_parquet(parquet_path, columns=[answer_column])
    allowed: set[str] = set()
    for answer in df[answer_column].tolist():
        allowed.update(parse_answer_codes(answer))
    return allowed


def _load_allowed_codes_file(path_str: str) -> set[str]:
    path = Path(path_str)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("codes", [])
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list (or {{\"codes\": [...]}}) in {path}")
        return {normalize_code(c) for c in payload if normalize_code(str(c))}
    return {normalize_code(line) for line in text.splitlines() if normalize_code(line)}
