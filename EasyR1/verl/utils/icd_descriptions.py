"""Per-code ICD description lookup used to annotate PHI hint prompts.

The JSON file at `DEFAULT_PATH` is produced by tasks/prepare_icd9_descriptions.py
(or prepare_icd10_descriptions.py) from the public-domain CMS / PLM-ICD
releases. It maps each dotted code to its text, e.g. "401.9", "33.24",
"V15.82", spanning both diagnosis and procedure namespaces.

Usage (see the hint rendering in verl/utils/dataset.py):

    lookup = DescriptionLookup.from_default()
    desc = lookup.get("401.9", max_chars=100)

The class is intentionally not a singleton — each training run holds one
instance so description hit/miss rates stay independent across rollout
workers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Resolve relative to the repo root so the lookup works regardless of where
# the user clones the project. Layout: <repo_root>/EasyR1/verl/utils/icd_descriptions.py
# → repo root is three directories up. Override at runtime via the
# ICD9_DESC_PATH env var, or by passing an explicit path to from_default.
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_PATH = os.environ.get(
    "ICD9_DESC_PATH",
    os.path.join(_REPO_ROOT, "data", "icd9_descriptions.json"),
)


def load_descriptions(path: str | Path) -> dict[str, str]:
    """Load the flat {code: description} dict from disk.

    Raises FileNotFoundError if the JSON is missing — caller should surface
    this clearly so the user knows to run prepare_icd9_descriptions.py.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"ICD-9 description file not found at {p}. Run "
            f"`python tasks/prepare_icd9_descriptions.py --out {p}` first."
        )
    with open(p) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected flat dict in {p}; got {type(data).__name__}"
        )
    return data


@dataclass
class DescriptionLookup:
    """Code → description lookup with hit/miss accounting."""

    _table: dict[str, str] = field(default_factory=dict)
    _hits: int = 0
    _misses: int = 0

    @classmethod
    def from_default(cls, path: Optional[str | Path] = None) -> "DescriptionLookup":
        # None / empty string → use the repo-relative DEFAULT_PATH so callers
        # don't need to repeat that constant.
        if not path:
            path = DEFAULT_PATH
        return cls(_table=load_descriptions(path))

    def get(self, code: str, max_chars: Optional[int] = None) -> Optional[str]:
        """Return description for `code` or None on miss.

        Codes are matched case-insensitively after stripping the trailing
        decimal (mirrors the normalisation used by extract_codes /
        _parse_ground_truth_codes in examples/reward_function/icd.py).

        max_chars > 0 truncates the description to that many characters,
        appending an ellipsis when truncation occurs. None or 0 disables
        truncation.
        """
        key = code.strip().rstrip(".").upper()
        desc = self._table.get(key)
        if desc is None:
            self._misses += 1
            return None
        self._hits += 1
        if max_chars and len(desc) > max_chars:
            # Truncate at the last whitespace before the limit so words
            # don't get split mid-token; fall back to hard cut on no
            # whitespace at all.
            cut = desc[:max_chars].rstrip()
            sp = cut.rfind(" ")
            if sp > max_chars * 0.6:
                cut = cut[:sp]
            return cut + "..."
        return desc

    def miss_rate(self) -> float:
        n = self._hits + self._misses
        return (self._misses / n) if n else 0.0

    def stats(self) -> tuple[int, int]:
        """Return (hits, misses) since the last reset_stats call."""
        return self._hits, self._misses

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
