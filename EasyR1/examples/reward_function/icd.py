import json
import math
import re
from collections import Counter
from typing import Any


# Metadata
REWARD_NAME = "icd"
REWARD_TYPE = "batch"


# Regexes for ICD-9 and ICD-10 code shapes (used only as a fallback when the
# response has no <code>...</code> block). Inside <code>, we trust the
# comma-separated payload and only sanity-check each token.
ICD_PATTERNS = [
    re.compile(r"\b(\d{2,3}\.\d{1,2})\b"),                # ICD-9 numeric: 038.9, 38.93, 96.6
    re.compile(r"\b([VE]\d{2,3}(?:\.\d{1,2})?)\b"),       # ICD-9 V/E: V58.61, V15.82, E910.9
    re.compile(r"\b([A-TV-Z]\d{2}\.\d{1,4}[A-Z0-9]*)\b"), # ICD-10 CM with dot: E11.22, J45.909
    re.compile(r"\b([A-TV-Z]\d{2}\.?)\b"),                # ICD-10 category: I10., D62.
    re.compile(r"\b(\d[A-Z0-9]{6})\b"),                   # ICD-10 PCS 7-char: 02HV33Z
]
# Lightweight sanity check for tokens pulled from inside a <code> block.
ICD_TOKEN_OK = re.compile(r"^[A-Z0-9][A-Z0-9.]{1,9}$")

# Reasoning-closing tags we accept in model output (new data uses <think>,
# older checkpoints trained with <reasoning> still work).
REASONING_CLOSE_TAGS = ("</reasoning>", "</think>")
_CODE_WEIGHT_CACHE: dict[str, dict[str, float]] = {}


def _normalize_code(code: str) -> str:
    return str(code).strip().rstrip(".").upper()


def _parse_ground_truth_codes(ground_truth: str) -> set[str]:
    if ground_truth is None:
        return set()
    text = str(ground_truth)
    if text.strip().lower() == "nan":
        return set()
    codes: set[str] = set()
    for part in text.split(","):
        code = _normalize_code(part)
        if code:
            codes.add(code)
    return codes


def _load_code_weights(path: str | None) -> dict[str, float]:
    """Load optional per-code reward weights from a flat JSON mapping."""
    if path is None:
        return {}
    path = str(path).strip()
    if not path or path.lower() in {"none", "null"}:
        return {}
    if path in _CODE_WEIGHT_CACHE:
        return _CODE_WEIGHT_CACHE[path]

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected code weight JSON object at {path}")

    weights: dict[str, float] = {}
    for code, value in payload.items():
        norm = _normalize_code(code)
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        if norm and math.isfinite(weight) and weight > 0:
            weights[norm] = weight
    _CODE_WEIGHT_CACHE[path] = weights
    return weights


def _sum_code_weights(
    codes: set[str] | frozenset[str],
    code_weights: dict[str, float],
    default: float,
) -> float:
    return sum(code_weights.get(code, default) for code in codes)


def extract_codes(text: str) -> set[str]:
    """Extract ICD codes from model output. Skip reasoning block if present."""
    # Skip any reasoning block by jumping past its closing tag.
    for closing in REASONING_CLOSE_TAGS:
        idx = text.rfind(closing)
        if idx != -1:
            text = text[idx:]
            break

    # Prefer <code>...</code>; trust comma-separated contents there.
    match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
    if match:
        codes: set[str] = set()
        for part in re.split(r"[,;\n]", match.group(1)):
            part = part.strip().rstrip(".").upper()
            if part and ICD_TOKEN_OK.match(part):
                codes.add(part)
        return codes

    # Fallback: regex-scan the text for ICD-shaped tokens.
    codes = set()
    for pat in ICD_PATTERNS:
        codes.update(m.upper() for m in pat.findall(text))
    return codes


def format_reward(response: str) -> float:
    """Reward when response has <reasoning|think>...</...> followed by <code>...</code>."""
    pattern = re.compile(r"<(reasoning|think)>.*?</\1>\s*<code>.*?</code>", re.DOTALL)
    return 1.0 if pattern.search(response) else 0.0


def _fbeta(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def accuracy_reward(response: str, ground_truth: str, beta: float = 1.0) -> dict[str, Any]:
    """Sample-level F-beta between predicted and ground-truth ICD codes.

    Returns fbeta (β-parameterized, used for training reward), f1 (β=1
    constant reference for cross-round comparison), precision, recall, and
    the TP/FP/FN sets so compute_score can aggregate corpus-level metrics.

    Args:
        beta: F-beta parameter. beta=1 is standard F1.
              beta=2 weights recall 2x more than precision.
    """
    pred_codes = extract_codes(response)
    true_codes = _parse_ground_truth_codes(ground_truth)

    if not pred_codes and not true_codes:
        return {
            "fbeta": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0,
            "tp_set": frozenset(), "fp_set": frozenset(), "fn_set": frozenset(),
        }
    if not pred_codes or not true_codes:
        return {
            "fbeta": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0,
            "tp_set": frozenset(),
            "fp_set": frozenset(pred_codes),
            "fn_set": frozenset(true_codes),
        }

    tp_set = frozenset(pred_codes & true_codes)
    fp_set = frozenset(pred_codes - true_codes)
    fn_set = frozenset(true_codes - pred_codes)
    tp = len(tp_set)
    precision = tp / len(pred_codes)
    recall = tp / len(true_codes)

    return {
        "fbeta": _fbeta(precision, recall, beta),
        "f1": _fbeta(precision, recall, 1.0),
        "precision": precision,
        "recall": recall,
        "tp_set": tp_set,
        "fp_set": fp_set,
        "fn_set": fn_set,
    }


def percode_reward(response: str, ground_truth: str, penalty: float = 0.5) -> dict[str, float]:
    """Per-code reward: +1/len(GT) per hit, -penalty*miss/max(pred,GT) per false positive."""
    pred_codes = extract_codes(response)
    true_codes = _parse_ground_truth_codes(ground_truth)

    if not pred_codes and not true_codes:
        return {"percode": 1.0, "precision": 1.0, "recall": 1.0}
    if not true_codes:
        return {"percode": -penalty, "precision": 0.0, "recall": 0.0}
    if not pred_codes:
        return {"percode": 0.0, "precision": 0.0, "recall": 0.0}

    tp = len(pred_codes & true_codes)
    fp = len(pred_codes) - tp
    precision = tp / len(pred_codes)
    recall = tp / len(true_codes)
    hit_score = tp / len(true_codes)
    miss_score = penalty * fp / max(len(pred_codes), len(true_codes))

    return {"percode": hit_score - miss_score, "precision": precision, "recall": recall}


def weighted_fbeta_reward(
    response: str,
    ground_truth: str,
    beta: float = 1.0,
    code_weights: dict[str, float] | None = None,
    code_weight_default: float = 1.0,
) -> dict[str, Any]:
    """F-beta over weighted TP/FP/FN sets.

    This keeps unweighted precision/recall/f1 for ordinary sample logs, but
    uses weighted precision/recall for the scalar training reward. Weighting
    rare codes higher makes a rare-code FN count more than a head-code FN.
    """
    pred_codes = extract_codes(response)
    true_codes = _parse_ground_truth_codes(ground_truth)
    code_weights = code_weights or {}
    default = float(code_weight_default)
    if not math.isfinite(default) or default <= 0:
        default = 1.0

    if not pred_codes and not true_codes:
        return {
            "weighted_fbeta": 1.0,
            "weighted_precision": 1.0,
            "weighted_recall": 1.0,
            "fbeta": 1.0,
            "f1": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "tp_set": frozenset(),
            "fp_set": frozenset(),
            "fn_set": frozenset(),
        }

    tp_set = frozenset(pred_codes & true_codes)
    fp_set = frozenset(pred_codes - true_codes)
    fn_set = frozenset(true_codes - pred_codes)

    tp_w = _sum_code_weights(tp_set, code_weights, default)
    fp_w = _sum_code_weights(fp_set, code_weights, default)
    fn_w = _sum_code_weights(fn_set, code_weights, default)
    weighted_precision = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0.0
    weighted_recall = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0.0

    tp = len(tp_set)
    precision = tp / len(pred_codes) if pred_codes else 0.0
    recall = tp / len(true_codes) if true_codes else 0.0

    return {
        "weighted_fbeta": _fbeta(weighted_precision, weighted_recall, beta),
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "fbeta": _fbeta(precision, recall, beta),
        "f1": _fbeta(precision, recall, 1.0),
        "precision": precision,
        "recall": recall,
        "tp_set": tp_set,
        "fp_set": fp_set,
        "fn_set": fn_set,
    }


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    beta: float = 1.0,
    reward_type: str = "fbeta",
    penalty: float = 0.5,
    code_weight_path: str | None = None,
    code_weight_default: float = 1.0,
) -> list[dict[str, float]]:
    """Compute reward scores for a batch of ICD coding responses.

    Args:
        format_weight: weight for format score in overall reward (0 to disable).
        beta: F-beta parameter for accuracy. 1.0=F1, 2.0=recall-heavy.
        reward_type: "fbeta", "weighted_fbeta", or "percode".
        penalty: penalty weight for false positives (percode only).
        code_weight_path: flat JSON mapping used by weighted_fbeta.
        code_weight_default: weight for codes absent from code_weight_path.
    """
    scores: list[dict[str, float]] = []
    code_weights = _load_code_weights(code_weight_path) if reward_type == "weighted_fbeta" else {}
    # Per-sample TP/FP/FN sets for corpus-level aggregation below.
    tp_per_code: Counter[str] = Counter()
    fp_per_code: Counter[str] = Counter()
    fn_per_code: Counter[str] = Counter()
    tp_total = fp_total = fn_total = 0

    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]

        fmt_score = format_reward(response)

        # Extract TP/FP/FN sets up-front, independent of reward_type. This
        # keeps corpus metrics (micro/macro P/R/F1) valid regardless of which
        # scalar training reward we use — percode_reward doesn't expose
        # set-level info, so if we only pulled sets from accuracy_reward the
        # percode path would silently feed empty sets into the Counter and
        # corpus metrics would degenerate.
        pred_codes = extract_codes(response)
        true_codes = _parse_ground_truth_codes(ground_truth)
        tp_set = frozenset(pred_codes & true_codes)
        fp_set = frozenset(pred_codes - true_codes)
        fn_set = frozenset(true_codes - pred_codes)

        tp_per_code.update(tp_set)
        fp_per_code.update(fp_set)
        fn_per_code.update(fn_set)
        tp_total += len(tp_set)
        fp_total += len(fp_set)
        fn_total += len(fn_set)

        if reward_type == "percode":
            acc = percode_reward(response, ground_truth, penalty=penalty)
            acc_score = acc["percode"]
        elif reward_type == "weighted_fbeta":
            acc = weighted_fbeta_reward(
                response,
                ground_truth,
                beta=beta,
                code_weights=code_weights,
                code_weight_default=code_weight_default,
            )
            acc_score = acc["weighted_fbeta"]
        elif reward_type == "fbeta":
            acc = accuracy_reward(response, ground_truth, beta=beta)
            acc_score = acc["fbeta"]
        else:
            raise ValueError(f"Unknown reward_type={reward_type!r}")
        # f1_sample computed from the independently-extracted sets so it
        # stays consistent across reward_type choices.
        f1_sample = _fbeta(acc["precision"], acc["recall"], 1.0)

        overall = (1 - format_weight) * acc_score + format_weight * fmt_score

        sample_score = {
            "overall": overall,
            "format": fmt_score,
            "accuracy": acc_score,
            "precision": acc["precision"],
            "recall": acc["recall"],
            # f1 = β=1 F-score, constant across rounds even when training
            # reward uses a different β. Enables apples-to-apples comparison.
            "f1": f1_sample,
        }
        if reward_type == "weighted_fbeta":
            sample_score["weighted_precision"] = acc["weighted_precision"]
            sample_score["weighted_recall"] = acc["weighted_recall"]
            sample_score["weighted_fbeta"] = acc["weighted_fbeta"]
        scores.append(sample_score)

    # ── Corpus-level aggregation (broadcast same value to every sample so
    # the trainer's mean-reduction across the batch produces the corpus
    # metric as-is) ─────────────────────────────────────────────────────
    # Micro (pooled TP/FP/FN across all samples/classes) — standard for
    # multi-label classification (PLM-ICD, LAAT, etc. report this).
    if tp_total + fp_total + fn_total == 0:
        micro_p = micro_r = micro_f1 = 1.0  # degenerate empty-batch
    else:
        micro_p = tp_total / max(tp_total + fp_total, 1)
        micro_r = tp_total / max(tp_total + fn_total, 1)
        micro_f1 = 2 * tp_total / max(2 * tp_total + fp_total + fn_total, 1)

    # Macro: per-class P/R/F1 then average across classes. Weights each class
    # equally so performance on rare long-tail codes surfaces — key metric
    # for the hint-driven phase-2 pipeline whose point is learning the tail.
    seen_codes = set(tp_per_code) | set(fp_per_code) | set(fn_per_code)
    per_class_p, per_class_r, per_class_f1 = [], [], []
    for c in seen_codes:
        tp = tp_per_code[c]
        fp = fp_per_code[c]
        fn = fn_per_code[c]
        # Standard convention: if a class has 0 predictions OR 0 GT instances
        # in the batch, its P/R is defined as 0 (sklearn's zero_division=0).
        per_class_p.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        per_class_r.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        denom = 2 * tp + fp + fn
        per_class_f1.append(2 * tp / denom if denom > 0 else 0.0)
    if seen_codes:
        macro_p = sum(per_class_p) / len(per_class_p)
        macro_r = sum(per_class_r) / len(per_class_r)
        macro_f1 = sum(per_class_f1) / len(per_class_f1)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    for s in scores:
        s["precision_micro"] = micro_p
        s["recall_micro"] = micro_r
        s["f1_micro"] = micro_f1
        s["precision_macro"] = macro_p
        s["recall_macro"] = macro_r
        s["f1_macro"] = macro_f1

    return scores
