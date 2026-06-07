"""Unit tests for new F1 / macro-F1 metrics in reward_function/icd.py."""

import os
import sys

# IMPORTANT: examples/reward_function/ contains a `math.py` (upstream
# EasyR1's math-reasoning reward function). Once that directory is on
# sys.path, any `import math` resolves to the LOCAL file instead of stdlib,
# which crashes any downstream code that uses real math (sympy / mpmath /
# transformers). icd.py does `import math` at module load, so we must
# pre-cache stdlib `math` in sys.modules BEFORE adding the directory.
import math as _stdlib_math  # forces sys.modules['math'] = stdlib math
_ = _stdlib_math.log         # touch it so the import isn't dead-code-eliminated

_EASYR1_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _EASYR1_ROOT)
sys.path.insert(0, os.path.join(_EASYR1_ROOT, "examples", "reward_function"))

from icd import compute_score, accuracy_reward


def test_sample_level_f1():
    # Perfect match (use real-looking ICD codes — token regex needs 2+ chars)
    acc = accuracy_reward("<code>401.9, 250.00</code>", "401.9, 250.00", beta=2.0)
    assert acc["fbeta"] == 1.0
    assert acc["f1"] == 1.0
    assert acc["tp_set"] == frozenset({"401.9", "250.00"})
    assert acc["fp_set"] == frozenset()
    assert acc["fn_set"] == frozenset()

    # Partial — 1 TP, 1 FP, 1 FN
    acc = accuracy_reward("<code>401.9, V15.82</code>", "401.9, 250.00", beta=2.0)
    # P=0.5, R=0.5 → F1=0.5, F2 also 0.5
    assert acc["f1"] == 0.5
    assert acc["precision"] == 0.5
    assert acc["recall"] == 0.5
    assert acc["tp_set"] == frozenset({"401.9"})
    assert acc["fp_set"] == frozenset({"V15.82"})
    assert acc["fn_set"] == frozenset({"250.00"})

    # Recall-heavy: beta=2 upweights recall
    acc = accuracy_reward("<code>401.9, 250.00, V15.82, 311, 272.4</code>",
                           "401.9, 250.00", beta=2.0)
    # P = 2/5, R = 2/2 = 1.0. F1 = 2*0.4*1/1.4 ≈ 0.571
    # F2 = 5*0.4*1/(4*0.4 + 1) = 2.0/2.6 ≈ 0.769
    assert abs(acc["f1"] - 0.5714) < 0.01
    assert abs(acc["fbeta"] - 0.7692) < 0.01
    print("  ✓ sample-level f1/fbeta independence")


def test_corpus_f1_micro_macro():
    """Build a 3-sample batch, verify micro and macro F1 match manual calc."""
    # Use codes: 401.9, 250.00, V15.82, 311, 272.4 (real ICD-9 shapes)
    inputs = [
        # Sample 0: perfect match on 401.9, 250.00
        {"response": "<code>401.9, 250.00</code>",
         "ground_truth": "401.9, 250.00"},
        # Sample 1: TP=401.9, FP=V15.82, FN=250.00
        {"response": "<code>401.9, V15.82</code>",
         "ground_truth": "401.9, 250.00"},
        # Sample 2: TP=311, FN=272.4
        {"response": "<code>311</code>",
         "ground_truth": "311, 272.4"},
    ]
    scores = compute_score(inputs, format_weight=0.0, beta=1.0)

    # Totals across batch:
    # TP: 401.9 (×2), 250.00 (×1), 311 (×1) → 4
    # FP: V15.82 (×1) → 1
    # FN: 250.00 (×1), 272.4 (×1) → 2
    # micro P = 4/5 = 0.8, R = 4/6 ≈ 0.6667, F1 = 2*4/(8+1+2) = 8/11 ≈ 0.7273
    assert abs(scores[0]["precision_micro"] - 0.8) < 1e-6
    assert abs(scores[0]["recall_micro"] - 4/6) < 1e-6
    assert abs(scores[0]["f1_micro"] - 8/11) < 1e-6

    # Per-class P/R/F1:
    # 401.9:  TP=2, FP=0, FN=0 → P=1, R=1, F1=1.0
    # 250.00: TP=1, FP=0, FN=1 → P=1, R=0.5, F1=2/3
    # 311:    TP=1, FP=0, FN=0 → P=1, R=1, F1=1.0
    # 272.4:  TP=0, FP=0, FN=1 → P=0, R=0, F1=0.0
    # V15.82: TP=0, FP=1, FN=0 → P=0, R=0, F1=0.0
    # macro_P  = (1+1+1+0+0)/5 = 0.6
    # macro_R  = (1+0.5+1+0+0)/5 = 0.5
    # macro_F1 = (1+2/3+1+0+0)/5 ≈ 0.5333
    expected_macro_p = (1.0 + 1.0 + 1.0 + 0.0 + 0.0) / 5
    expected_macro_r = (1.0 + 0.5 + 1.0 + 0.0 + 0.0) / 5
    expected_macro_f1 = (1.0 + 2/3 + 1.0 + 0.0 + 0.0) / 5
    assert abs(scores[0]["precision_macro"] - expected_macro_p) < 1e-6
    assert abs(scores[0]["recall_macro"] - expected_macro_r) < 1e-6
    assert abs(scores[0]["f1_macro"] - expected_macro_f1) < 1e-6

    # Broadcast: all 3 samples should have IDENTICAL corpus values
    corpus_keys = ["f1_micro", "f1_macro", "precision_micro", "recall_micro",
                    "precision_macro", "recall_macro"]
    for i in range(1, 3):
        for k in corpus_keys:
            assert scores[i][k] == scores[0][k], f"broadcast mismatch: {k}"

    # Sanity: per-sample f1 for each
    # Sample 0: P=R=1 → F1=1
    # Sample 1: P=0.5, R=0.5 → F1=0.5
    # Sample 2: P=1, R=0.5 → F1=2*1*0.5/1.5 = 0.667
    assert scores[0]["f1"] == 1.0
    assert scores[1]["f1"] == 0.5
    assert abs(scores[2]["f1"] - 2/3) < 1e-6

    print("  ✓ micro F1 pooled correctly")
    print("  ✓ macro F1 per-class averaged correctly")
    print("  ✓ corpus metrics broadcast to all samples")
    print("  ✓ per-sample f1 independent of broadcast")


def test_training_reward_unchanged():
    """Overall reward must still be (1-fw) * F_beta + fw * format — NOT f1."""
    inputs = [{"response": "<code>401.9, 250.00, V15.82</code>",
               "ground_truth": "401.9, 250.00"}]
    # P=2/3, R=1
    # F_beta=2: F2 = 5*(2/3)*1 / (4*(2/3) + 1) = (10/3)/(11/3) = 10/11
    scores_b2 = compute_score(inputs, format_weight=0.0, beta=2.0)
    expected_f2 = 5 * (2/3) * 1 / (4 * (2/3) + 1)
    assert abs(scores_b2[0]["overall"] - expected_f2) < 1e-6
    assert abs(scores_b2[0]["accuracy"] - expected_f2) < 1e-6

    # F_beta=1: F1 = 2*(2/3)*1 / (2/3 + 1) = (4/3)/(5/3) = 4/5 = 0.8
    scores_b1 = compute_score(inputs, format_weight=0.0, beta=1.0)
    assert abs(scores_b1[0]["overall"] - 0.8) < 1e-6
    # but f1 is ALWAYS 0.8 regardless of beta
    assert abs(scores_b1[0]["f1"] - 0.8) < 1e-6
    assert abs(scores_b2[0]["f1"] - 0.8) < 1e-6  # constant across β!

    print("  ✓ overall reward follows β schedule; f1 stays constant")


def test_percode_reward_type_corpus_metrics():
    """Corpus-level f1_micro/f1_macro must be valid even when training uses
    reward_type='percode'. This guards against the Phase 7 bug where the
    percode branch fed empty sets into the counter."""
    inputs = [
        {"response": "<code>401.9, 250.00</code>", "ground_truth": "401.9, 250.00"},
        {"response": "<code>401.9, V15.82</code>", "ground_truth": "401.9, 250.00"},
        {"response": "<code>311</code>",          "ground_truth": "311, 272.4"},
    ]
    scores_percode = compute_score(inputs, format_weight=0.0, beta=1.0,
                                    reward_type="percode")
    scores_fbeta = compute_score(inputs, format_weight=0.0, beta=1.0,
                                  reward_type="fbeta")

    # overall differs (different training reward), but corpus metrics must match
    # because they depend only on TP/FP/FN sets, not on the chosen scalar reward.
    for key in ("f1_micro", "f1_macro", "precision_micro", "recall_micro",
                "precision_macro", "recall_macro"):
        assert abs(scores_percode[0][key] - scores_fbeta[0][key]) < 1e-9, \
            f"corpus metric {key} diverges between reward_types: " \
            f"percode={scores_percode[0][key]}, fbeta={scores_fbeta[0][key]}"
    # Sanity: corpus f1_micro is nontrivial (not 0)
    assert scores_percode[0]["f1_micro"] > 0.5
    # Sanity: f1 (β=1 constant) also valid under percode
    assert scores_percode[0]["f1"] == 1.0  # sample 0 is perfect

    print("  ✓ percode reward_type preserves corpus metrics")


def test_empty_and_edge_cases():
    # Empty batch
    scores = compute_score([], format_weight=0.0, beta=1.0)
    assert scores == []

    # Empty GT + empty pred
    acc = accuracy_reward("<code></code>", "", beta=1.0)
    assert acc["f1"] == 1.0
    assert acc["tp_set"] == frozenset()

    # Non-empty GT, empty pred
    acc = accuracy_reward("no code block here", "401.9, 250.00", beta=1.0)
    assert acc["f1"] == 0.0
    assert acc["fn_set"] == frozenset({"401.9", "250.00"})
    assert acc["fp_set"] == frozenset()

    print("  ✓ empty/edge cases")


def main():
    print("Running reward metric tests:")
    test_sample_level_f1()
    test_corpus_f1_micro_macro()
    test_training_reward_unchanged()
    test_percode_reward_type_corpus_metrics()
    test_empty_and_edge_cases()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
