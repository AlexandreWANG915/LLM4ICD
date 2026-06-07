# MIMIC-IV ICD-10 Top-50 Dataset

This dataset is derived from MIMIC-IV discharge summaries for multi-label ICD-10 code classification.

## Dataset Statistics

| Split | Samples |
|-------|---------|
| train | 80,668 |
| val | 11,529 |
| test | 23,046 |
| **Total** | **115,243** |

- **Number of labels**: 50 ICD-10 codes
- **Average labels per sample**: ~8 codes

## File Structure

```
icd10_top50/
├── train.csv       # Training set
├── val.csv         # Validation set
├── test.csv        # Test set
├── ALL_CODES.txt   # List of 50 ICD-10 codes
└── README.md
```

## Data Format

Each CSV file has 2 columns:

| Column | Description |
|--------|-------------|
| `text` | Clinical note text with 6 sections, formatted as `[Section Name]` followed by content |
| `label` | ICD-10 codes separated by semicolons (e.g., `K21.9;F41.9;Z87.891`) |

### Text Structure

The text contains 6 sections in order:
1. `[Discharge Diagnosis]`
2. `[Chief Complaint]`
3. `[History of Present Illness]`
4. `[Past Medical History]`
5. `[Physical Exam]`
6. `[Brief Hospital Course]`

Example:
```
[Discharge Diagnosis]
Type 2 diabetes mellitus
Essential hypertension
...

[Chief Complaint]
Chest pain

[History of Present Illness]
Patient is a 65-year-old male...
...
```

## Loading the Dataset

```python
import pandas as pd

# Load data
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')

# Load label list
with open('ALL_CODES.txt', 'r') as f:
    all_codes = [line.strip() for line in f if line.strip()]

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Number of labels: {len(all_codes)}")

# Parse labels
def parse_labels(label_str):
    return [c.strip() for c in label_str.split(';') if c.strip()]

train_df['label_list'] = train_df['label'].apply(parse_labels)
```

## Multi-label Encoding

```python
from sklearn.preprocessing import MultiLabelBinarizer

# Create label encoder
mlb = MultiLabelBinarizer(classes=all_codes)
mlb.fit([all_codes])

# Transform labels to binary matrix
y_train = mlb.transform(train_df['label_list'].tolist())
y_val = mlb.transform(val_df['label_list'].tolist())
y_test = mlb.transform(test_df['label_list'].tolist())

print(f"Label matrix shape: {y_train.shape}")  # (80668, 50)
```

## Evaluation Metrics

We use **Micro** and **Macro** metrics computed at the label level:

```python
import numpy as np
from collections import defaultdict
from typing import List, Set, Dict

def compute_micro_metrics(all_predictions: List[Set[str]],
                          all_ground_truths: List[Set[str]]) -> Dict[str, float]:
    """
    Compute Micro Precision/Recall/F1 (aggregate TP/FP/FN across all samples)
    """
    total_tp, total_fp, total_fn = 0, 0, 0

    for pred, gt in zip(all_predictions, all_ground_truths):
        pred_set = set(pred) if not isinstance(pred, set) else pred
        gt_set = set(gt) if not isinstance(gt, set) else gt

        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        'micro_precision': micro_p,
        'micro_recall': micro_r,
        'micro_f1': micro_f1
    }


def compute_macro_metrics(all_predictions: List[Set[str]],
                          all_ground_truths: List[Set[str]],
                          all_codes: List[str]) -> Dict[str, float]:
    """
    Compute Macro Precision/Recall/F1 (average per-label metrics)
    """
    code_tp = defaultdict(int)
    code_fp = defaultdict(int)
    code_fn = defaultdict(int)

    for pred, gt in zip(all_predictions, all_ground_truths):
        pred_set = set(pred) if not isinstance(pred, set) else pred
        gt_set = set(gt) if not isinstance(gt, set) else gt

        for code in pred_set & gt_set:
            code_tp[code] += 1
        for code in pred_set - gt_set:
            code_fp[code] += 1
        for code in gt_set - pred_set:
            code_fn[code] += 1

    precisions = []
    recalls = []
    f1s = []

    for code in all_codes:
        tp = code_tp.get(code, 0)
        fp = code_fp.get(code, 0)
        fn = code_fn.get(code, 0)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0

    return {
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1
    }
```

### Example Usage

```python
# Assume you have predictions as list of code sets
# predictions: List[Set[str]] - predicted codes for each sample
# ground_truths: List[Set[str]] - ground truth codes for each sample

# Convert from label strings
ground_truths = [set(parse_labels(label)) for label in test_df['label']]
predictions = [...]  # Your model predictions

# Compute metrics
micro = compute_micro_metrics(predictions, ground_truths)
macro = compute_macro_metrics(predictions, ground_truths, all_codes)

print(f"Micro - P: {micro['micro_precision']:.4f}, R: {micro['micro_recall']:.4f}, F1: {micro['micro_f1']:.4f}")
print(f"Macro - P: {macro['macro_precision']:.4f}, R: {macro['macro_recall']:.4f}, F1: {macro['macro_f1']:.4f}")
```

## Notes

1. **PHI Protection**: Patient identifiers are replaced with `___` placeholders
2. **Multi-label**: Each sample can have multiple ICD-10 codes (average ~8 per sample)
3. **Threshold**: For probability-based models, tune the classification threshold on validation set

## Citation

If you use this dataset, please cite the MIMIC-IV database:

```bibtex
@article{johnson2023mimic,
  title={MIMIC-IV, a freely accessible electronic health record dataset},
  author={Johnson, Alistair EW and Bulgarelli, Lucas and Shen, Lu and others},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={1},
  year={2023}
}
```
