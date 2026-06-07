# MIMIC-III ICD-9 Dataset (50 codes)

This dataset is derived from MIMIC-III discharge summaries for multi-label ICD-9 code classification.

## Dataset Statistics

| Split | Samples |
|-------|---------|
| train | 47,709 |
| val | 1,631 |
| test | 3,372 |
| **Total** | **52,712** |

- **Number of labels**: 50 ICD-9 codes
- **Average labels per sample**: ~16 codes

## File Structure

```
icd9_top50/
├── train.csv       # Training set
├── val.csv         # Validation set
├── test.csv        # Test set
├── ALL_CODES.txt   # List of 50 ICD-9 codes
└── README.md
```

## Data Format

Each CSV file has 2 columns:

| Column | Description |
|--------|-------------|
| `text` | Clinical note text with 6 sections, formatted as `[Section Name]` followed by content |
| `label` | ICD-9 codes separated by semicolons (e.g., `038.9;285.9;584.9`) |

### Text Structure

The text contains 6 sections in order:
1. `[Discharge Diagnosis]`
2. `[Chief Complaint]`
3. `[History of Present Illness]`
4. `[Past Medical History]`
5. `[Physical Exam]`
6. `[Brief Hospital Course]`

## Loading the Dataset

```python
import pandas as pd

train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')

with open('ALL_CODES.txt', 'r') as f:
    all_codes = [line.strip() for line in f if line.strip()]

def parse_labels(label_str):
    return [c.strip() for c in label_str.split(';') if c.strip()]

train_df['label_list'] = train_df['label'].apply(parse_labels)
```
