# Data preparation

This repo contains code, training scripts, and the public CMS ICD-9 dictionary — but **no MIMIC notes or anything derived from them**, because MIMIC-III/IV are PhysioNet-credentialed datasets that cannot be redistributed.

This page lays out the complete data flow for both stages and tells you exactly which files you need to put where before training.

---

## Data flow at a glance

```
┌────────────────────────────────┐  PhysioNet DUA + credentialing required
│ Raw MIMIC-III (or MIMIC-IV)    │
│   NOTEEVENTS.csv               │
│   DIAGNOSES_ICD.csv            │
│   PROCEDURES_ICD.csv           │
└──────────────┬─────────────────┘
               │  Stage 0: medical-coding-reproducibility prep
               │  (not shipped — clone + run that repo)
               ▼
┌────────────────────────────────┐
│ Sectioned-notes feather         │   data/raw/mimiciii_icd9_full_with_sections.feather
│ (text + target + split + id)   │   data/raw/mimiciv_icd10_with_sections_v4.feather
└──────────────┬─────────────────┘
               │  Stage 0.5: LlamaFactory/tasks/prepare_naive_sft.py
               ▼
┌────────────────────────────────┐
│ SFT JSON (sharegpt format)     │  → goes into LlamaFactory
│   data/icd_naive_sft_mimic3_   │     for Stage 1 SFT
│   top50/{train,val,test}.json  │
└──────────────┬─────────────────┘
               │  Stage 0.6: EasyR1/tasks/prepare_icd_grpo.py
               ▼
┌────────────────────────────────┐
│ GRPO parquet (problem, answer) │  → goes into EasyR1
│   data/icd_grpo_top50_nothink/ │     for Stage 2 RL (baseline)
│   {train,val,val_small,        │
│    test}.parquet               │
└──────────────┬─────────────────┘
               │  Stage 0.7: EasyR1/tasks/prepare_hint_data.py  (optional)
               ▼
┌────────────────────────────────┐
│ Hint parquet (+ hint_pool col) │  → only needed for the iterative
│   data/icd_grpo_top50_hint/... │     hint-training research extension
└────────────────────────────────┘
```

---

## What this repo ships vs what you need

| Asset | Shipped? | How to get it |
|---|---|---|
| **CMS ICD-9-CM v32 descriptions** (`data/icd9_descriptions.json`) | ✅ Yes (CMS public domain) | Already in repo. Optional regeneration via `EasyR1/tasks/prepare_icd9_descriptions.py`. |
| **Top-50 ICD-9 code list** (`data/icd9_top50/ALL_CODES.txt`) | ✅ Yes (just a 50-line text file) | Already in repo. |
| **Raw MIMIC-III notes + ICD assignments** | ❌ No (PhysioNet DUA) | Sign DUA at <https://physionet.org/content/mimiciii/>, download to your machine. |
| **SFT JSONs** (`data/icd_naive_sft_mimic3_*/`, `data/icd_naive_sft_mimic4_*/`) | ❌ No (PHI-derived) | Generate via Stage 0.5 below (after PhysioNet access). |
| **GRPO parquets** (`data/icd_grpo_*/`) | ❌ No (PHI-derived) | Generate via Stage 0 → 0.6 below. |
| **SFT base models** (`models/qwen2.5-1.5b-icd-sft-*`) | ❌ No (large + reproducible) | Train via Stage 1 — see `docs/sft_training.md`. |

> 💡 **Tip**: once the SFT JSONs and SFT base models exist, the GRPO parquets, RL training, and eval are all ~30 min away.

---

## Stage 0 — Raw MIMIC → sectioned-notes feather (upstream, not in this repo)

The very first step takes raw PhysioNet CSVs (NOTEEVENTS, DIAGNOSES_ICD, PROCEDURES_ICD, plus MIMIC-IV-Note's `discharge.csv.gz`) and produces a "sectioned notes" feather: one row per discharge summary with a clean `text`, the assigned `target` codes (deduplicated and split-stable), a `split` column, and a row id.

We don't reimplement this — use the **medical-coding-reproducibility** pipeline:

| Dataset | Recommended pipeline | Output filename |
|---|---|---|
| MIMIC-III ICD-9 (full + top50) | [medical-coding-reproducibility](https://github.com/JoakimEdin/medical-coding-reproducibility)'s `prepare_data/prepare_mimiciii_clean.py` | `mimiciii_icd9_full_with_sections.feather` (~52k rows) |
| MIMIC-IV ICD-10 (full + top50) | [medical-coding-reproducibility](https://github.com/JoakimEdin/medical-coding-reproducibility)'s `prepare_data/prepare_mimiciv.py` | `mimiciv_icd10_with_sections.feather` |

Place the resulting feather files in `data/raw/` (or override via env vars — see Stage 0.5 below).

> ⚠️ **Do not use the `mimiciv_icd10_with_sections_v4.feather` shipped with PLM-ICD.** Its `text` column was populated from MIMIC-IV's *patient-facing* discharge summaries (~5-7k chars per row, structured as `[Discharge Diagnosis] ... WHY WAS I IN THE HOSPITAL?`) rather than the full clinical discharge note (~15k chars per row, the version clinicians actually write). Models trained on it will see a fraction of the clinical content. The medical-coding-reproducibility pipeline above reads the raw `discharge.csv.gz` from MIMIC-IV-Note and builds a feather whose `text` column is the actual full clinical note.

---

## Stage 0.5 — Sectioned feather → SFT JSON (in this repo)

**Script: `LlamaFactory/tasks/prepare_naive_sft.py`**

Reads the feather from Stage 0 and emits LlamaFactory-format `sharegpt` JSON.

```bash
# All four variants in one shot
python LlamaFactory/tasks/prepare_naive_sft.py --dataset all

# Or one at a time
python LlamaFactory/tasks/prepare_naive_sft.py --dataset mimic3_top50
python LlamaFactory/tasks/prepare_naive_sft.py --dataset mimic3
python LlamaFactory/tasks/prepare_naive_sft.py --dataset mimic4_top50
python LlamaFactory/tasks/prepare_naive_sft.py --dataset mimic4
```

**Override input feather paths via env vars** (defaults are `data/raw/mimiciii_icd9_full_with_sections.feather` etc.):

```bash
export MIMIC3_FULL_FEATHER=/somewhere/mimiciii_icd9_full_with_sections.feather
export MIMIC4_FEATHER=/somewhere/mimiciv_icd10_with_sections_v4.feather
export MIMIC4_TOP50_CODES=/somewhere/icd10_top50/ALL_CODES.txt   # only needed for mimic4_top50
python LlamaFactory/tasks/prepare_naive_sft.py --dataset all
```

**Top-50 code lists**:
- MIMIC-III ICD-9: hardcoded inside the script (CAML/Mullenbach), and also at `data/icd9_top50/ALL_CODES.txt`.
- MIMIC-IV ICD-10: read from `${MIMIC4_TOP50_CODES}` (50-line text file).

**SFT JSON format LlamaFactory expects** (one entry per discharge summary):

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "You are a medical coding specialist. ...\n\nDischarge Summary:\n[**full discharge summary**]\n\nOutput ONLY the applicable codes in the format: <code>CODE1, CODE2, ...</code>"
      },
      {
        "from": "gpt",
        "value": "<code>038.9, 38.93, 410.71, 428.0, 584.9, 96.04, 96.6, 96.72</code>"
      }
    ],
    "_meta": {"note_id": "145834", "num_codes": 8, "codes": ["038.9", "38.93", ...]}
  }
]
```

For CAML-style top-50 splits expect roughly **47k train / 1.6k val / 3.4k test**.

After Stage 0.5 you'll have:

```
data/icd_naive_sft_mimic3_top50/{train,val,test}.json
data/icd_naive_sft_mimic3_full/{train,val,test}.json
data/icd_naive_sft_mimic4_top50/{train,val,test}.json   (if --dataset mimic4_top50 / all)
data/icd_naive_sft_mimic4_full/{train,val,test}.json    (if --dataset mimic4 / all)
```

These paths match the entries already registered in `LlamaFactory/data/dataset_info.json` — Stage 1 SFT will pick them up automatically.

---

## Stage 0.6 — SFT JSON → GRPO parquet (in this repo)

**Script: `EasyR1/tasks/prepare_icd_grpo.py`**

Once SFT JSON is on disk, convert to GRPO parquet:

```bash
# Top50 GRPO baseline data
python EasyR1/tasks/prepare_icd_grpo.py \
  --src_dir data/icd_naive_sft_mimic3_top50 \
  --out_dir data/icd_grpo_top50_nothink \
  --icd_version icd9 --code_scope top50 --no_think

# Full-code GRPO baseline data
python EasyR1/tasks/prepare_icd_grpo.py \
  --src_dir data/icd_naive_sft_mimic3_full \
  --out_dir data/icd_grpo_full \
  --icd_version icd9 --code_scope full --no_think

# (optional) split val_small for fast in-training validation
python EasyR1/tasks/make_val_small.py \
  --src_parquet data/icd_grpo_top50_nothink/val.parquet \
  --out_parquet data/icd_grpo_top50_nothink/val_small.parquet \
  --n 1000 --seed 42
```

**Output schema** (every parquet):
| Column | Type | Description |
|---|---|---|
| `problem` | str | Full prompt: system instruction + discharge summary + `Output format:` block |
| `answer`  | str | Comma-separated GT codes, e.g. `"401.9, 250.00, 428.0"` |

> Long discharge summaries are truncated automatically at load time by a
> tiered middle-out scheme (`verl/utils/note_truncate.py`) that preserves
> section headers and the discharge-diagnoses block, so notes longer than
> `data.max_prompt_length` still fit the context window.

---

## Stage 0.7 — Hint parquet (only for the hint-training research extension)

The GRPO baseline does **not** need this. Skip unless you're running `tasks/run_iterative_hint_training_*.sh` (PHI).

```bash
python EasyR1/tasks/prepare_hint_data.py \
  --src_dir data/icd_grpo_top50_nothink \
  --out_dir data/icd_grpo_top50_hint
```

Adds a `hint_pool` column to `train.parquet` (val/val_small/test are copied unchanged so eval has no hint signal).

---

## ICD description dictionaries

Both shipped in this repo (CMS / public-domain descriptions, no PHI):

| File | Codes | Coverage | Source |
|---|---|---|---|
| `data/icd9_descriptions.json` | 18,449 | ICD-9-CM diagnoses (~14.5k) + ICD-9 procedures (~3.9k), dotted form | CMS v32 final (2014), public domain |
| `data/icd10_descriptions.json` | 50,212 | ICD-10-CM diagnoses (~46.9k) + ICD-10-PCS procedures (~3.3k), dotted form | PLM-ICD's published `all/code_descriptions.json` + `mimic4/icd10_code_descriptions.jsonl` (CMS / WHO descriptions) |

Both use the same flat `{code: "description"}` schema, so `verl/utils/icd_descriptions.py:DescriptionLookup` works on either — just point `ICD9_DESC_PATH` at the right file:

```bash
# MIMIC-III (ICD-9) — default
# (no env var needed; DEFAULT_PATH already lands on data/icd9_descriptions.json)

# MIMIC-IV (ICD-10)
export ICD9_DESC_PATH=$REPO_ROOT/data/icd10_descriptions.json
```

(The env var name is historical; the lookup itself is namespace-agnostic.)

### Regenerating the dictionaries

Only do this if you want to upgrade to a newer release of CMS / WHO descriptions.

**ICD-9** — from CMS v32 zip:

```bash
mkdir -p /tmp/cms_icd9 && cd /tmp/cms_icd9
curl -fsSL -o icd9.zip \
  https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/Downloads/ICD-9-CM-v32-master-descriptions.zip
unzip icd9.zip

cd $REPO_ROOT
python EasyR1/tasks/prepare_icd9_descriptions.py
# Writes to $REPO_ROOT/data/icd9_descriptions.json
```

**ICD-10** — from PLM-ICD's two JSON sources:

```bash
cd $REPO_ROOT
python EasyR1/tasks/prepare_icd10_descriptions.py \
  --all-json    /path/to/PLM-ICD/data_v4/icd_knowledge/all/code_descriptions.json \
  --mimic4-jsonl /path/to/PLM-ICD/data_v4/icd_knowledge/mimic4/icd10_code_descriptions.jsonl
# Writes to $REPO_ROOT/data/icd10_descriptions.json
```

The two PLM-ICD JSONs together provide both ICD-10-CM (from the comprehensive `all/`) and ICD-10-PCS procedures (from the MIMIC-IV-specific jsonl). They are publicly distributed by the PLM-ICD project; see <https://github.com/MiuLab/PLM-ICD>.

---

## Final checklist before training

| Path | Required for |
|---|---|
| `data/icd_naive_sft_mimic3_top50/train.json` etc. | Stage 1 SFT (LlamaFactory) |
| `data/icd_grpo_top50_nothink/{train,val,val_small,test}.parquet` | Stage 2 RL baselines (top50) |
| `data/icd_grpo_full/{train,val,val_small,test}.parquet` | Stage 2 RL baselines (full-code) |
| `data/icd9_descriptions.json` | Already shipped (CMS public domain) — used for MIMIC-III description-grounded research extension |
| `data/icd10_descriptions.json` | Already shipped (PLM-ICD public domain) — used for MIMIC-IV description-grounded research extension; point `ICD9_DESC_PATH` at it for MIMIC-IV runs |
| `data/icd_grpo_top50_hint/...` | Only for hint-training research extension |
| `models/qwen2.5-1.5b-icd-sft-top50/` | Stage 2 — produced by Stage 1, or pulled from private share |
