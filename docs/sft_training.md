# Stage 1 — SFT (LlamaFactory)

GRPO and PHI both start from a Qwen2.5-1.5B (or Qwen3-4B) base that has been
supervised-fine-tuned on note→code demonstrations, so it follows the prompt
template and emits a parseable `<code>...</code>` block. SFT is the largest
single jump in the pipeline.

Recipe (paper Appendix E): **LoRA rank 8 · lr 1e-5 cosine · 3 epochs · AdamW**,
cross-entropy on the assistant response tokens only (the user prompt is masked).

## Configs

Shipped under `LlamaFactory/examples/icd_sft/` — one training config per
dataset × label scope, plus LoRA-merge/export configs:

```
LlamaFactory/examples/icd_sft/
  icd_naive_sft_mimic3_top50.yaml   icd_naive_sft_mimic3_full.yaml
  icd_naive_sft_mimic4_top50.yaml   icd_naive_sft_mimic4_full.yaml
  merge_qwen25_1_5b_top50.yaml      merge_qwen25_1_5b_full.yaml
  merge_qwen3_4b_top50.yaml
```

Each training config reads a dataset registered in
`LlamaFactory/data/dataset_info.json` (e.g. `icd_naive_sft_mimic3_top50_train`),
so drop your Stage-0.5 JSONs into `data/icd_naive_sft_mimic*/` and they resolve
by name. See `docs/data_prep.md`.

## Train, then merge

LlamaFactory trains a LoRA adapter, then exports a merged full model that the RL
stage loads as its base:

```bash
cd LlamaFactory
llamafactory-cli train  examples/icd_sft/icd_naive_sft_mimic3_top50.yaml
llamafactory-cli export examples/icd_sft/merge_qwen25_1_5b_top50.yaml
```

Point the RL stage at the merged model:

```bash
export SFT_MODEL_PATH=$PWD/outputs/qwen2.5-1.5b/icd_naive_sft_mimic3_top50
# full-code base:  export SFT_MODEL_PATH_FULL=...
# Qwen3-4B base:   merge with merge_qwen3_4b_top50.yaml, then export its path
```

(The merge config's `export_dir` is where the base lands; the RL scripts default
to `models/qwen2.5-1.5b-icd-sft-*`, overridable via the env vars above.)

## Sanity check

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('$SFT_MODEL_PATH')
AutoTokenizer.from_pretrained('$SFT_MODEL_PATH')
print(m.config.architectures, m.num_parameters())
"
```
