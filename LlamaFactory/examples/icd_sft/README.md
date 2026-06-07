# ICD SFT configs

LoRA SFT recipes that produce the base model EasyR1's GRPO stage starts from. All run via:

```bash
llamafactory-cli train examples/icd_sft/<config>.yaml
```

| Config | Base | Codes | Output |
|---|---|---|---|
| `icd_naive_sft_mimic3_top50.yaml` | Qwen2.5-1.5B | MIMIC-III ICD-9 top-50 | `saves/qwen2.5-1.5b/lora/icd_naive_sft_mimic3_top50` |
| `icd_naive_sft_mimic3_full.yaml`  | Qwen2.5-1.5B | MIMIC-III ICD-9 full (~3k) | `saves/qwen2.5-1.5b/lora/icd_naive_sft_mimic3_full` |
| `icd_naive_sft_mimic4_top50.yaml` | Qwen2.5-1.5B | MIMIC-IV ICD-10 top-50 | `saves/qwen2.5-1.5b/lora/icd_naive_sft_mimic4_top50` |
| `icd_naive_sft_mimic4_full.yaml`  | Qwen2.5-1.5B | MIMIC-IV ICD-10 full | `saves/qwen2.5-1.5b/lora/icd_naive_sft_mimic4_full` |

After training, merge the LoRA adapter into a full checkpoint:

```bash
llamafactory-cli export examples/icd_sft/merge_qwen25_1_5b_top50.yaml
# Output: outputs/qwen2.5-1.5b/icd_naive_sft_mimic3_top50/
# Then point EasyR1 at it:
export SFT_MODEL_PATH=$PWD/outputs/qwen2.5-1.5b/icd_naive_sft_mimic3_top50
```

## Required dataset entries

These configs reference `icd_naive_sft_mimic{3,4}_{top50,full}_{train,val,test}` datasets. The `data/dataset_info.json` overlay in this repo registers them — make sure your MIMIC-derived JSONs land at the relative paths it points to (e.g. `data/icd_sft_top50/train.json`). See top-level `docs/data_prep.md`.

## template note

The `qwen3_nothink` template uses the `enable_thinking=None` mode added by our patch to `src/llamafactory/data/template.py`. That mode extracts any `<think>...</think>` block from the response, drops it onto the prompt side, and zeroes its loss — so the SFT base learns to emit final answers without `<think>` while still seeing reasoning traces during training. If you reset `template.py` to upstream, this template falls back to plain "no-think" behavior.
