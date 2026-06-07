# LLM4ICD

**Can Post-Training Turn LLMs into Good Medical Coders? An Empirical Study of Generative ICD Coding**

Reproducible code for a controlled study of **task-specific post-training** for
generative ICD coding. We take instruction-tuned LLMs and walk them up a
post-training ladder — **Prompting → SFT → GRPO → PHI** — under a single shared
protocol (same datasets, splits, parser, and metrics), on MIMIC-III (ICD-9-CM)
and MIMIC-IV (ICD-10-CM/PCS), in both the Top-50 and Full label settings.

---

## TL;DR

Automated ICD coding maps a clinical discharge summary to a set of standardized
diagnosis/procedure codes — an extreme multi-label problem over a taxonomy with
tens of thousands of codes. Generative LLMs are often reported as *weak* medical
coders, but that conclusion comes almost entirely from **inference-time** use
(zero/few-shot prompting, CoT, retrieval, tools). This repo asks what happens
when you **post-train** the model instead.

| Stage | What it does | Framework |
|---|---|---|
| **Prompting** | Zero/few-shot ± chain-of-thought. Baseline; near-unusable on its own. | — |
| **SFT** | LoRA fine-tune on note→code demonstrations. Teaches the parseable output schema and the empirical code prior. **The main capability jump.** | LlamaFactory |
| **GRPO** | RL with a **sample-level F1 reward** over the parsed code set. Improves code-set prediction beyond SFT, especially in the Full setting. | EasyR1 + vLLM |
| **PHI** | *Progressive Hint Injection.* Extends GRPO with a missed-code curriculum: codes a checkpoint keeps missing are fed back as stochastic **training-time** hints. Inference stays hint-free. Targeted gains on macro / long-tail. | EasyR1 + vLLM |

**Headline:** prompting-only evaluation substantially underestimates LLM coders.
After post-training they become competitive with strong discriminative PLM
coders — matching or exceeding PLM-ICD in Top-50, and closing much of the gap in
Full.

### Representative results (micro-F1, %)

Qwen3-4B, our pipeline vs. the PLM-ICD reference (full tables in the paper):

| Setting | Best Prompting | SFT | GRPO | PHI | PLM-ICD |
|---|--:|--:|--:|--:|--:|
| MIMIC-III ICD-9, Top-50 | 18.2 | 72.6 | **73.6** | 73.5 | 68.1 |
| MIMIC-III ICD-9, Full   |  5.7 | 45.8 | 56.5 | **56.6** | 59.7 |
| MIMIC-IV ICD-10, Top-50 | 11.7 | 73.0 | **73.6** | 73.4 | 73.5 |
| MIMIC-IV ICD-10, Full   |  2.6 | 52.0 | 58.6 | **58.7** | 59.6 |

PHI's benefit is clearest on **macro-F1** in the Full setting (rare-code recall),
where it consistently edges GRPO; on headline micro-F1 it is close to GRPO by
design (it concentrates training on unresolved missed-code cases).

---

## Repository layout

| Path | Stage | Description |
|---|---|---|
| `LlamaFactory/` | SFT | Upstream fork. Our SFT configs under `examples/icd_sft/`, the note→code converter `tasks/prepare_naive_sft.py`, and a small `data/template.py` patch (`enable_thinking=None` mode). |
| `EasyR1/` | GRPO + PHI | Upstream (verl-based) fork. Our reward `examples/reward_function/icd.py`, prompt templates `examples/format_prompt/icd*.jinja`, training scripts, and the PHI/data tooling under `tasks/`. |
| `data/icd9_descriptions.json` | both | 18,449 ICD-9-CM descriptions (CMS v32, public domain). MIMIC-III runs use this. |
| `data/icd10_descriptions.json` | both | 50,212 ICD-10-CM/PCS descriptions (public domain). MIMIC-IV runs point `ICD9_DESC_PATH` here. |
| `data/icd9_top50/`, `data/icd10_top50/` | both | Top-50 code lists for the Top-50 regimes. |
| `requirements.txt` | both | Minimal top-level deps; installs both sub-frameworks editable and pins the vLLM/torch/flash-attn ABI chain. |
| `requirements.lock.txt` | both | Full `pip freeze` (309 packages) for byte-identical reproduction. |
| `docs/` | — | Setup, data prep, SFT and RL walkthroughs. |

> **No MIMIC-derived data, parquets, or model weights are shipped** (PHI /
> size). See [Data](#data) for how to rebuild them after PhysioNet
> credentialing. The repo ships only public-domain ICD descriptions and the
> Top-50 code lists.

---

## Setup

```bash
# Python 3.11; CUDA 12.x; H100/A100. See docs/setup.md for the full chain.
pip install -r requirements.txt          # installs LlamaFactory + EasyR1 editable
# or, for the exact developer environment:
pip install -r requirements.lock.txt
```

Key versions (ABI-locked): torch 2.6 · vLLM 0.8.2 · flash-attn 2.7.3 ·
transformers 4.56 · peft 0.18 · ray 2.53 · deepspeed 0.16.

---

## Data

This repo ships **no** MIMIC-derived data (it is PHI under the PhysioNet DUA).
What is in vs. out:

| Asset | Shipped? | How to get it |
|---|---|---|
| ICD-9 / ICD-10 description JSONs | ✅ public domain | already in `data/` |
| Top-50 code lists | ✅ | already in `data/` |
| Raw MIMIC-III/IV notes + code assignments | ❌ | PhysioNet credentialing + DUA |
| SFT JSONs / GRPO parquets / SFT base models | ❌ (PHI / large) | rebuild via the pipeline below |

Rebuild chain (each step's script is in this repo or linked from
`docs/data_prep.md`):

| Step | Input | Output | Script |
|---|---|---|---|
| 0 (upstream) | Raw MIMIC CSVs | Sectioned-notes feather | [medical-coding-reproducibility](https://github.com/JoakimEdin/medical-coding-reproducibility) (Edin et al. 2023 clean splits) |
| 0.5 | Feather | SFT JSON (ShareGPT) | `LlamaFactory/tasks/prepare_naive_sft.py --dataset {mimic3,mimic4,mimic3_top50,mimic4_top50}` |
| 0.6 | SFT JSON | GRPO parquet (`problem` + `answer`) | `EasyR1/tasks/prepare_icd_grpo.py --src_dir <sft_dir> --out_dir <grpo_dir>` |
| 0.7 | GRPO parquet | PHI hint parquet (`hint_pool` column) | `EasyR1/tasks/prepare_hint_data.py` |

The Top-50 and Full SFT datasets are pre-registered in
`LlamaFactory/data/dataset_info.json` (MIMIC-III + MIMIC-IV × top50/full ×
train/val/test) — drop your rebuilt JSONs into `data/icd_naive_sft_mimic*/` and
they resolve by name.

---

## Reproducing the pipeline

### Stage 1 — SFT (LlamaFactory)

```bash
cd LlamaFactory
llamafactory-cli train  examples/icd_sft/icd_naive_sft_mimic3_top50.yaml
llamafactory-cli export examples/icd_sft/merge_qwen25_1_5b_top50.yaml
export SFT_MODEL_PATH=$PWD/outputs/qwen2.5-1.5b/icd_naive_sft_mimic3_top50
```

LoRA rank 8 · lr 1e-5 cosine · 3 epochs · AdamW · loss on assistant tokens only.
Configs exist for MIMIC-III/IV × Top-50/Full; Qwen3-4B uses
`merge_qwen3_4b_top50.yaml`. See `docs/sft_training.md`.

### Stage 2 — GRPO baseline (EasyR1)

```bash
export SFT_MODEL_PATH=/path/to/sft-base        # top-50 SFT base
bash EasyR1/examples/qwen2_5_1_5b_icd_grpo_top50_nohint_4gpu.sh
```

GRPO from the SFT policy: G = 8 rollouts/prompt, standardized **sample-level F1**
reward over the parsed `<code>...</code>` set, KL coefficient 1e-3, max response
196 tokens, vLLM rollouts on 4 GPUs.

### Stage 3 — PHI (Progressive Hint Injection)

```bash
export SFT_MODEL_PATH_FULL=/path/to/sft-full-base
CUDA_VISIBLE_DEVICES=0,1,2,3 ROUNDS=3 NUM_GPUS=4 \
  bash EasyR1/tasks/run_iterative_hint_training_full_hintE_desc_4gpu.sh
```

The orchestrator runs the round-0 SFT checkpoint hint-free to seed each
example's missed-code pool `H = Y \ Ŷ`, then alternates: GRPO round with
stochastic description-annotated hints (inject prob 0.5, 1–5 codes, rare/low-recall
weighting, temperature 3.0) → hint-free inference → refresh the pool. The reward
target is always the **full** gold set, not the hinted subset. Inference is
always hint-free. See `docs/rl_training.md`.

### Switching dataset / label scope / backbone

Both training scripts are templates; everything is an env var or a Hydra-style
CLI override — no code edits.

```bash
# MIMIC-IV / ICD-10 instead of MIMIC-III / ICD-9:
export SFT_MODEL_PATH=/path/to/mimic4-sft-base
export ICD9_DESC_PATH=$PWD/data/icd10_descriptions.json   # ICD-10 descriptions
bash EasyR1/examples/qwen2_5_1_5b_icd_grpo_top50_nohint_4gpu.sh \
  data.train_files=$PWD/data/icd_grpo_mimic4_top50/train.parquet \
  data.val_files=$PWD/data/icd_grpo_mimic4_top50/val_small.parquet

# PHI on ICD-10: also tell the hint renderer to use ICD-10 wording + example:
bash EasyR1/tasks/run_iterative_hint_training_full_hintE_desc_4gpu.sh \
  data.hint_icd_version=icd10

# Full label space (longer answers):
bash EasyR1/examples/qwen2_5_1_5b_icd_grpo_top50_nohint_4gpu.sh \
  data.max_response_length=512 \
  data.train_files=$PWD/data/icd_grpo_full/train.parquet
```

Other common overrides: `worker.reward.reward_function_kwargs.beta=2.0`
(recall-heavy Fβ), `worker.actor.model.lora.rank=16`, `worker.rollout.n=4`,
`trainer.total_epochs=5`, `algorithm.kl_coef=0`.

---

## Evaluation

`EasyR1/tasks/eval_checkpoints_on_test.py` evaluates the SFT base and/or any LoRA
checkpoints on the held-out test parquet, reporting micro/macro precision,
recall, and F1 with the same formulas as PLM-ICD / Edin et al. (2023) (macro-F1 =
mean of per-class F1).

```bash
python EasyR1/tasks/eval_checkpoints_on_test.py \
  --base_model "$SFT_MODEL_PATH" \
  --test_parquet data/icd_grpo_top50/test.parquet \
  --output_dir EasyR1/tasks/eval_top50 \
  --num_gpus 4 \
  --adapter_paths NONE \
                  EasyR1/checkpoints/.../global_step_100/actor/lora_adapter
```

`--adapter_paths NONE` scores the SFT base alone; the rest are LoRA-adapter
directories. Output: per-sample JSONL (`pred_codes`, `gt_codes`, TP/FP/FN sets) +
an aggregate `summary.json`.

---

## The reward

`EasyR1/examples/reward_function/icd.py`. The model emits
`<reasoning>…</reasoning><code>C1, C2, …</code>`; a deterministic parser extracts
the code set, deduplicates, drops malformed codes, and scores it.

The paper's reward is **sample-level F1** between the predicted and gold code
sets (`reward_type=fbeta`, `beta=1.0`, `format_weight=0.0`), with corpus-level
micro/macro F1 computed for logging only. The reward also exposes two options
explored as **future-work** directions in the paper — `beta` for recall-weighted
Fβ, and `reward_type=weighted_fbeta` for per-code class-balanced weighting (rare
codes weighted higher) — both overridable from the command line.

---

## License

Apache-2.0, inheriting [EasyR1](https://github.com/hiyouga/EasyR1) and
[LlamaFactory](https://github.com/hiyouga/LLaMA-Factory). ICD-9-CM descriptions
are CMS public domain (v32 final, 2014).
