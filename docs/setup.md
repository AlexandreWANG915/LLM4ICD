# Setup

Three ways to install — pick one.

## Option A — `requirements.txt` (minimal, recommended)

Installs both sub-frameworks in editable mode + pins the vLLM/torch/flash-attn ABI chain. Pip resolves the rest of the transitive deps from each framework's own pyproject.toml.

```bash
conda create -n llm4icd python=3.11 -y
conda activate llm4icd
pip install -r requirements.txt
```

Use this when you want a clean install that follows the latest patch releases of LlamaFactory's / EasyR1's other transitive dependencies.

## Option B — `requirements.lock.txt` (byte-identical reproduction)

Reproduces the exact dependency set we developed and validated against, all 309 transitive deps pinned (Python 3.11.6, torch 2.6.0, vLLM 0.8.2, transformers 4.56.1, ray 2.53.0, peft 0.18.0, deepspeed 0.16.4, flash-attn 2.7.3).

```bash
conda create -n llm4icd python=3.11 -y
conda activate llm4icd
pip install -r requirements.lock.txt

# Editable installs so the in-tree patches are picked up:
pip install -e EasyR1
pip install -e LlamaFactory
```

Use this when you want byte-identical reproduction (e.g. debugging a numeric difference vs our published numbers).

## Option C — Two separate envs from upstream requirements (lightest)

Use this if you only need one of the two frameworks, or if neither A nor B resolves cleanly on your CUDA / driver stack.

### EasyR1 env (Stage 2 — RL)
```bash
cd EasyR1
conda create -n easyr1 python=3.11 -y
conda activate easyr1
pip install -e .
```
Pulls vLLM, Ray, PEFT (LoRA), Transformers, etc. from EasyR1's own pyproject.

### LlamaFactory env (Stage 1 — SFT)
```bash
cd LlamaFactory
conda create -n llamafactory python=3.11 -y
conda activate llamafactory
pip install -e .
```
See LlamaFactory's upstream README for the full install matrix and optional extras (deepspeed, vllm, awq, ...).

---

## Hardware

Tested on **4× H100 80GB**. The default scripts (`worker.actor.global_batch_size=1024`, `data.rollout_batch_size=1024`, `worker.rollout.n=8`) saturate ~50-65 GB/GPU. For smaller cards:
- 4× A100 40GB: drop `rollout_batch_size` to 256, `worker.actor.micro_batch_size_per_device_for_update` to 4.
- 1-2 GPUs: drop `rollout_batch_size` to 64-128 and use the full-code script's smaller defaults.

CUDA: tested on 12.x (driver supports torch 2.6 + flash-attn 2.7.3).

---

## Sanity check

```bash
cd EasyR1
python tasks/test_reward_metrics.py         # ICD reward + corpus F1 sanity
```

