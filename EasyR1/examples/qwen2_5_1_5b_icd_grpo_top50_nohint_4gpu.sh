#!/bin/bash
# Top-50 GRPO baseline: pure GRPO from the SFT base, no hints.
# Uses the no-hint parquet and does not set data.hint_pool_key, so all
# hint injection / filtering is disabled. This is the GRPO entry point;
# switch dataset / label scope / backbone via env vars + CLI overrides
# (see the README).

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -f "${SCRIPT_DIR}/../.wandb_env" ]; then
    source "${SCRIPT_DIR}/../.wandb_env"
fi

MODEL_PATH=${SFT_MODEL_PATH:-${REPO_ROOT}/models/qwen2.5-1.5b-icd-sft-top50}
DATA_ROOT=${REPO_ROOT}/data/icd_grpo_top50_nothink
PY=python
N_GPUS=${NUM_GPUS:-4}

${PY} -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_ROOT}/train.parquet \
    data.val_files=${DATA_ROOT}/val_small.parquet \
    data.format_prompt=./examples/format_prompt/icd_passthrough.jinja \
    data.max_prompt_length=5120 \
    data.max_response_length=196 \
    data.rollout_batch_size=1024 \
    data.val_batch_size=1000 \
    data.min_hint_codes=0 \
    data.max_hint_codes=0 \
    data.p_inject_hint=0.0 \
    data.skip_empty_hint_pool=false \
    data.filter_overlong_prompts=false \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.lora.rank=8 \
    worker.actor.model.lora.alpha=8 \
    worker.actor.global_batch_size=1024 \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    worker.actor.optim.lr=1e-5 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=8 \
    worker.rollout.gpu_memory_utilization=0.90 \
    worker.reward.reward_function=./examples/reward_function/icd.py:compute_score \
    worker.reward.reward_function_kwargs.format_weight=0.0 \
    worker.reward.reward_function_kwargs.beta=1.0 \
    worker.reward.reward_function_kwargs.reward_type=fbeta \
    worker.reward.reward_function_kwargs.penalty=0.5 \
    trainer.logger='["console","file","wandb"]' \
    trainer.experiment_name=qwen2_5_1_5b_icd_grpo_top50_nohint \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.total_epochs=5 \
    trainer.save_freq=10 \
    trainer.save_limit=-1 \
    trainer.val_freq=10 \
    "$@"
