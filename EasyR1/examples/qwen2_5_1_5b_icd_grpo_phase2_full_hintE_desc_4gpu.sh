#!/bin/bash
# Phase 2 full-code training with E-style description hints on 4 GPUs.
# Separated from the old full-code hint runs to avoid overwriting checkpoints.

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -f "${SCRIPT_DIR}/../.wandb_env" ]; then
    source "${SCRIPT_DIR}/../.wandb_env"
fi

# Override CUDA_VISIBLE_DEVICES when using a different 4-GPU allocation.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}

MODEL_PATH=${MODEL_PATH:-${SFT_MODEL_PATH_FULL:-${REPO_ROOT}/models/qwen2.5-1.5b-icd-sft-full}}
DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}/data/icd_grpo_full_hintE_desc}
PY=${PY:-python}

N_GPUS=${NUM_GPUS:-4}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-512}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
EXP_NAME=${EXP_NAME:-qwen2_5_1_5b_icd_grpo_phase2_full_hintE_desc_4gpu}
SAVE_FREQ=${SAVE_FREQ:-10}
VAL_FREQ=${VAL_FREQ:-10}
# Reward knobs (env-overridable so wrappers can swap to weighted_fbeta).
REWARD_TYPE=${REWARD_TYPE:-fbeta}
REWARD_BETA=${REWARD_BETA:-1.0}
REWARD_CODE_WEIGHT_PATH=${REWARD_CODE_WEIGHT_PATH:-}
REWARD_CODE_WEIGHT_DEFAULT=${REWARD_CODE_WEIGHT_DEFAULT:-1.0}

${PY} -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_ROOT}/train.parquet \
    data.val_files=${DATA_ROOT}/val_small.parquet \
    data.format_prompt=./examples/format_prompt/icd_passthrough.jinja \
    data.max_prompt_length=5120 \
    data.max_response_length=512 \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.val_batch_size=1000 \
    data.hint_pool_key=hint_pool \
    data.min_hint_codes=1 \
    data.max_hint_codes=5 \
    data.p_inject_hint=0.5 \
    data.skip_empty_hint_pool=true \
    data.hint_temperature=3.0 \
    data.hint_use_descriptions=true \
    data.hint_descriptions_path=${ICD9_DESC_PATH:-${REPO_ROOT}/data/icd9_descriptions.json} \
    data.hint_description_max_chars=100 \
    data.hint_include_mini_example=true \
    data.filter_overlong_prompts=false \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.lora.rank=8 \
    worker.actor.model.lora.alpha=8 \
    worker.actor.global_batch_size=${ROLLOUT_BATCH_SIZE} \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.optim.lr=1e-5 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=8 \
    worker.rollout.gpu_memory_utilization=0.70 \
    worker.reward.reward_function=./examples/reward_function/icd.py:compute_score \
    worker.reward.reward_function_kwargs.format_weight=0.0 \
    worker.reward.reward_function_kwargs.beta=${REWARD_BETA} \
    worker.reward.reward_function_kwargs.reward_type=${REWARD_TYPE} \
    worker.reward.reward_function_kwargs.penalty=0.5 \
    "worker.reward.reward_function_kwargs.code_weight_path=${REWARD_CODE_WEIGHT_PATH}" \
    worker.reward.reward_function_kwargs.code_weight_default=${REWARD_CODE_WEIGHT_DEFAULT} \
    trainer.logger='["console","file","wandb"]' \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.save_limit=-1 \
    trainer.val_freq=${VAL_FREQ} \
    "$@"
