#!/bin/bash
# Phase 2 orchestrator for full-code iterative hint refresh with E-style
# description hints. Defaults to 4 GPUs and writes to isolated logs/checkpoints.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 ROUNDS=3 NUM_GPUS=4 \
#     bash tasks/run_iterative_hint_training_full_hintE_desc_4gpu.sh

set -euo pipefail
set -x

# Resolve repo root from this script's location (EasyR1/tasks/ -> repo root).
# Overridable via REPO_ROOT for non-standard layouts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}

export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}

SFT_MERGED=${SFT_MERGED:-${SFT_MODEL_PATH_FULL:-${REPO_ROOT}/models/qwen2.5-1.5b-icd-sft-full}}
BASE_MODEL=${BASE_MODEL:-$SFT_MERGED}
SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT:-${REPO_ROOT}/data/icd_grpo_full_hint}
DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}/data/icd_grpo_full_hintE_desc}
TRAIN_PARQUET=${TRAIN_PARQUET:-${DATA_ROOT}/train.parquet}
EXP_NAME_BASE=${EXP_NAME_BASE:-qwen2_5_1_5b_icd_grpo_phase2_full_hintE_desc_4gpu}
CKPT_ROOT=${CKPT_ROOT:-${REPO_ROOT}/EasyR1/checkpoints/easy_r1/${EXP_NAME_BASE}}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/EasyR1/tasks/logs/phase2_qwen25_1_5b_fullcode_hintE_desc_4gpu}
ROUNDS=${ROUNDS:-3}
NUM_GPUS=${NUM_GPUS:-4}
INFER_TP=${INFER_TP:-4}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-${REPO_ROOT}/EasyR1/examples/qwen2_5_1_5b_icd_grpo_phase2_full_hintE_desc_4gpu.sh}
PY=${PY:-python}
INFER_MAX_PROMPT_LENGTH=${INFER_MAX_PROMPT_LENGTH:-5120}
INFER_MAX_TOKENS=${INFER_MAX_TOKENS:-512}
INFER_GPU_UTIL=${INFER_GPU_UTIL:-0.70}

mkdir -p "$LOG_DIR"

if [ ! -s "$TRAIN_PARQUET" ]; then
    echo "=== Preparing isolated full-code hintE parquet directory: $DATA_ROOT ==="
    mkdir -p "$DATA_ROOT"
    cp "$SOURCE_DATA_ROOT"/train.parquet "$DATA_ROOT"/train.parquet
    cp "$SOURCE_DATA_ROOT"/val.parquet "$DATA_ROOT"/val.parquet
    cp "$SOURCE_DATA_ROOT"/val_small.parquet "$DATA_ROOT"/val_small.parquet
    if [ -s "$SOURCE_DATA_ROOT"/test.parquet ]; then
        cp "$SOURCE_DATA_ROOT"/test.parquet "$DATA_ROOT"/test.parquet
    fi
fi

R0_JSONL=$LOG_DIR/inference_round0.jsonl
if [ ! -s "$R0_JSONL" ]; then
    echo "=== Round 0: inference with full SFT merged model ==="
    ${PY} ${REPO_ROOT}/EasyR1/tasks/inference_on_train.py \
        --model_path "$SFT_MERGED" \
        --train_parquet "$TRAIN_PARQUET" \
        --allowed_codes_parquet "$TRAIN_PARQUET" \
        --output_jsonl "$R0_JSONL" \
        --num_gpus "$INFER_TP" \
        --max_prompt_length "$INFER_MAX_PROMPT_LENGTH" \
        --max_tokens "$INFER_MAX_TOKENS" \
        --gpu_memory_utilization "$INFER_GPU_UTIL" \
        2>&1 | tee "$LOG_DIR/round0_inference.log"
    ${PY} ${REPO_ROOT}/EasyR1/tasks/update_hint_pool.py \
        --inference_jsonl "$R0_JSONL" \
        --train_parquet "$TRAIN_PARQUET" \
        --code_weights_out "$LOG_DIR/code_weights_round0.json" \
        2>&1 | tee "$LOG_DIR/round0_update.log"
else
    echo "Round 0 inference already exists at $R0_JSONL, skipping"
fi

for ROUND in $(seq 1 "$ROUNDS"); do
    R_JSONL="$LOG_DIR/inference_round${ROUND}.jsonl"
    R_WEIGHTS="$LOG_DIR/code_weights_round${ROUND}.json"
    if [ -s "$R_JSONL" ] && [ -s "$R_WEIGHTS" ]; then
        echo "=== Round $ROUND already complete ($R_JSONL + $R_WEIGHTS exist), skipping ==="
        continue
    fi

    echo "=== Round $ROUND: training through epoch $ROUND ==="
    PREV_ROUND=$((ROUND - 1))
    PREV_WEIGHTS="$LOG_DIR/code_weights_round${PREV_ROUND}.json"
    if [ ! -s "$PREV_WEIGHTS" ]; then
        echo "ERROR: code weights file from round $PREV_ROUND not found: $PREV_WEIGHTS"
        exit 1
    fi

    find "$CKPT_ROOT" -name 'dataloader.pt' -delete 2>/dev/null || true

    PREV_STEP=$({ ls -d "$CKPT_ROOT"/global_step_* 2>/dev/null || true; } \
        | awk -F_ '{print $NF}' | sort -n | tail -1)
    PREV_STEP=${PREV_STEP:-0}

    DATA_ROOT="$DATA_ROOT" EXP_NAME="$EXP_NAME_BASE" WANDB_NAME="${EXP_NAME_BASE}_round${ROUND}" \
        NUM_GPUS="$NUM_GPUS" bash "$TRAIN_SCRIPT" \
        data.min_hint_codes=1 \
        data.max_hint_codes=5 \
        data.p_inject_hint=0.5 \
        data.skip_empty_hint_pool=true \
        "data.code_weights_path=${PREV_WEIGHTS}" \
        data.hint_temperature=3.0 \
        data.hint_use_descriptions=true \
        data.hint_descriptions_path=${ICD9_DESC_PATH:-${REPO_ROOT}/data/icd9_descriptions.json} \
        data.hint_description_max_chars=100 \
        data.hint_include_mini_example=true \
        worker.reward.reward_function_kwargs.beta=1.0 \
        trainer.total_epochs="$ROUND" \
        "trainer.experiment_name=${EXP_NAME_BASE}" \
        2>&1 | tee "$LOG_DIR/round${ROUND}_train.log"

    NEW_STEP=$({ ls -d "$CKPT_ROOT"/global_step_* 2>/dev/null || true; } \
        | awk -F_ '{print $NF}' | sort -n | tail -1)
    NEW_STEP=${NEW_STEP:-0}
    if [ "$NEW_STEP" = "$PREV_STEP" ]; then
        echo ">>> No training progress in round $ROUND (global_step stayed at $PREV_STEP)."
        echo ">>> Dataset shrunk below total_epochs=$ROUND x rollout_batch. Stopping early."
        break
    fi
    echo ">>> Round $ROUND progress: $PREV_STEP -> $NEW_STEP"

    LATEST_ACTOR=$({ ls -td "$CKPT_ROOT"/global_step_*/actor/lora_adapter/ 2>/dev/null || true; } | head -1)
    if [ -z "$LATEST_ACTOR" ]; then
        echo "ERROR: no actor/lora_adapter/ found under $CKPT_ROOT after round $ROUND training"
        exit 1
    fi
    echo "Using adapter: $LATEST_ACTOR"

    echo "=== Round $ROUND: inference with LoRA adapter ==="
    ${PY} ${REPO_ROOT}/EasyR1/tasks/inference_on_train.py \
        --model_path "$BASE_MODEL" \
        --adapter_path "$LATEST_ACTOR" \
        --train_parquet "$TRAIN_PARQUET" \
        --allowed_codes_parquet "$TRAIN_PARQUET" \
        --output_jsonl "$R_JSONL" \
        --num_gpus "$INFER_TP" \
        --max_prompt_length "$INFER_MAX_PROMPT_LENGTH" \
        --max_tokens "$INFER_MAX_TOKENS" \
        --gpu_memory_utilization "$INFER_GPU_UTIL" \
        2>&1 | tee "$LOG_DIR/round${ROUND}_inference.log"

    ${PY} ${REPO_ROOT}/EasyR1/tasks/update_hint_pool.py \
        --inference_jsonl "$R_JSONL" \
        --code_weights_out "$LOG_DIR/code_weights_round${ROUND}.json" \
        --train_parquet "$TRAIN_PARQUET" \
        2>&1 | tee "$LOG_DIR/round${ROUND}_update.log"
done

echo "=== All full-code hintE 4GPU rounds complete ==="
