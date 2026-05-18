#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# OpenR1-Math-220k · full FT · 2 epochs · BS = 64 · max_len = 8192
#
# Examples:
#   bash training/scripts/run_openR1.sh Qwen/Qwen2.5-7B-Instruct nll 5e-6
#   bash training/scripts/run_openR1.sh Qwen/Qwen2.5-7B-Instruct infosft 5e-6
#   bash training/scripts/run_openR1.sh Qwen/Qwen2.5-7B-Instruct dft 5e-6
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL_ID=${1:?"Specify model_id (e.g. Qwen/Qwen2.5-7B-Instruct)"}
LOSS_TYPE=${2:?"Specify loss_type: nll | infosft | dft"}
LR=${3:-5e-6}
SEED=${4:-42}

AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)
GPU_LIST=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")
NUM_GPUS=${#AVAILABLE_GPUS[@]}

MODEL_SHORT=$(basename "$MODEL_ID" | tr '.' '-')
EXP_NAME="${MODEL_SHORT}_openR1_${LOSS_TYPE}"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DS_CONFIG="$SCRIPT_DIR/ds_z2_config.json"
CHAT_TEMPLATE="$SCRIPT_DIR/qwen2_5_training.jinja"

CUDA_VISIBLE_DEVICES=$GPU_LIST accelerate launch \
  --multi_gpu --num_processes $NUM_GPUS \
  training/openr1_train.py \
    --model_id "$MODEL_ID" \
    --exp_name "$EXP_NAME" \
    --output_root outputs/openR1 \
    --loss_type "$LOSS_TYPE" --lr "$LR" --seed "$SEED" \
    --batch_size 64 \
    --deepspeed_config "$DS_CONFIG" \
    --chat_template_path "$CHAT_TEMPLATE" \
    "${@:5}"