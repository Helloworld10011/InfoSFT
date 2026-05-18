#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# NuminaMath-CoT  ·  100k samples  ·  full fine-tuning  ·  BS = 256
#
# Learning rate: 5e-5 for 1.5B models, 2e-5 for 7B / 8B models
#
# Usage:
#   bash scripts/run_math.sh <model_id> <loss_type> <lr> [P] [seed]
#
# Examples:
#   bash scripts/run_math.sh Qwen/Qwen2.5-Math-1.5B nll 5e-5
#   bash scripts/run_math.sh Qwen/Qwen2.5-Math-1.5B infosft 5e-5
#   bash scripts/run_math.sh Qwen/Qwen2.5-Math-7B   infosft 2e-5
#   bash scripts/run_math.sh meta-llama/Llama-3.1-8B dft 2e-5
#
# Extra flags (chat template, deepspeed, etc.) can be appended:
#   bash scripts/run_math.sh Qwen/Qwen2.5-Math-7B infosft 2e-5 42 \
#        --chat_template_path templates/qwen2_5_training.jinja       \
#        --deepspeed_config   configs/ds_z2_config.json
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL_ID=${1:?"Specify model_id (e.g. Qwen/Qwen2.5-Math-7B)"}
LOSS_TYPE=${2:?"Specify loss_type: nll | infosft | dft"}
LR=${3:?"Specify lr (e.g. 5e-5 for 1.5B, 2e-5 for 7B/8B)"}
SEED=${4:-42}

AVAILABLE_GPUS=(0 1 2 3)
GPU_LIST=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")
NUM_GPUS=${#AVAILABLE_GPUS[@]}

MODEL_SHORT=$(basename "$MODEL_ID" | tr '.' '-')
EXP_NAME="${MODEL_SHORT}_numinamath_${LOSS_TYPE}"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DS_CONFIG="$SCRIPT_DIR/ds_z2_config.json"

if echo "$MODEL_ID" | grep -qi "llama"; then
  CHAT_TEMPLATE="$SCRIPT_DIR/llama3_1_training.jinja"
else
  CHAT_TEMPLATE="$SCRIPT_DIR/qwen2_5_training.jinja"
fi

CUDA_VISIBLE_DEVICES=$GPU_LIST accelerate launch \
  --multi_gpu --num_processes $NUM_GPUS \
  training/main_train.py \
    --task math \
    --model_id "$MODEL_ID" \
    --exp_name "$EXP_NAME" \
    --output_root outputs/numinamath \
    --loss_type "$LOSS_TYPE" --lr "$LR" --seed "$SEED" \
    --batch_size 256 \
    --deepspeed_config "$DS_CONFIG" \
    --chat_template_path "$CHAT_TEMPLATE" \
    "${@:6}"