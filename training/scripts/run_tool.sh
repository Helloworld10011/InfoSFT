#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Tool-use sweep: {nll, dft, infosft} × {2, 1} epochs × 4 LRs
# Effective batch size = 32  (originally: 8 per-device × 1 ga × 4 GPUs)
#
# Usage:
#   bash training/scripts/run_tool_sweep.sh \
#        Qwen/Qwen2.5-7B-Instruct /path/to/tooluse_data/train_data
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL_ID=${1:?"Specify model_id (e.g. Qwen/Qwen2.5-7B-Instruct)"}
DATA_PATH=${2:?"Specify data_path (e.g. data/tooluse/train_data)"}
SEED=${3:-42}

AVAILABLE_GPUS=(0 1 2 3)
GPU_LIST=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")
NUM_GPUS=${#AVAILABLE_GPUS[@]}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DS_CONFIG="$SCRIPT_DIR/ds_z2_config.json"
CHAT_TEMPLATE="$SCRIPT_DIR/qwen2_5_training.jinja"

for LR in 1e-6 2e-6 5e-6 7e-6; do
  for EPOCHS in 2 1; do
    for LOSS in nll dft infosft; do
      EXP_NAME="${LOSS}_tool_${LR}_${EPOCHS}epoch"

      echo "════════ ${EXP_NAME} ════════"
      CUDA_VISIBLE_DEVICES=$GPU_LIST accelerate launch \
        --multi_gpu --num_processes $NUM_GPUS \
        training/Science_Tooluse.py \
          --task tool \
          --data_path "$DATA_PATH" \
          --model_id "$MODEL_ID" \
          --exp_name "$EXP_NAME" \
          --output_root outputs/tool \
          --loss_type "$LOSS" --lr "$LR" --epochs "$EPOCHS" --seed "$SEED" \
          --batch_size 32 \
          --deepspeed_config "$DS_CONFIG" \
          --chat_template_path "$CHAT_TEMPLATE"
    done
  done
done