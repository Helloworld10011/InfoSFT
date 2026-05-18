set -euo pipefail

MODEL_ID=${1:?"Specify model_id (e.g. Qwen/Qwen2.5-Math-7B)"}
LOSS_TYPE=${2:?"Specify loss_type: nll | infosft | dft"}
LR=${3:?"Specify lr (we use 2ep5 for all models.)"}
SEED=${4:-42}

AVAILABLE_GPUS=(0 1)
GPU_LIST=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")
NUM_GPUS=${#AVAILABLE_GPUS[@]}

MODEL_SHORT=$(basename "$MODEL_ID" | tr '.' '-')
EXP_NAME="${MODEL_SHORT}_ultrafeedback_${LOSS_TYPE}"

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
    --task code \
    --model_id "$MODEL_ID" \
    --exp_name "$EXP_NAME" \
    --output_root outputs/code \
    --loss_type "$LOSS_TYPE" --lr "$LR" --seed "$SEED" \
    --batch_size 32 \
    --warmup_ratio 0.05 --num_samples 12000 --use_lora \
    --deepspeed_config "$DS_CONFIG" \
    --chat_template_path "$CHAT_TEMPLATE" \
    "${@:6}"