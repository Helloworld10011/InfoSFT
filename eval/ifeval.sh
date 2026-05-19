#!/bin/bash
set -euo pipefail

# 1. GPU Configuration
BASE_DIR=  ###Set this to your desired base directory for logs and results

### GPU Configuration
AVAILABLE_GPUS=(0 1)
GPU_LIST=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")
NUM_GPUS=${#AVAILABLE_GPUS[@]}

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1

### Add the models and experiment names you want to run here. Make sure they are in the same order.
MODELS=(
  "path/to/qwen7b_math_infosft"
)

EXPS=(
  "qwen7b_math_infosft"
)


# 4. Helper Function to ensure isolation
wait_for_free_gpus() {
  echo "Checking for free GPUs ($GPU_LIST)..."
  while true; do
    # check if any process is running on the specific GPUs we want to use
    busy="$(nvidia-smi -i "$GPU_LIST" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF')"
    if [[ -z "$busy" ]]; then
        echo "GPUs $GPU_LIST are free. Starting next run."
        break
    fi
    echo "GPUs busy. Waiting 30s..."
    sleep 30
  done
}

mkdir -p "$BASE_DIR/logs"
mkdir -p "$BASE_DIR/results"

for i in "${!MODELS[@]}"; do
    wait_for_free_gpus
    
    MODEL="${MODELS[$i]}"
    EXP_NAME="${EXPS[$i]}"
    
    echo "========================================================"
    echo "Running IFeval for: $EXP_NAME"
    echo "Using $NUM_GPUS GPUs (Tensor Parallelism)"
    echo "========================================================"    
    MODEL_ARGS="pretrained=$MODEL,"
    MODEL_ARGS+="tensor_parallel_size=$NUM_GPUS,"
    MODEL_ARGS+="dtype=bfloat16,"
    MODEL_ARGS+="gpu_memory_utilization=0.9,"
    MODEL_ARGS+="enable_prefix_caching=False,"
    MODEL_ARGS+="enforce_eager=True,"
    MODEL_ARGS+="trust_remote_code=True,"
    MODEL_ARGS+="max_gen_toks=2048"

    CUDA_VISIBLE_DEVICES="$GPU_LIST" lm_eval \
        --model vllm \
        --model_args "$MODEL_ARGS" \
        --tasks ifeval \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --batch_size auto \
        --output_path "$BASE_DIR/results/$EXP_NAME" \
        --gen_kwargs temperature=0,do_sample=False,seed=42 \
        --log_samples \
        2>&1 | tee "$BASE_DIR/logs/${EXP_NAME}.log"
    done

echo "========================================================"
echo "ALL RUNS COMPLETE. FINAL SCORES:"
echo "========================================================"
grep "prompt_level_strict_acc" "$BASE_DIR"/results/*/results.json