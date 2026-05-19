#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# Evaluate models on ToolAlpaca (greedy acc@1 via vLLM)
#
# Usage:
#   bash eval/run_tool_eval.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

AVAILABLE_GPUS=(0 1)
GPU_LIST=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1

# ── Edit these ──────────────────────────────────────────────────────
DATA_PATH="path/to/tooluse_data/eval_data" ### default: "data/tooluse_data/eval_data"

MODELS=(
  "path/to/infosft_tool_7e-6_2epoch"
  "path/to/sft_tool_7e-6_2epoch"
  "path/to/dft_tool_7e-6_2epoch"
)
EXPS=(
  "infosft_tool_7e-6_2epoch"
  "sft_tool_7e-6_2epoch"
  "dft_tool_7e-6_2epoch"
)
# ────────────────────────────────────────────────────────────────────

[[ ${#MODELS[@]} -eq ${#EXPS[@]} ]] || { echo "MODELS and EXPS must have same length"; exit 1; }

wait_for_free_gpus() {
  while true; do
    busy="$(nvidia-smi -i "$GPU_LIST" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF')"
    [[ -z "$busy" ]] && break
    echo "GPUs busy. Waiting 30s..."
    sleep 30
  done
}

PY="$SCRIPT_DIR/tool_eval.py"

for i in "${!MODELS[@]}"; do
    wait_for_free_gpus

    echo "════════ ${EXPS[$i]} ════════"
    CUDA_VISIBLE_DEVICES="$GPU_LIST" python "$PY" \
        --model_path "${MODELS[$i]}" \
        --exp_name   "${EXPS[$i]}" \
        --data_path  "$DATA_PATH"
done