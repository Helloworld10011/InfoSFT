#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# Evaluate models on math benchmarks (acc@1 + pass@k via vLLM)
#
# Usage:
#   bash eval/run_math_eval.sh
#
# Edit MODELS / EXPS / DATASETS below, then run.
# Use --max_tokens 16384 for long-CoT models (e.g. OpenR1).
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

AVAILABLE_GPUS=(0 1 2 3)
GPU_LIST=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1

# ── Edit these ──────────────────────────────────────────────────────
DATASETS=("math500" "amc" "minerva")   # aime | amc | math500 | hmmt | minerva
MAX_TOKENS=2048                         # 16384 for long-CoT models

MODELS=(
  "path/to/qwen7b_math_nll"
  "path/to/qwen7b_math_infosft"
  "path/to/qwen7b_math_dft"
)
EXPS=(
  "qwen7b_math_nll"
  "qwen7b_math_infosft"
  "qwen7b_math_dft"
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

PY="$SCRIPT_DIR/math_eval.py"

for DATASET in "${DATASETS[@]}"; do
  for i in "${!MODELS[@]}"; do
    echo "════════ ${EXPS[$i]} · ${DATASET} ════════"
    wait_for_free_gpus
    CUDA_VISIBLE_DEVICES=$GPU_LIST python "$PY" \
      --model_path "${MODELS[$i]}" \
      --exp_name   "${EXPS[$i]}" \
      --dataset    "$DATASET" \
      --max_tokens "$MAX_TOKENS"
  done
done