#!/bin/bash
set -euo pipefail
LANGS=(cpp java php ts cs sh js) ###Set this to the languages you want to evaluate on. We average these results including humaneval for the 8 primary languages reported in the paper. 
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

BASE_DIR=  ###Set this to your desired base directory for logs and results
GEN_SCRIPT="$SCRIPT_DIR/gen_multiple.py"

### GPU Configuration
AVAILABLE_GPUS=(6 7)
GPU_LIST=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")
NUM_GPUS=${#AVAILABLE_GPUS[@]}

export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false


### Add the models and experiment names you want to run here. Make sure they are in the same order.
MODELS=(
    "path/to/qwen7b_math_infosft"
)

EXPS=(
    "qwen7b_math_infosft"
)


mkdir -p "$BASE_DIR"/{gens,results,logs}

for i in "${!MODELS[@]}"; do
    for LANG in "${LANGS[@]}"; do
        EXP="${EXPS[$i]}"
        GEN="$BASE_DIR/gens/${EXP}_${LANG}"
        RES="$BASE_DIR/results/${EXP}_${LANG}"
        [[ -d "$RES" && $(ls -A "$RES" 2>/dev/null) ]] && continue

        echo "=== $EXP / $LANG ==="
        CUDA_VISIBLE_DEVICES="$GPU_LIST" python "$GEN_SCRIPT" \
            --model "${MODELS[$i]}" --lang "$LANG" --out_dir "$GEN" --tp $NUM_GPUS \
            2>&1 | tee "$BASE_DIR/logs/${EXP}_${LANG}.log"

        mkdir -p "$RES"
        docker run --rm -v "$BASE_DIR/gens:/g" -v "$BASE_DIR/results:/r" \
            ghcr.io/nuprl/multipl-e-evaluation:latest \
            --dir "/g/${EXP}_${LANG}" --output-dir "/r/${EXP}_${LANG}" --max-workers 8
    done
done

python - <<'PY'
import json, glob, os, gzip, collections
base = "/home/ubuntu/newSFT/eval_code/multiple/results"
def load(p):
    return json.load(gzip.open(p,"rt") if p.endswith(".gz") else open(p))
agg = collections.defaultdict(dict)
for d in sorted(glob.glob(f"{base}/*")):
    if not os.path.isdir(d): continue
    exp, lang = os.path.basename(d).rsplit("_", 1)
    files = glob.glob(f"{d}/*.json") + glob.glob(f"{d}/*.json.gz")
    ok = tot = 0
    for f in files:
        try: r = load(f)
        except: continue
        if "results" not in r: continue
        tot += 1
        ok += r["results"][0]["status"] == "OK"
    if tot: agg[exp][lang] = ok/tot
for exp, langs in agg.items():
    avg = sum(langs.values())/len(langs)*100
    print(f"{exp}: avg pass@1 = {avg:.2f}  ({len(langs)} langs)  {langs}")
PY