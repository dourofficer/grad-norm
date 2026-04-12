#!/bin/bash
# sweep_entropy.sh — Run entropy scoring across models × subsets.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/entropy/sweep_entropy.sh
#   CUDA_VISIBLE_DEVICES=1 TEMPS="1.0 0.5" bash scripts/entropy/sweep_entropy.sh
#   CUDA_VISIBLE_DEVICES=2 START=0 END=10   bash scripts/entropy/sweep_entropy.sh

set -euo pipefail

MAX_TOKENS="${MAX_TOKENS:-8192}"
START="${START:-0}"
END="${END:-}"
TEMPS="${TEMPS:-1.0}"   # space-separated list of temperatures to sweep
BASE="outputs/entropy"

MODELS=(
    "/data/hoang/resources/models/Qwen/Qwen3-8B|qwen3-8b"
    "/data/hoang/resources/models/meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b"
)

SUBSETS=(
    "hand-crafted"
    "algorithm-generated"
)

END_FLAG=""
if [[ -n "$END" ]]; then
    END_FLAG="--end_idx $END"
fi

for entry in "${MODELS[@]}"; do
    IFS="|" read -r model_path model_tag <<< "$entry"
    for subset in "${SUBSETS[@]}"; do
        for temp in $TEMPS; do
            # tag: temp=1.0 → no suffix; temp=0.5 → -t0.5
            if [[ "$temp" == "1.0" ]]; then
                tag="${model_tag}"
            else
                tag="${model_tag}-t${temp}"
            fi
            echo "━━━ ${tag} / ${subset} / temperature=${temp} ━━━"
            python -m gradnorm.entropy \
                --model       "$model_path" \
                --input       "ww/${subset}" \
                --max_tokens  "$MAX_TOKENS" \
                --temperature "$temp" \
                --output      "${BASE}/${tag}/${subset}" \
                --start_idx   "$START" \
                $END_FLAG
            echo ""
        done
    done
done

echo "All done."