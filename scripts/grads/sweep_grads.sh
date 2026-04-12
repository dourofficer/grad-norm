#!/bin/bash
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/grads/sweep_extract.sh

set -euo pipefail

# Configuration
MODELS=(
    "/data/hoang/resources/models/Qwen/Qwen3-8B|qwen3-8b"
    "/data/hoang/resources/models/meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b"
)

SUBSETS=(
    "hand-crafted"
    "algorithm-generated"
)

BASE_OUT="outputs/grads"
MAX_TOKENS=8192

# Optional environment overrides (e.g. START=10 bash ...)
START="${START:-0}"
END="${END:-}"
END_FLAG=""
if [[ -n "$END" ]]; then
    END_FLAG="--end_idx $END"
fi

for entry in "${MODELS[@]}"; do
    IFS="|" read -r model_path model_tag <<< "$entry"
    for subset in "${SUBSETS[@]}"; do
        echo "━━━ Model: ${model_tag} | Subset: ${subset} ━━━"
        
        python -m grads.extract \
            --model         "$model_path" \
            --input         "ww/${subset}" \
            --output        "${BASE_OUT}/${model_tag}/${subset}" \
            --target_params all \
            --max_tokens    "$MAX_TOKENS" \
            --start_idx     "$START" \
            $END_FLAG
            
        echo ""
    done
done

echo "Sweep completed."
