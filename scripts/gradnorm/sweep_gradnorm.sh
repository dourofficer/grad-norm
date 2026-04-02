#!/bin/bash
# sweep_gradnorm.sh — Run gradnorm inference across models × subsets.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash sweep_gradnorm.sh
#   CUDA_VISIBLE_DEVICES=1 MAX_TOKENS=12000 bash sweep_gradnorm.sh

set -euo pipefail

MAX_TOKENS="${MAX_TOKENS:-8192}"
START="${START:-0}"
END="${END:-}"  # empty = all
BASE="outputs/gradnorm-v2"

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

# -full suffix in the output indicates full context.
for entry in "${MODELS[@]}"; do
    IFS="|" read -r model_path model_tag <<< "$entry"
    for subset in "${SUBSETS[@]}"; do
        echo "━━━ ${model_tag} / ${subset} ━━━"
        python -m gradnorm.run \
            --model      "$model_path" \
            --input      "ww/${subset}" \
            --max_tokens "$MAX_TOKENS" \
            --output     "${BASE}/${model_tag}-full/${subset}" \
            --start_idx  "$START" \
            $END_FLAG
        echo ""
    done
done

echo "All done."