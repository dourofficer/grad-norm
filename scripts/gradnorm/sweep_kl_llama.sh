#!/bin/bash
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/gradnorm/sweep_kl_llama.sh

set -euo pipefail

MAX_TOKENS="${MAX_TOKENS:-8192}"
START="${START:-0}"
END="${END:-}"  # empty = all
BASE="outputs/gradnorm-losses"

MODELS=(
    # "/data/hoang/resources/models/Qwen/Qwen3-8B|qwen3-8b"
    "/data/hoang/resources/models/meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b"
)

SUBSETS=(
    "algorithm-generated"
    "hand-crafted"
)

# Define the temperature range: 1.6 to 3.0 with step 0.2
TEMPS=("1.2" "1.4" "1.6" "1.8" "2.2" "2.4" "2.6" "2.8" "3.0")

END_FLAG=""
if [[ -n "$END" ]]; then
    END_FLAG="--end_idx $END"
fi

for entry in "${MODELS[@]}"; do
    IFS="|" read -r model_path model_tag <<< "$entry"
    for subset in "${SUBSETS[@]}"; do
        for temp in "${TEMPS[@]}"; do
            # Format suffix for output folder (e.g., 1.6 -> 16, 2.0 -> 20)
            temp_suffix=$(echo "$temp" | tr -d '.')
            
            echo "━━━ ${model_tag} / ${subset} / kl temp loss ${temp} ━━━"
            python -m gradnorm.run \
                --model      "$model_path" \
                --input      "ww/${subset}" \
                --max_tokens "$MAX_TOKENS" \
                --loss kl_temp --temperature "$temp" \
                --output     "${BASE}/${model_tag}-kl_${temp_suffix}/${subset}" \
                --start_idx  "$START" \
                $END_FLAG
        done
        echo ""
    done
done

echo "Sweep completed."
