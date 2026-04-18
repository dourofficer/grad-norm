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
CONTEXT_STRATEGY="all"  # "dependency" or "all"

TEMPS=("1.2" "1.6" "2.0" "2.2" "2.4" "2.6" "3.0")

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
        # echo "━━━ Model: ${model_tag} | Subset: ${subset} ━━━ | NTP Loss"
        # python -m grads.extract \
        #     --model         "$model_path" \
        #     --loss          "ntp" \
        #     --input         "ww/${subset}" \
        #     --output        "${BASE_OUT}/${model_tag}/${subset}" \
        #     --target_params all \
        #     --max_tokens    "$MAX_TOKENS" \
        #     --context      "$CONTEXT_STRATEGY" \
        #     --start_idx     "$START" \
        #     $END_FLAG

        echo "━━━ Model: ${model_tag} | Subset: ${subset} ━━━ | KL Uniform Loss"
        python -m grads.extract \
            --model         "$model_path" \
            --loss          "kl_uniform" \
            --input         "ww/${subset}" \
            --output        "${BASE_OUT}/${model_tag}-kl_uniform/${subset}" \
            --target_params all \
            --max_tokens    "$MAX_TOKENS" \
            --context      "$CONTEXT_STRATEGY" \
            --start_idx     "$START" \
            $END_FLAG

        for temp in "${TEMPS[@]}"; do
            # Format suffix for output folder (e.g., 1.6 -> 16, 2.0 -> 20)
            temp_suffix=$(echo "$temp" | tr -d '.')
            echo "━━━ Model: ${model_tag} | Subset: ${subset} | KL Temp: ${temp} ━━━"
            python -m grads.extract \
                --model         "$model_path" \
                --loss kl_temp --temperature "$temp" \
                --input         "ww/${subset}" \
                --output        "${BASE_OUT}/${model_tag}-kl_${temp_suffix}/${subset}" \
                --target_params all \
                --max_tokens    "$MAX_TOKENS" \
                --context      "$CONTEXT_STRATEGY" \
                --start_idx     "$START" \
                $END_FLAG
        done
        echo ""
    done
done

echo "Sweep completed."
