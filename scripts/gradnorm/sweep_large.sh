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

BASE="/data/projects/32000010/thanhdo/grad-norm/outputs/gradnorm"

HUB="/data/projects/32000010/hub"
MODELS=(
    # "${HUB}/Qwen/Qwen3-14B|qwen3-14b"
    "${HUB}/Qwen/Qwen3-32B|qwen3-32b"
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
        echo "━━━ ${model_tag} / ${subset} / KL Uniform ━━━"
	    python -m gradnorm.run \
            --model      "$model_path" \
            --input      "ww/${subset}" \
            --max_tokens "$MAX_TOKENS" \
            --loss       "kl_uniform" \
            --output     "${BASE}/${model_tag}-kl_uniform/${subset}" \
            --start_idx  "$START" \
            $END_FLAG
        
        echo "━━━ ${model_tag} / ${subset} / NTP ━━━"
        python -m gradnorm.run \
            --model      "$model_path" \
            --input      "ww/${subset}" \
            --max_tokens "$MAX_TOKENS" \
            --loss       "ntp" \
            --output     "${BASE}/${model_tag}-ntp/${subset}" \
            --start_idx  "$START" \
            $END_FLAG
        echo ""
    done
done

echo "All done."

