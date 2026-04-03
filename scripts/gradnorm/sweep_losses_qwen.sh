#!/bin/bash
#
# Usage:
#   CUDA_VISIBLE_DEVICES=3 bash scripts/gradnorm/sweep_losses.sh
#   CUDA_VISIBLE_DEVICES=3 START=33 END=34 bash scripts/gradnorm/sweep_losses.sh
#   CUDA_VISIBLE_DEVICES=1 MAX_TOKENS=12000 bash scripts/gradnorm/sweep_gradnorm.sh

set -euo pipefail

MAX_TOKENS="${MAX_TOKENS:-8192}"
START="${START:-0}"
END="${END:-}"  # empty = all
BASE="outputs/gradnorm-losses"

MODELS=(
    # "/data/hoang/resources/models/meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b"
    "/data/hoang/resources/models/Qwen/Qwen3-8B|qwen3-8b"
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
        echo "━━━ ${model_tag} / ${subset} / kl uniform loss ━━━"
        python -m gradnorm.run \
            --model      "$model_path" \
            --input      "ww/${subset}" \
            --max_tokens "$MAX_TOKENS" \
            --loss        kl_uniform \
            --output     "${BASE}/${model_tag}-kl_uniform/${subset}" \
            --start_idx  "$START" \
            $END_FLAG

        echo "━━━ ${model_tag} / ${subset} / kl temp loss 1.5 ━━━"
        python -m gradnorm.run \
            --model      "$model_path" \
            --input      "ww/${subset}" \
            --max_tokens "$MAX_TOKENS" \
            --loss kl_temp --temperature 1.5 \
            --output     "${BASE}/${model_tag}-kl_15/${subset}" \
            --start_idx  "$START" \
            $END_FLAG

        echo "━━━ ${model_tag} / ${subset} / kl temp loss 2.0 ━━━"
        python -m gradnorm.run \
            --model      "$model_path" \
            --input      "ww/${subset}" \
            --max_tokens "$MAX_TOKENS" \
            --loss kl_temp --temperature 2.0 \
            --output     "${BASE}/${model_tag}-kl_20/${subset}" \
            --start_idx  "$START" \
            $END_FLAG
        echo ""
    done
done

echo "All done."