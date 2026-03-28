#!/bin/bash
set -euo pipefail

# ── Models (parallel arrays — index i describes one model) ───────────────────
MODEL_PATHS=(
    "/data/hoang/resources/models/Qwen/Qwen3-8B"
    "/data/hoang/resources/models/meta-llama/Llama-3.1-8B-Instruct"
)
MODEL_NAMES=(
    "qwen3-8b"
    "llama-3.1-8b"
)
TARGET_PARAMS=(
    "model.layers.35.self_attn.v_proj.weight"
    "model.layers.31.self_attn.v_proj.weight"
)

SUBSETS=("hand-crafted" "algorithm-generated")
INPUT_BASE="ww"
OUTPUT_BASE="sal_outputs/grads"
MAX_TOKENS=8192

# ── Sweep ────────────────────────────────────────────────────────────────────
for i in "${!MODEL_PATHS[@]}"; do
    model_path="${MODEL_PATHS[$i]}"
    model_name="${MODEL_NAMES[$i]}"
    target_param="${TARGET_PARAMS[$i]}"

    for subset in "${SUBSETS[@]}"; do
        output_dir="${OUTPUT_BASE}/${model_name}/${subset}"

        echo "════════════════════════════════════════════════════════════════"
        echo "Model:        ${model_name} (${model_path})"
        echo "Subset:       ${subset}"
        echo "Target param: ${target_param}"
        echo "Output:       ${output_dir}"
        echo "════════════════════════════════════════════════════════════════"

        python -m cli.sal_extract \
            --model "$model_path" \
            --input "${INPUT_BASE}/${subset}" \
            --output "$output_dir" \
            --target_param "$target_param" \
            --max_tokens "$MAX_TOKENS"

        echo ""
    done
done

echo "All done."