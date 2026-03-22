#!/bin/bash

CONFIG="configs/qwen3-8b.yaml"
BASE_INPUT="ww"
BASE_OUTPUT="outputs/qwen3-8b"

METHODS=("all_at_once" "step_by_step_full" "step_by_step_partial")
INPUTS=("algorithm-generated" "hand-crafted")

for METHOD in "${METHODS[@]}"; do
    for INPUT in "${INPUTS[@]}"; do
        # Convert method underscores to hyphens for output path
        METHOD_DIR=$(echo "$METHOD" | tr '_' '-')

        INPUT_PATH="${BASE_INPUT}/${INPUT}"
        OUTPUT_PATH="${BASE_OUTPUT}/${METHOD_DIR}/${INPUT}"

        echo "========================================"
        echo "Method : $METHOD"
        echo "Input  : $INPUT_PATH"
        echo "Output : $OUTPUT_PATH"
        echo "========================================"

        python -m cli.inference \
            --method "$METHOD" \
            --config "$CONFIG" \
            --input "$INPUT_PATH" \
            --output "$OUTPUT_PATH"

        if [ $? -ne 0 ]; then
            echo "ERROR: Failed on method=$METHOD, input=$INPUT. Continuing..."
        fi

    done
done

echo "Sweep complete."
