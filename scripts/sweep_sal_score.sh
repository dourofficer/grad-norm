#!/bin/bash

MODELS=("qwen3-8b" "llama-3.1-8b")
SUBSETS=("hand-crafted" "algorithm-generated")

for MODEL in "${MODELS[@]}"; do
    for SUBSET in "${SUBSETS[@]}"; do
        echo "=== Running: $MODEL / $SUBSET ==="
        python -m cli.sal_score \
            --input  "sal_outputs/grads/$MODEL/$SUBSET" \
            --output "sal_outputs/results/$MODEL/$SUBSET" \
            --data ww/$SUBSET \
            --reference all
    done
done