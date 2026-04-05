#!/bin/bash
HUB="/data/projects/32000010/hub"
MODEL_PATH="${HUB}/Qwen/Qwen3-8B"
MODEL_NAME="qwen3-8b"
SUBSETS=("algorithm-generated" "hand-crafted")

for SUBSET in "${SUBSETS[@]}"; do
    python -m sal.gradient \
        --model "${MODEL_PATH}" \
        --input "ww/${SUBSET}" \
        --output "outputs/sal/grads/${MODEL_NAME}/${SUBSET}" \
        --target_params q/34 k/34 v/35 k/21 v/24 \
        --max_tokens 8192
done