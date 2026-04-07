#!/bin/bash

GRADIENT_DIR="/home/hoangpham/exchange"
MODEL="qwen3-8b"
SUBSETS=("algorithm-generated")
CONFIGS=("k/21" "k/34" "v/24" "v/35")

for SUBSET in "${SUBSETS[@]}"; do
  for CONFIG in "${CONFIGS[@]}"; do
    echo "Running SUBSET=${SUBSET}, CONFIG=${CONFIG}"
    python -m sal.score \
      --subset "$SUBSET" \
      --config "${GRADIENT_DIR}/${SUBSET}/${CONFIG}" \
      --output "outputs/sal/scores/${MODEL}/${SUBSET}/${CONFIG}"
  done
done