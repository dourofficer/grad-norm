#!/bin/bash
# sweep_ablate.sh — Run the ablation sweep across models × subsets × k × norm-type.
#
# Usage:
#   bash scripts/gradnorm/sweep_ablate.sh
#   BASE_DIR="outputs/gradnorm-v2" scripts/gradnorm/sweep_ablate.sh
#   BASE_DIR="outputs/gradnorm-v2" MODELS="qwen3-8b-full" KS="1 5" scripts/gradnorm/sweep_ablate.sh
#   BASE_DIR="outputs/gradnorm-v2" MODELS="qwen3-8b-full" SUBSETS="hand-crafted" KS="1 5" scripts/gradnorm/sweep_ablate.sh

set -euo pipefail

BASE_DIR="outputs/gradnorm-losses"
SUBSETS="${SUBSETS:-algorithm-generated hand-crafted}"
KS="${KS:-1 3 5 10}"
WORKERS="${WORKERS:-32}"

# Manually listed models from outputs/gradnorm-losses
MODELS=(
    llama-3.1-8b-kl_12
    llama-3.1-8b-kl_14
    llama-3.1-8b-kl_15
    llama-3.1-8b-kl_16
    llama-3.1-8b-kl_20
    llama-3.1-8b-kl_22
    llama-3.1-8b-kl_24
    llama-3.1-8b-kl_26
    llama-3.1-8b-kl_28
    llama-3.1-8b-kl_30
    llama-3.1-8b-kl_uniform
    llama-3.1-8b-ntp
    # qwen3-8b-kl_12
    # qwen3-8b-kl_14
    # qwen3-8b-kl_15
    # qwen3-8b-kl_16
    # qwen3-8b-kl_18
    # qwen3-8b-kl_20
    # qwen3-8b-kl_22
    # qwen3-8b-kl_24
    # qwen3-8b-kl_26
    # qwen3-8b-kl_28
    # qwen3-8b-kl_30
    # qwen3-8b-kl_uniform
    # qwen3-8b-ntp
)

echo "Sweeping ${#MODELS[@]} models..."

# shellcheck disable=SC2086
python -m gradnorm.ablate \
    --base_dir  "$BASE_DIR" \
    --models    "${MODELS[@]}" \
    --subsets   $SUBSETS    \
    --ks        $KS         \
    --workers   $WORKERS

echo "All done."
