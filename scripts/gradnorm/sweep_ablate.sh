#!/bin/bash
# sweep_ablate.sh — Run the ablation sweep across models × subsets × k × norm-type.
#
# Usage:
#   bash scripts/gradnorm/sweep_ablate.sh
#   BASE_DIR="outputs/gradnorm-v2" scripts/gradnorm/sweep_ablate.sh
#   BASE_DIR="outputs/gradnorm-v2" MODELS="qwen3-8b-full" KS="1 5" scripts/gradnorm/sweep_ablate.sh
#   BASE_DIR="outputs/gradnorm-v2" MODELS="qwen3-8b-full" SUBSETS="hand-crafted" KS="1 5" scripts/gradnorm/sweep_ablate.sh

set -euo pipefail

BASE_DIR="${BASE_DIR:-outputs/gradnorm}"
# MODELS="${MODELS:-qwen3-8b llama-3.1-8b}"
# MODELS="${MODELS:-qwen3-8b-kl_15 qwen3-8b-kl_20 qwen3-8b-kl_uniform llama-3.1-8b-kl_15 llama-3.1-8b-kl_20 llama-3.1-8b-kl_uniform}"
# MODELS=(qwen3-14b-kl_uniform)
MODELS="${MODELS:-qwen3-14b-ntp qwen3-14b-kl_uniform qwen3-32b-ntp qwen3-32b-kl_uniform}"
SUBSETS="${SUBSETS:-algorithm-generated hand-crafted}"
KS="${KS:-1 3 5 10}"
WORKERS="${WORKERS:-32}"

# shellcheck disable=SC2086
python -m gradnorm.ablate \
    --base_dir  $BASE_DIR \
    --models    $MODELS     \
    --subsets   $SUBSETS    \
    --ks        $KS         \
    --workers   $WORKERS

echo "All done."