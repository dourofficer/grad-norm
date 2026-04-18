#!/bin/bash
# sweep_ablate.sh — Evaluate entropy scores → accuracy TSVs.
#
# Reads per-trajectory JSONs written by sweep_entropy.sh and writes
#   outputs/entropy/<model>/<subset>/accuracy_{ascending,descending}.tsv
#
# Usage:
#   bash scripts/entropy/sweep_ablate.sh
#   TEMPS="1.0 0.5" KS="1 3 5 10" bash scripts/entropy/sweep_ablate.sh

set -euo pipefail

KS="${KS:-1 3 5 10}"
TEMPS="${TEMPS:-1.0}"
BASE="outputs/entropy"

MODELS=(
    "qwen3-8b"
    "llama-3.1-8b"
)

SUBSETS=(
    "hand-crafted"
    "algorithm-generated"
)

# Collect model tags (temp-suffixed variants match sweep_entropy.sh naming)
TAGS=()
for model_tag in "${MODELS[@]}"; do
    for temp in $TEMPS; do
        if [[ "$temp" == "1.0" ]]; then
            TAGS+=("${model_tag}")
        else
            TAGS+=("${model_tag}-t${temp}")
        fi
    done
done

python -m gradnorm.entropy_ablate \
    --base_dir "$BASE" \
    --models   "${TAGS[@]}" \
    --subsets  "${SUBSETS[@]}" \
    --ks       $KS \
    --orders   ascending descending

echo "All done."
