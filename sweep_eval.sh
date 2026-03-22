#!/bin/bash
# sweep_eval.sh
# For every inference output, runs cli.predict then cli.evaluate.
#
# Usage:
#   bash sweep_eval.sh                         # defaults below
#   MODEL=qwen3-8b KS="1 5 10" bash sweep_eval.sh
#   MODEL=llama-3.1-8b KS="1 5 10" bash sweep_eval.sh

set -e

MODEL="${MODEL:-gpt-oss-20b}"
BASE_OUTPUT="outputs/${MODEL}"
KS="${KS:-1 5 10}"
SAVE="${SAVE:-${BASE_OUTPUT}/sweep_results.tsv}"

# method folder name → predict --method value
declare -A PREDICT_METHOD=(
    ["all-at-once"]="all_at_once"
    ["step-by-step-full"]="step_by_step"
    ["step-by-step-partial"]="step_by_step"
)

METHODS=("all-at-once" "step-by-step-full" "step-by-step-partial")
SUBSETS=("hand-crafted" "algorithm-generated")

# ------------------------------------------------------------------ #
# 1. Predict — populate predictions for every output dir             #
# ------------------------------------------------------------------ #
echo "========================================"
echo "Phase 1: Populating predictions"
echo "========================================"

CONFIGS=()

for METHOD in "${METHODS[@]}"; do
    for SUBSET in "${SUBSETS[@]}"; do
        DIR="${BASE_OUTPUT}/${METHOD}/${SUBSET}"

        if [ ! -d "${DIR}" ]; then
            echo "Skipping (not found): ${DIR}"
            continue
        fi

        PREDICT_M="${PREDICT_METHOD[$METHOD]}"
        echo ""
        echo "--- predict: method=${PREDICT_M}  dir=${DIR} ---"

        python -m cli.predict \
            --dir    "${DIR}" \
            --method "${PREDICT_M}"

        CONFIGS+=("${DIR}")
    done
done

# ------------------------------------------------------------------ #
# 2. Evaluate — produce the results table                            #
# ------------------------------------------------------------------ #
echo ""
echo "========================================"
echo "Phase 2: Evaluating (ks: ${KS})"
echo "========================================"

# Build --ks argument from space-separated string
KS_ARGS=""
for k in $KS; do
    KS_ARGS="${KS_ARGS} ${k}"
done

# Build --configs argument from the discovered dirs
CONFIGS_ARGS="${CONFIGS[*]}"

python -m cli.evaluate \
    --sweep \
    --configs ${CONFIGS_ARGS} \
    --ks      ${KS_ARGS} \
    --save    "${SAVE}"

echo ""
echo "Done. Results saved to ${SAVE}"