#!/bin/bash
# sweep_eval.sh
# Predict + evaluate every inference output under outputs/<MODEL>.
#
# Usage:
#   bash sweep_eval.sh                          # defaults below
#   MODEL=qwen3-8b  KS="1 5 10" bash sweep_eval.sh
#   MODEL=llama-3.1-8b KS="1 5 10" bash sweep_eval.sh
#
# What this does:
#   Calls cli.evaluate --predict_first, which:
#     1. Auto-discovers all <base_output>/<strategy>/<subset> dirs
#     2. Runs cli.predict on each (method inferred from the strategy folder name)
#     3. Prints and saves the accuracy sweep table
#
# To skip the predict phase (predictions already populated):
#   bash sweep_eval.sh --skip_predict

set -e

MODEL="${MODEL:-gpt-oss-20b}"
BASE_OUTPUT="outputs/${MODEL}"
KS="${KS:-1 5 10}"
SAVE="${SAVE:-${BASE_OUTPUT}/sweep_results.tsv}"

PREDICT_FLAG="--predict_first"
if [[ "$1" == "--skip_predict" ]]; then
    PREDICT_FLAG=""
    echo "Skipping predict phase (--skip_predict passed)."
fi

python -m cli.evaluate \
    --sweep \
    --base_output "${BASE_OUTPUT}" \
    --ks          ${KS} \
    --save        "${SAVE}" \
    ${PREDICT_FLAG}

echo ""
echo "Done. Results saved to ${SAVE}"