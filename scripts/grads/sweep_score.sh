#!/usr/bin/env bash
# scripts/grads/sweep_score.sh
#
# Usage:
#   bash scripts/grads/sweep_score.sh                  # all defaults
#   bash scripts/grads/sweep_score.sh --ks 1 3 5       # override specific args
#   GRAD_DIR=/data/grads bash scripts/grads/sweep_score.sh

set -euo pipefail

# ── Hardcoded defaults ────────────────────────────────────────────────────────

MODELS=(
    llama-3.1-8b-kl_12
    llama-3.1-8b-kl_14
    llama-3.1-8b-kl_16
    llama-3.1-8b-kl_18
    llama-3.1-8b-kl_20
    llama-3.1-8b-kl_22
    llama-3.1-8b-kl_24
    llama-3.1-8b-kl_26
    llama-3.1-8b-kl_28
    llama-3.1-8b-kl_30
    llama-3.1-8b-kl_uniform
    llama-3.1-8b-ntp
    qwen3-8b-kl_12
    qwen3-8b-kl_14
    qwen3-8b-kl_16
    qwen3-8b-kl_18
    qwen3-8b-kl_20
    qwen3-8b-kl_22
    qwen3-8b-kl_24
    qwen3-8b-kl_26
    qwen3-8b-kl_28
    qwen3-8b-kl_30
    qwen3-8b-kl_uniform
    qwen3-8b-ntp
)

SUBSETS=(
    hand-crafted
    algorithm-generated
)

KS=(1 3 5 10)

DATA_DIR="ww"
GRAD_DIR="${GRAD_DIR:-outputs/grads}"
OUT_DIR="${OUT_DIR:-outputs/grads}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Models  : ${MODELS[*]}"
echo "  Subsets : ${SUBSETS[*]}"
echo "  ks      : ${KS[*]}"
echo "  Data    : $DATA_DIR"
echo "  Grads   : $GRAD_DIR"
echo "  Out     : $OUT_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -m grads.score          \
    --models   "${MODELS[@]}"  \
    --subsets  "${SUBSETS[@]}" \
    --ks       "${KS[@]}"      \
    --data-dir "$DATA_DIR"     \
    --grad-dir "$GRAD_DIR"     \
    --out-dir  "$OUT_DIR"      