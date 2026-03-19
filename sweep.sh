#!/usr/bin/env bash

MODELS=(
    "/data/hoang/resources/models/meta-llama/Llama-3.1-8B-Instruct"
    "/data/hoang/resources/models/Qwen/Qwen3-8B"
)
SUBSETS=("hand-crafted" "algorithm-generated")
LAYERS=("lm_head" "out_proj" "final_layer")

mkdir -p outputs

for MODEL in "${MODELS[@]}"; do
    MODEL_SLUG="${MODEL##*/}"
    for SUBSET in "${SUBSETS[@]}"; do
        for LAYER in "${LAYERS[@]}"; do
            OUTPUT="outputs/${MODEL_SLUG}_${SUBSET}_${LAYER}.json"
            echo "=== ${MODEL_SLUG} | ${SUBSET} | ${LAYER} ==="

            CUDA_VISIBLE_DEVICES=7 python main.py \
                --dataset   ww \
                --model     "$MODEL" \
                --layer     "$LAYER" \
                --max_len   12000 \
                --strategy  split \
                --subset    "$SUBSET" \
                --output    "$OUTPUT" \
                --verbose

        done
    done
done
