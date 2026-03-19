#!/usr/bin/env bash
 
CUDA_VISIBLE_DEVICES=7 python main.py \
    --dataset   ww \
    --model     /data/hoang/resources/models/Qwen/Qwen3-8B \
    --layer     final_layer \
    --max_len   16000 \
    --strategy  split \
    --subset    hand-crafted \
    --output    outputs/qwen_lmhead.json \
    --verbose


CUDA_VISIBLE_DEVICES=7 python main.py \
    --dataset   ww \
    --model     /data/hoang/resources/models/meta-llama/Llama-3.1-8B-Instruct \
    --layer     lm_head \
    --max_len   16000 \
    --strategy  split \
    --subset    algorithm-generated \
    --output    outputs/qwen_lmhead.json \
    --verbose