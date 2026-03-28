#!/bin/bash

# Define the arrays
MODELS=("qwen3-8b" "llama-3.1-8b")
SUBSETS=("hand-crafted" "algorithm-generated")

# Loop through each model and subset
for MODEL in "${MODELS[@]}"; do
  for SUBSET in "${SUBSETS[@]}"; do

    echo "Starting sweep for Model: $MODEL | Subset: $SUBSET"

    python -m ablation.length_dist   --model "$MODEL" --subset "$SUBSET"
    python -m ablation.score_dist    --model "$MODEL" --subset "$SUBSET"
    python -m ablation.distance_dist --model "$MODEL" --subset "$SUBSET"

  done
done

echo "Plot complete!"