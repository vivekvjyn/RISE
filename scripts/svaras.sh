#!/bin/bash
CONFIG="config/svaras.yaml"

BATCH_SIZE=$(yq '.params.batch_size' "$CONFIG")
CATCHUP=$(yq '.params.catchup' "$CONFIG")
DEPTH=$(yq '.params.depth' "$CONFIG")
EARLY_STOPPING=$(yq '.params.early_stopping' "$CONFIG")
EMBED_DIM=$(yq '.params.embed_dim' "$CONFIG")
EPOCHS=$(yq '.params.epochs' "$CONFIG")
LR=$(yq '.params.lr' "$CONFIG")

datasets=(abhogi begada kalyani mohanam sahana saveri sri)

for dataset in "${datasets[@]}"; do
    python -m svaras \
        --dataset "$dataset" \
        --batch-size "$BATCH_SIZE" \
        --catchup "$CATCHUP" \
        --depth "$DEPTH" \
        --early-stopping "$EARLY_STOPPING" \
        --epochs "$EPOCHS" \
        --embed-dim "$EMBED_DIM" \
        --lr "$LR"
done
