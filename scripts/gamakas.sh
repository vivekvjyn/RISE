#!/bin/bash
CONFIG="config/gamakas.yaml"

RESULTS_DIR="results/clustering"
if [ -d "$RESULTS_DIR" ]; then
    rm -rf "$RESULTS_DIR"
fi

BATCH_SIZE=$(yq '.params.batch_size' "$CONFIG")
CATCHUP=$(yq '.params.catchup' "$CONFIG")
DEPTH=$(yq '.params.depth' "$CONFIG")
EARLY_STOPPING=$(yq '.params.early_stopping' "$CONFIG")
EMBED_DIM=$(yq '.params.embed_dim' "$CONFIG")
EPOCHS=$(yq '.params.epochs' "$CONFIG")
LR=$(yq '.params.lr' "$CONFIG")

python -m gamakas \
    --batch-size "$BATCH_SIZE" \
    --catchup "$CATCHUP" \
    --depth "$DEPTH" \
    --early-stopping "$EARLY_STOPPING" \
    --embed-dim "$EMBED_DIM" \
    --epochs "$EPOCHS" \
    --lr "$LR"
