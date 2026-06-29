#!/bin/bash
CONFIG="config/simclr.yaml"

BATCH_SIZE=$(yq '.params.batch_size' "$CONFIG")
DEPTH=$(yq '.params.depth' "$CONFIG")
EMBED_DIM=$(yq '.params.embed_dim' "$CONFIG")
EPOCHS=$(yq '.params.epochs' "$CONFIG")
LR=$(yq '.params.lr' "$CONFIG")
OUT_DIM=$(yq '.params.out_dim' "$CONFIG")
PATIENCE=$(yq '.params.patience' "$CONFIG")

python -m simclr \
    --batch-size "$BATCH_SIZE" \
    --depth "$DEPTH" \
    --embed-dim "$EMBED_DIM" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --out-dim "$OUT_DIM" \
    --patience "$PATIENCE"
