#!/bin/bash
CONFIG="config/synth.yaml"

BATCH_SIZE=$(yq -r '.params.batch_size' "$CONFIG")
EXTRACT_BATCH_SIZE=$(yq -r '.params.extract_batch_size' "$CONFIG")
DEPTH=$(yq -r '.params.depth' "$CONFIG")
EMBED_DIM=$(yq -r '.params.embed_dim' "$CONFIG")
EPOCHS=$(yq -r '.params.epochs' "$CONFIG")
LR=$(yq -r '.params.lr' "$CONFIG")
PATIENCE=$(yq -r '.params.patience' "$CONFIG")
OUT_DIM=$(yq -r '.params.out_dim' "$CONFIG")
ACCUMULATION_STEPS=$(yq -r '.params.accumulation_steps' "$CONFIG")

python -m synth \
    --batch-size "$BATCH_SIZE" \
    --extract-batch-size "$EXTRACT_BATCH_SIZE" \
    --depth "$DEPTH" \
    --embed-dim "$EMBED_DIM" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --patience "$PATIENCE" \
    --out-dim "$OUT_DIM" \
    --accumulation-steps "$ACCUMULATION_STEPS"
