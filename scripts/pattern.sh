#!/bin/bash
CONFIG="config/pattern.yaml"

DEPTH=$(yq '.params.depth' "$CONFIG")
EMBED_DIM=$(yq '.params.embed_dim' "$CONFIG")
WINDOW_SIZE=$(yq '.params.window_size' "$CONFIG")

python -m pattern \
    --depth "$DEPTH" \
    --embed-dim "$EMBED_DIM" \
    --window-size "$WINDOW_SIZE"
