#!/bin/bash
CONFIG="config/pitch.yaml"

SMOOTHING_FACTOR=$(yq '.params.smoothing_factor' "$CONFIG")
INTERPOLATION_GAP=$(yq '.params.interpolation_gap' "$CONFIG")

python -m pitch --smoothing-factor "$SMOOTHING_FACTOR" --interpolation-gap "$INTERPOLATION_GAP"
