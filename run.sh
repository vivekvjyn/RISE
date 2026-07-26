#!/bin/bash
set -e

EXPERIMENT=${1:?"Usage: ./run.sh <experiment>\n\nExperiments: preprocess, pretrain, classification, clustering, pattern_recognition, synthesis"}

case "$EXPERIMENT" in
    preprocess)
        python -m preprocess \
            --smoothing-factor $(yq '.pitch.smoothing_factor' configs.yaml) \
            --interpolation-gap $(yq '.pitch.interpolation_gap' configs.yaml)
        ;;
    pretrain)
        python -m pretrain \
            --batch-size $(yq '.pretrain.batch_size' configs.yaml) \
            --depth $(yq '.pretrain.depth' configs.yaml) \
            --embed-dim $(yq '.pretrain.embed_dim' configs.yaml) \
            --epochs $(yq '.pretrain.epochs' configs.yaml) \
            --lr $(yq '.pretrain.lr' configs.yaml) \
            --out-dim $(yq '.pretrain.out_dim' configs.yaml) \
            --patience $(yq '.pretrain.patience' configs.yaml)
        ;;
    classification)
        for raga in abhogi begada kalyani mohanam sahana saveri sri; do
            python -m classification \
                --dataset "$raga" \
                --batch-size $(yq '.classification.batch_size' configs.yaml) \
                --catchup $(yq '.classification.catchup' configs.yaml) \
                --depth $(yq '.classification.depth' configs.yaml) \
                --early-stopping $(yq '.classification.early_stopping' configs.yaml) \
                --embed-dim $(yq '.classification.embed_dim' configs.yaml) \
                --epochs $(yq '.classification.epochs' configs.yaml) \
                --lr $(yq '.classification.lr' configs.yaml)
        done
        ;;
    clustering)
        python -m clustering \
            --batch-size $(yq '.clustering.batch_size' configs.yaml) \
            --catchup $(yq '.clustering.catchup' configs.yaml) \
            --depth $(yq '.clustering.depth' configs.yaml) \
            --early-stopping $(yq '.clustering.early_stopping' configs.yaml) \
            --embed-dim $(yq '.clustering.embed_dim' configs.yaml) \
            --epochs $(yq '.clustering.epochs' configs.yaml) \
            --lr $(yq '.clustering.lr' configs.yaml)
        ;;
    pattern_recognition)
        python -m pattern_recognition \
            --depth $(yq '.pattern_recognition.depth' configs.yaml) \
            --embed-dim $(yq '.pattern_recognition.embed_dim' configs.yaml) \
            --window-size $(yq '.pattern_recognition.window_size' configs.yaml)
        ;;
    synthesis)
        python -m synthesis \
            --batch-size $(yq '.synthesis.batch_size' configs.yaml) \
            --extract-batch-size $(yq '.synthesis.extract_batch_size' configs.yaml) \
            --depth $(yq '.synthesis.depth' configs.yaml) \
            --embed-dim $(yq '.synthesis.embed_dim' configs.yaml) \
            --epochs $(yq '.synthesis.epochs' configs.yaml) \
            --lr $(yq '.synthesis.lr' configs.yaml) \
            --patience $(yq '.synthesis.patience' configs.yaml) \
            --out-dim $(yq '.synthesis.out_dim' configs.yaml) \
            --accumulation-steps $(yq '.synthesis.accumulation_steps' configs.yaml)
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Experiments: preprocess, pretrain, classification, clustering, pattern_recognition, synthesis"
        exit 1
        ;;
esac
