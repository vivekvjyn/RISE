#!/bin/bash
set -e

EXPERIMENT=${1:?"Usage: ./run.sh <experiment>\n\nExperiments: preprocess, pretrain, classification, clustering, pattern_recognition, synthesis"}

case "$EXPERIMENT" in
    preprocess)
        python -m preprocess \
            --smoothing-factor $(yq '.pitch.smoothing_factor' config.yaml) \
            --interpolation-gap $(yq '.pitch.interpolation_gap' config.yaml)
        ;;
    pretrain)
        python -m pretrain \
            --batch-size $(yq '.pretrain.batch_size' config.yaml) \
            --depth $(yq '.pretrain.depth' config.yaml) \
            --embed-dim $(yq '.pretrain.embed_dim' config.yaml) \
            --epochs $(yq '.pretrain.epochs' config.yaml) \
            --lr $(yq '.pretrain.lr' config.yaml) \
            --out-dim $(yq '.pretrain.out_dim' config.yaml) \
            --patience $(yq '.pretrain.patience' config.yaml)
        ;;
    classification)
        for raga in abhogi begada kalyani mohanam sahana saveri sri; do
            python -m classification \
                --dataset "$raga" \
                --batch-size $(yq '.classification.batch_size' config.yaml) \
                --catchup $(yq '.classification.catchup' config.yaml) \
                --depth $(yq '.classification.depth' config.yaml) \
                --early-stopping $(yq '.classification.early_stopping' config.yaml) \
                --embed-dim $(yq '.classification.embed_dim' config.yaml) \
                --epochs $(yq '.classification.epochs' config.yaml) \
                --lr $(yq '.classification.lr' config.yaml)
        done
        ;;
    clustering)
        python -m clustering \
            --batch-size $(yq '.clustering.batch_size' config.yaml) \
            --catchup $(yq '.clustering.catchup' config.yaml) \
            --depth $(yq '.clustering.depth' config.yaml) \
            --early-stopping $(yq '.clustering.early_stopping' config.yaml) \
            --embed-dim $(yq '.clustering.embed_dim' config.yaml) \
            --epochs $(yq '.clustering.epochs' config.yaml) \
            --lr $(yq '.clustering.lr' config.yaml)
        ;;
    pattern_recognition)
        python -m pattern_recognition \
            --depth $(yq '.pattern_recognition.depth' config.yaml) \
            --embed-dim $(yq '.pattern_recognition.embed_dim' config.yaml) \
            --window-size $(yq '.pattern_recognition.window_size' config.yaml)
        ;;
    synthesis)
        python -m synthesis \
            --batch-size $(yq '.synthesis.batch_size' config.yaml) \
            --extract-batch-size $(yq '.synthesis.extract_batch_size' config.yaml) \
            --depth $(yq '.synthesis.depth' config.yaml) \
            --embed-dim $(yq '.synthesis.embed_dim' config.yaml) \
            --epochs $(yq '.synthesis.epochs' config.yaml) \
            --lr $(yq '.synthesis.lr' config.yaml) \
            --patience $(yq '.synthesis.patience' config.yaml) \
            --out-dim $(yq '.synthesis.out_dim' config.yaml) \
            --accumulation-steps $(yq '.synthesis.accumulation_steps' config.yaml)
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Experiments: preprocess, pretrain, classification, clustering, pattern_recognition, synthesis"
        exit 1
        ;;
esac
