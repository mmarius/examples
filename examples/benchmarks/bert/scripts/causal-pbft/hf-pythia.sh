#!/usr/bin/env bash

PROJECT_PATH=/nethome/mmosbach/projects/mosaicml_examples/examples/benchmarks/bert

# run setup
source $PROJECT_PATH/scripts/setup.sh

# parse arguments from submit file
REVISION=$1
NUM_SAMPLES=$2
SEED=$3

# run the fine-tuning script
composer -v $PROJECT_PATH/sequence_classification.py \
    $PROJECT_PATH/yamls/finetuning/causal-pbft/pythia.yaml \
    model.revision=$REVISION \
    train_loader.num_samples=$NUM_SAMPLES \
    seed=$SEED