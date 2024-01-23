#!/usr/bin/env bash

PROJECT_PATH=/nethome/mmosbach/projects/mosaicml_examples/examples/benchmarks/bert

# run setup
source $PROJECT_PATH/scripts/setup.sh

# parse arguments from submit file
NUM_SAMPLES=$1
SEED=$2

# run the fine-tuning script
composer $PROJECT_PATH/sequence_classification.py \
    $PROJECT_PATH/yamls/finetuning/pbft/roberta-large.yaml \
    train_loader.num_samples=$NUM_SAMPLES \
    seed=$SEED
