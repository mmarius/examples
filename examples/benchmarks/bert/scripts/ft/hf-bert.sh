#!/usr/bin/env bash

PROJECT_PATH=/nethome/mmosbach/projects/mosaicml_examples/examples/benchmarks/bert

# run setup
source $PROJECT_PATH/scripts/setup.sh

# parse arguments from submit file
SEED=$1

# run the fine-tuning script
composer $PROJECT_PATH/sequence_classification.py \
    $PROJECT_PATH/yamls/finetuning/hf-bert/bert-large-uncased.yaml \
    seed=$SEED
