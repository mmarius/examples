#!/usr/bin/env bash

# rename gpus
source /nethome/mmosbach/projects/mosaicml_examples/examples/benchmarks/bert/scripts/rename_gpus.sh

# setup basic paths
export CACHE_BASE_DIR=/data/users/mmosbach/logs/mosaicml_examples/cache
export OUTPUT_DIR=/data/users/mmosbach/logs/mosaicml_examples/logfiles

# setup wandb
export WANDB_DISABLED=false
export WANDB_API_KEY=483d16b0cbe267d445706c02abeacc9547c967c8
export WANDB_USERNAME=mmosbach
export WANDB_CACHE_DIR=$CACHE_BASE_DIR/wandb
export WANDB_CONFIG_DIR=$WANDB_CACHE_DIR

# create cash dirs if they don't exist yet
mkdir -p $WANDB_CACHE_DIR

# install dependencies
# TODO(mm): run this when building the image
pip install -r /nethome/mmosbach/projects/mosaicml_examples/examples/benchmarks/bert/requirements.txt --user

# print some stuff
echo $HOSTNAME
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
