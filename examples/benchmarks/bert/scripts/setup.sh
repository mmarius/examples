#!/usr/bin/env bash

# rename gpus
source /nethome/mmosbach/projects/mosaicml_examples/examples/benchmarks/bert/scripts/rename_gpus.sh

# setup basic paths
export CACHE_BASE_DIR=/data/users/mmosbach/logs/mosaicml_examples/cache
export OUTPUT_DIR=/data/users/mmosbach/logs/mosaicml_examples/logfiles

# set paths for transformers, datasets
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_CACHE=$CACHE_BASE_DIR/transformers
export HF_DATASETS_CACHE=$CACHE_BASE_DIR/datasets

# setup wandb
export WANDB_DISABLED=false
export WANDB_API_KEY=TODO
export WANDB_USERNAME=mmosbach
export WANDB_DIR=$CACHE_BASE_DIR/wandb/logs
export WANDB_CACHE_DIR=$CACHE_BASE_DIR/wandb/artifacts
export WANDB_CONFIG_DIR=$CACHE_BASE_DIR/wandb/configs

# create cash dirs if they don't exist yet
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_DATASETS_CACHE
mkdir -p $WANDB_DIR
mkdir -p $WANDB_CACHE_DIR
mkdir -p $WANDB_CONFIG_DIR


# install dependencies
# TODO(mm): run this when building the image
pip install -r /nethome/mmosbach/projects/mosaicml_examples/examples/benchmarks/bert/requirements.txt --user

# print some stuff
echo $HOSTNAME
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
