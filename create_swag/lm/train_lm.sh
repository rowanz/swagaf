#!/usr/bin/env bash

FOLD_ID=$1
NUM_GPUS=3
export CUDA_VISIBLE_DEVICES=$((FOLD_ID % NUM_GPUS))
echo "Sampling the candidates. remember to do this do this for all of the GPUS and to pretrain first!"
python train_lm.py -fold $1
