#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
echo "Sampling the candidates. remember to do this do this for all of the GPUS!"
python sample_candidates.py -fold $1