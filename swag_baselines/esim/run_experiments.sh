#!/usr/bin/env bash

# Run experiments with a bunch of different models
export CUDA_VISIBLE_DEVICES=$1
if [ $1 == "2" ]; then
    echo "fuck! Numberbatch"
    python -m allennlp.run train train-numberbatch.json -s tmp/numberbatch --include-package swag_baselines.esim
    echo "fuck! NUMBERBATCH GOLD ONLY!!!!"
    python -m allennlp.run train train-numberbatch-goldonly.json -s tmp/numberbatchgold --include-package swag_baselines.esim
elif [ $1 == "1" ]; then
    echo "fuck! Glove"
    python -m allennlp.run train train-glove.json -s tmp/glove --include-package swag_baselines.esim
#    echo "fuck! ELMO GOLD ONLY!!!!"
#    python -m allennlp.run train train-elmo-goldonly.json -s tmp/elmogold --include-package swag_baselines.esim
elif [ $1 == "0" ]; then
    echo "fuck! Elmo"
    python -m allennlp.run train train-elmo.json -s tmp/elmo --include-package swag_baselines.esim
    echo "fuck! GLOVE GOLD ONLY!!!!"
    python -m allennlp.run train train-glove-goldonly.json -s tmp/glovegold --include-package swag_baselines.esim
fi

