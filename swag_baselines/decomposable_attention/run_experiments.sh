#!/usr/bin/env bash

# Run skipthoughts with a bunch of different modes
export CUDA_VISIBLE_DEVICES=$1
if [ $1 == "0" ]; then
    echo "fuck! Numberbatch"
    python -m allennlp.run train train-numberbatch.json -s tmp/numberbatchJUSTS2 --include-package swag_baselines.decomposable_attention
    echo "fuck! NUMBERBATCH GOLD ONLY!!!!"
    python -m allennlp.run train train-numberbatch-goldonly.json -s tmp/numberbatchallJUSTS2 --include-package swag_baselines.decomposable_attention
elif [ $1 == "1" ]; then
    echo "fuck! Glove"
    python -m allennlp.run train train-glove-840.json -s tmp/glove840JUSTS2 --include-package swag_baselines.decomposable_attention
    echo "fuck! ELMO GOLD ONLY!!!!"
    python -m allennlp.run train train-elmo-goldonly.json -s tmp/elmo2allJUSTS2 --include-package swag_baselines.decomposable_attention
elif [ $1 == "2" ]; then
    echo "fuck! Elmo"
    python -m allennlp.run train train-elmo.json -s tmp/elmo2JUSTS2 --include-package swag_baselines.decomposable_attention
    echo "fuck! GLOVE GOLD ONLY!!!!"
    python -m allennlp.run train train-glove-goldonly-840.json -s tmp/gloveall840JUSTS2 --include-package swag_baselines.decomposable_attention
fi

