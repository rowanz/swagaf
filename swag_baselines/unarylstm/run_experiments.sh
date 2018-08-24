#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1

if [ $1 == "0" ]; then
    echo "fuck! LSTM Numberbatch"
    python -m allennlp.run train train-lstmbasic-numberbatch.json -s tmp/lstmbasicnumberbatch --include-package swag_baselines.unarylstm
    echo "fuck! LSTM NUMBERBATCH GOLD ONLY!!!!"
    python -m allennlp.run train train-lstmbasic-numberbatch-goldonly.json -s tmp/lstmbasicnumberbatchall --include-package swag_baselines.unarylstm
elif [ $1 == "1" ]; then
    echo "fuck! LSTM GloVe"
    python -m allennlp.run train train-lstmbasic-glove.json -s tmp/lstmbasicglove --include-package swag_baselines.unarylstm
    echo "fuck! LSTM GLOVE GOLD ONLY!!!!"
    python -m allennlp.run train train-lstmbasic-glove-goldonly.json -s tmp/lstmbasicgloveall --include-package swag_baselines.unarylstm
elif [ $1 == "2" ]; then
    echo "fuck! LSTM Elmo"
    python -m allennlp.run train train-lstmbasic-elmo.json -s tmp/lstmbasicelmo --include-package swag_baselines.unarylstm
    echo "fuck! LSTM ELMO GOLD ONLY!!!!"
    python -m allennlp.run train train-lstmbasic-elmo-goldonly.json -s tmp/lstmbasicelmoall --include-package swag_baselines.unarylstm
fi

