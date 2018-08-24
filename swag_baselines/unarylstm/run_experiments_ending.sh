#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1

echo "ONLY ENDING!!!!"
if [ $1 == "0" ]; then
    echo "fuck! LSTM Numberbatch"
    python -m allennlp.run train train-lstmbasic-numberbatch-endingonly.json -s tmp/lstmbasicnumberbatch2 --include-package swag_baselines.unarylstm
    echo "fuck! LSTM NUMBERBATCH GOLD ONLY!!!!"
    python -m allennlp.run train train-lstmbasic-numberbatch-goldonly-endingonly.json -s tmp/lstmbasicnumberbatchgold2 --include-package swag_baselines.unarylstm
elif [ $1 == "1" ]; then
    echo "fuck! LSTM GloVe"
    python -m allennlp.run train train-lstmbasic-glove-endingonly.json -s tmp/lstmbasicglove2 --include-package swag_baselines.unarylstm
    echo "fuck! LSTM GLOVE GOLD ONLY!!!!"
    python -m allennlp.run train train-lstmbasic-glove-goldonly-endingonly.json -s tmp/lstmbasicglovegold2 --include-package swag_baselines.unarylstm
elif [ $1 == "2" ]; then
    echo "fuck! LSTM Elmo"
    python -m allennlp.run train train-lstmbasic-elmo-endingonly.json -s tmp/lstmbasicelmo2 --include-package swag_baselines.unarylstm
    echo "fuck! LSTM ELMO GOLD ONLY!!!!"
    python -m allennlp.run train train-lstmbasic-elmo-goldonly-endingonly.json -s tmp/lstmbasicelmogold2 --include-package swag_baselines.unarylstm
fi

