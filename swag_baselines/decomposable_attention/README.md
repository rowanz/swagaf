#to run

python -m allennlp.run train train.json -s tmp/output0 --include-package swag_baselines.decomposable_attention

python -m allennlp.run evaluate tmp/output0/best.th --evaluation-data-file ../../data/test.csv