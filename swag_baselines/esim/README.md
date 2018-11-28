# ESIM on swag

ESIM seems to work pretty well on SWAG, so here's it in action. You can train using the following command:

```
allennlp train train-glove.json -s tmp/glove --include-package swag_baselines.esim
```

Once you've trained a model, you can run it with:

```
python -m predict tmp/glove/model.tar.gz ../../data/val.csv --cuda-device 0 --include-package swag_baselines.esim --output-file lol.csv
```

And you can verify the results with

```
import pandas as pd
print( (pd.read_csv('lol.csv').pred == pd.read_csv('../../data/val.csv').label).mean() )
```

