# fasttext baseline

This is a wrapper around the fasttext library for getting results on SWAG. See [https://fasttext.cc](https://fasttext.cc) for installation info. 

To use this, first run ```prep_data.py``` to prepare the data in a format that fasttext can handle. Then, you can train a fasttext model and obtain predictions using 

```
~/tools/fastText/fasttext supervised -input train.txt -output model -lr 0.1 -wordNgrams 5 -epoch 50
~/tools/fastText/fasttext predict-prob model.bin val.txt 2 > val_preds.txt
```
Then compute performance using ```python compute_performance.py```.