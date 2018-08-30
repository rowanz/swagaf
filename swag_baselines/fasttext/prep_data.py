import pandas as pd
from tqdm import tqdm
from allennlp.common.util import get_spacy_model

USE_ONLY_GOLD_EXAMPLES = False


spacy_model = get_spacy_model("en_core_web_sm", pos_tags=False, parse=False, ner=False)
def _tokenize(sent):
    return ' '.join([x.orth_.lower() for x in spacy_model(sent)])


for split in ('train', 'val', 'test'):
    df = pd.read_csv('../../data/{}.csv'.format(split))
    df['distractor-3'].fillna('', inplace=True)
    if USE_ONLY_GOLD_EXAMPLES and split == 'train':
        oldsize = df.shape[0]
        df = df[df['gold-source'].str.startswith('gold')]
        print("Going from {} -> {} items in train".format(oldsize, df.shape[0]))

    with open('{}-{}.txt'.format(split, 'goldonly' if USE_ONLY_GOLD_EXAMPLES else 'genalso'), 'w') as f:
        for _, item in tqdm(df.iterrows()):
            # num_distractors = 4 if (len(item['distractor-3']) != 0 and split == 'train') else 3
            num_distractors = 3
            prefix_tokenized = '{} '.format(_tokenize(item['sent1']))
            gold_ending_tokenized = _tokenize('{} {}'.format(item['sent2'], item['gold-ending']))
            for i in range(num_distractors if split == 'train' else 1):
                f.write('__label__gold {}\n'.format(prefix_tokenized + gold_ending_tokenized))
            for i in range(num_distractors):
                f.write('__label__fake {}\n'.format(prefix_tokenized + _tokenize('{} {}'.format(item['sent2'], item['distractor-{}'.format(i)]))))

# let's just automate this...
"""
Unigrams
~/tools/fastText/fasttext supervised -input train-goldonly.txt -output model -lr 0.01 -wordNgrams 1 -epoch 5
~/tools/fastText/fasttext predict-prob model.bin val-goldonly.txt 2 > valpreds-goldonly.txt
python compute_performance.py valpreds-goldonly.txt



------------
accuracy 29.2% with 5-grams
~/tools/fastText/fasttext supervised -input train.txt -output model -lr 0.1 -wordNgrams 5 -epoch 50
~/tools/fastText/fasttext predict-prob model.bin val.txt 2 > val_preds.txt
python compute_performance.py
rm val_preds.txt



~/tools/fastText/fasttext predict-prob model.bin test.txt 2 > val_preds.txt
python compute_performance.py
rm val_preds.txt


dafuq
~/tools/fastText/fasttext supervised -input train-genalso.txt -output model -lr 0.921 -wordNgrams 1 -epoch 45
~/tools/fastText/fasttext predict-prob model.bin val-goldonly.txt 2 > preds.txt
python compute_performance.py preds.txt

"""