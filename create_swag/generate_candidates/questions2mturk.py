import random
import pickle as pkl
import numpy as np
from tqdm import tqdm
from nltk.tokenize.moses import MosesDetokenizer
import pandas as pd
import re
detokenizer = MosesDetokenizer()

NUM_DISTRACTORS = 5

def _detokenize(sent):
    s0 = detokenizer.detokenize(sent, return_str=True)
    s1 = re.sub(r'\b(ca|do|wo)\sn\'t', r"\1n't", s0, flags=re.IGNORECASE)
    return s1


assignments = np.load('assignments-22.npy')
start_ind = 0
df = []

for fold_id in range(5):
    with open('../../generate_candidates/examples{}-of-5.pkl'.format(fold_id), 'rb') as f:
        ex_this_fold = pkl.load(f)
    assignments_this_fold = assignments[start_ind:start_ind+len(ex_this_fold)]
    start_ind += len(ex_this_fold)

    for i, (this_example, assignments_i) in enumerate(zip(tqdm(ex_this_fold), assignments_this_fold)):
        selected_gens = [this_example['generations'][i] for i in assignments_i.tolist()]

        # Find sent1 from the given sentences
        sent1 = _detokenize(this_example['sent1'])
        if sent1[0].islower():
            sent1 = sent1[0].upper() + sent1[1:]
        sent2 = _detokenize(this_example['startphrase'])
        # perm = np.random.permutation(NUM_DISTRACTORS+1)
        perm = np.arange(10)

        series_dict = {
            'item_ind': i,
            'fold_id': 0,
            'item_id': this_example['dataset'] + this_example['id'],
            'startphrase': '{} {}'.format(sent1,sent2),
            'sent1': sent1,
            'sent2': sent2,
            'gold': int(np.where(perm == 0)[0][0]),
        }
        for i, perm in enumerate(perm.tolist()):
            series_dict['completion-{:d}'.format(i)] = _detokenize(selected_gens[perm])
        df.append(pd.Series(series_dict))

random.seed(123456)
random.shuffle(df)
df = pd.DataFrame(df)


batch_size=1
batch_df = []
for j in range(df.shape[0] // batch_size):
    batch_df.append(pd.Series({'{}-{}'.format(i, name):val for i, (_, item) in enumerate(df[j*batch_size:(j+1)*batch_size].iterrows()) for name, val in item.items()}))
batch_df = pd.DataFrame(batch_df)
batch_df.to_csv('batch_df_FULL.csv', index=False)
