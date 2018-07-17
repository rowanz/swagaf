"""
The big idea will be to add in the worst scoring one. But we want to use a MULTILAYER PERCEPTRON.
Also not using word features for now

"""
import matplotlib as mpl

mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

from allennlp.data import Vocabulary
from torch.nn import functional as F
from torch import nn
from torch.autograd import Variable
import pickle as pkl
import numpy as np
from torch import optim
import torch
from tqdm import tqdm, trange
from pytorch_misc import clip_grad_norm, time_batch
import pandas as pd
import os
######### PARAMETERS

NUM_DISTRACTORS = 9
TRAIN_PERC = 0.8

vocab = Vocabulary.from_files('../lm/vocabulary')

all_data = []
if os.path.exists('feats_cached.npy'):
    all_data = np.load('feats_cached.npy')
else:
    print("loading data. this will take hella time probably!", flush=True)
    for fold in trange(5):
        print("tryna load {}".format(fold, flush=True))
        with open('examples{}-of-5.pkl'.format(fold), 'rb') as f:
            examples = pkl.load(f)
            for this_ex in examples:
                feats_vals = this_ex['scores'].values
                if np.isinf(feats_vals).any():
                    feats_vals[np.isinf(feats_vals)] = 1e17
                feats = np.column_stack((
                    np.log(feats_vals),
                    np.array([len(gen) for gen in this_ex['generations']], dtype=np.float32),
                    np.ones(feats_vals.shape[0], dtype=np.float32) * len(this_ex['startphrase']),
                    np.ones(feats_vals.shape[0], dtype=np.float32) * len(this_ex['sent1']),
                ))
                all_data.append(feats)
    all_data = np.stack(all_data)
    np.save('feats_cached.npy', all_data)

print("There are {} things".format(all_data.shape[0]), flush=True)
assignments = np.arange(NUM_DISTRACTORS + 1, dtype=np.uint16)[None].repeat(all_data.shape[0], axis=0)


class SimpleCudaLoader(object):
    """ silly cuda loader"""
    def __init__(self,
                 indices,
                 is_train=True,
                 recompute_assignments=False,
                 batch_size=512,
                 ):
        self.indices = indices
        self.is_train = is_train
        self.recompute_assignments = recompute_assignments
        if self.recompute_assignments:
            self.feats = all_data[self.indices]
        else:
            self.feats = all_data[np.arange(all_data.shape[0])[:, None], assignments][self.indices]

        self.batch_size = batch_size

    def __iter__(self):
        """
        Iterator for a cuda type application.
        :return:
        """
        # First cuda-ize everything
        if self.is_train:
            perm_vec = np.random.permutation(self.feats.shape[0])
            feats_to_use = self.feats[perm_vec]
            inds_to_use = self.indices[perm_vec]
        else:
            feats_to_use = self.feats
            inds_to_use = self.indices

        feats_cuda = torch.FloatTensor(feats_to_use).contiguous().cuda(async=True)

        for s_idx in range(len(self)):
            s_ind = s_idx * self.batch_size
            e_ind = min(s_ind + self.batch_size, self.feats.shape[0])
            if e_ind < self.batch_size and self.is_train:
                # Skip small batch on training
                return
            yield Variable(feats_cuda[s_ind:e_ind], volatile=not self.is_train), inds_to_use[s_ind:e_ind]

    @classmethod
    def randomsplits(cls):
        """
        Makes some random splits! But keeping in mind the (global) assignments info
        :return:
        """
        idx = np.random.permutation(all_data.shape[0])
        train_idx = idx[:int(TRAIN_PERC * idx.shape[0])]
        val_idx = np.sort(idx[int(TRAIN_PERC * idx.shape[0]):])
        return cls(train_idx, is_train=True), cls(val_idx, is_train=False), cls(val_idx, recompute_assignments=True, is_train=False),

    def __len__(self):
        if self.is_train:
            return self.feats.shape[0] // self.batch_size
        else:
            return (self.feats.shape[0] + self.batch_size - 1) // self.batch_size


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        # self.mapping = nn.Linear(train_data.feats.shape[2], 1, bias=False)

        self.mapping = nn.Sequential(
            nn.Linear(all_data.shape[-1], 2048, bias=True),
            nn.SELU(),
            nn.AlphaDropout(p=0.2),
            nn.Linear(2048, 2048, bias=True),
            nn.SELU(),
            nn.AlphaDropout(p=0.2),
            nn.Linear(2048, 1, bias=False),
        )

    def forward(self, feats):
        # Contribution from embeddings
        # (batch, #ex, length, dim) -> (batch, #ex, dim)
        return self.mapping(feats).squeeze(-1)

    def fit(self, data, val_data=None, n_epoch=10):
        self.train()
        optimizer = optim.Adam(self.parameters(), weight_decay=1e-4, lr=1e-3)
        best_val = 0.0
        for epoch_num in range(n_epoch):
            tr = []
            for b, (time_per_batch, batch) in enumerate(time_batch(data, reset_every=100)):
                feats, inds_to_use = batch
                results = model(feats)
                loss = F.cross_entropy(results, Variable(results.data.new(results.size(0)).long().fill_(0)))
                summ_dict = {'loss': loss.data[0], 'acc': (results.max(1)[1] == 0).float().mean().data[0]}

                tr.append(pd.Series(summ_dict))
                optimizer.zero_grad()
                loss.backward()

                clip_grad_norm(
                    [(n, p) for n, p in model.named_parameters() if p.grad is not None],
                    max_norm=1.0, verbose=False, clip=True)
                optimizer.step()

            mean_stats = pd.concat(tr, axis=1).mean(1)
            if val_data is not None:
                vp, val_acc = self.predict(val_data)
                print("e{:2d}: train loss {:.3f} train acc {:.3f} val acc {:.3f}".format(epoch_num, mean_stats['loss'],
                                                                          mean_stats['acc'], val_acc), flush=True)
                if val_acc < best_val or epoch_num == (n_epoch - 1):
                    return
                best_val = val_acc

    def predict(self, data):
        self.eval()
        all_predictions = []
        for b, (time_per_batch, batch) in enumerate(time_batch(data, reset_every=100)):
            feats, inds_to_use = batch
            all_predictions.append(model(feats).data.cpu().numpy())
        all_predictions = np.concatenate(all_predictions, 0)
        if data.recompute_assignments:
            masked_predictions = all_predictions[np.arange(data.feats.shape[0])[:, None], assignments[data.indices]]
        else:
            masked_predictions = all_predictions
        acc = (masked_predictions.argmax(1) == 0).mean()
        mr = (-masked_predictions).argsort(1).argsort(1)[:, 0].mean()
        # print("acc is {:.3f}, mean rank is {:.3f}".format(acc, mr))
        return all_predictions, acc



accs = []
for iter in trange(100):
    train, val, test = SimpleCudaLoader.randomsplits()
    model = MLPModel()
    model.cuda()
    model.fit(train, val)
    predictions, acc = model.predict(test)
    accs.append(acc)
    # Now do some remapping
    n2chs = []
    for pred, val_ind in zip(predictions, test.indices):
        high2low = (-pred).argsort()  # Things at the beginning of this list seem real
        idx2rank = high2low.argsort()

        cur_assign = assignments[val_ind]
        adversarial_examples = high2low[:idx2rank[0]]
        adversarial_examples = adversarial_examples[
            ~np.in1d(adversarial_examples, cur_assign)]  # not currently assigned

        easy_idxs = high2low[idx2rank[0] + 1:][::-1]
        easy_idxs = easy_idxs[np.in1d(easy_idxs, cur_assign)]

        # Make the easy indices map according to their position in the assignments
        easy_inds = np.argmax(easy_idxs[:, None] == cur_assign[None], 1)
        assert np.allclose(cur_assign[easy_inds], easy_idxs)

        num2change = min(2, adversarial_examples.shape[0], easy_idxs.shape[0])
        n2chs.append(num2change)
        # print("adversarial ex we can add {:4d} easy idxs {:4d} were changing {:4d}".format(
        #     adversarial_examples.shape[0], easy_idxs.shape[0], num2change))
        if num2change == 0:
            # print("Continuing, nothing we can change")
            pass
        else:
            # change a random index
            ind_loc = np.random.choice(easy_inds, replace=False, size=num2change)
            adv_loc = np.random.choice(adversarial_examples, replace=False, size=num2change)
            assignments[val_ind, ind_loc] = adv_loc
            # Change the first index over.
            # ind_loc = easy_inds[0]
            # assignments[val_ind, ind_loc] = adversarial_examples[0]
    print("{:.3f} val accuracy: {:.3f} n2chs".format(acc, np.mean(n2chs)), flush=True)
    assert np.all(assignments[:, 0] == 0)

# Plot the accuracy as time goes by
np.save('assignments-pretrained.npy', assignments)
start_idx = 0
for fold in trange(5):
    with open('examples{}-of-5.pkl'.format(fold), 'rb') as f:
        examples = pkl.load(f)
    assignments_this_fold = assignments[start_idx:start_idx+len(examples)]
    np.save('assignments-pretrained-fold{}.npy'.format(fold), assignments_this_fold)
    start_idx += len(examples)

plt.clf()
accuracy = pd.Series(np.array(accs))
df = pd.DataFrame(pd.concat([accuracy,
                             # accuracy.rolling(window=int(1/(1-TRAIN_PERC)), win_type='gaussian', min_periods=1, center=True).mean(std=2)
                             accuracy.rolling(window=2 * int(1 / (1 - TRAIN_PERC)), win_type=None, min_periods=1,
                                              center=True).mean()
                             ], 0),
                  columns=['accuracy'])
df['subject'] = 0
df['series'] = ['accuracy'] * accuracy.shape[0] + ['smoothed accuracy'] * accuracy.shape[0]
df.index.rename('iteration', inplace=True)
df.reset_index(inplace=True)
df.to_csv('pretrain-rebalance-mlp.csv')
sns.set(color_codes=True)
fig = sns.tsplot(time='iteration', value='accuracy', data=df, unit='subject', condition='series').get_figure()
fig.savefig('rebalancing-mlp-acc.pdf')
