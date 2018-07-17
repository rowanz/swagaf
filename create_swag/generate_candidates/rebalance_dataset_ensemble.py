"""
The big idea will be to add in the worst scoring one. But we want to use a MULTILAYER PERCEPTRON.
Also not using word features for now

"""

import pickle as pkl
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import pandas as pd
import spacy
import torch
from allennlp.data import Instance
from allennlp.data import Token
from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from create_swag.lm.config import NUM_FOLDS
from create_swag.generate_candidates.classifiers import Ensemble, LMFeatsModel

######### PARAMETERS


NUM_DISTRACTORS = 9
TRAIN_PERC = 0.8
BATCH_SIZE = 1024

vocab = Vocabulary.from_files('../lm/vocabulary')
pos_vocab = Vocabulary(counter={'tokens': {name: i + 9000 for i, name in enumerate(
    [vocab.get_token_from_index(x) for x in range(100)] + [pos for pos in spacy.parts_of_speech.NAMES.values() if
                                                           len(pos) > 0]
)}})
vocab._token_to_index['pos'] = pos_vocab._token_to_index['tokens']
vocab._index_to_token['pos'] = pos_vocab._index_to_token['tokens']

parser = ArgumentParser(description='which fold to use')
parser.add_argument('-fold', dest='fold', help='Which fold to use. If you say -1 we will use ALL OF THEM!', type=int,
                    default=0)
fold = parser.parse_args().fold
assert fold in set(range(NUM_FOLDS)) or fold == -1
print("~~~~~~~~~USING SPLIT#{}~~~~~~~~~~~~~".format(fold))
if fold == -1:
    assignments = []
    assignments = np.load('assignments-pretrained.npy')
    # for i in range(5):
    #     assignments.append(np.load('assignments-fold-{}-19.npy'.format(i)))
    # assignments = np.concatenate(assignments)
else:
    assignments = np.load('assignments-pretrained-fold{}.npy'.format(fold))

#########################################


# TODO can we do this in parallel?
class AssignmentsDataLoader(Dataset):
    # TODO: we might need to load the dataset again on every iteration because memory is a big problem.
    def __init__(self, instances, inds, train=True, recompute_assignments=False):
        self.instances = instances
        self.inds = inds
        self.train = train
        self.recompute_assignments = recompute_assignments

        self.dataloader = DataLoader(dataset=self, batch_size=128 if not recompute_assignments else 16,
                                     shuffle=self.train, num_workers=0,
                                     collate_fn=self.collate, drop_last=self.train)

    def collate(self, items_l):
        # Assume all of these have the same length
        index_l, second_sentences_l, pos_tags_l, feats_l, context_len_l = zip(*items_l)

        feats = Variable(torch.FloatTensor(np.stack(feats_l)))
        inds = np.array(index_l)

        instances = []
        for second_sentences, pos_tags, context_len in zip(second_sentences_l, pos_tags_l, context_len_l):
            for second_sent, pos_tag in zip(second_sentences, pos_tags):
                instance_d = {
                    'words': TextField([Token(token) for token in ['@@bos@@'] + second_sent + ['@@eos@@']],
                                       {'tokens': SingleIdTokenIndexer(namespace='tokens', lowercase_tokens=True)}),
                    'postags': TextField([Token(token) for token in ['@@bos@@'] + pos_tag + ['@@eos@@']],
                                         {'pos': SingleIdTokenIndexer(namespace='pos', lowercase_tokens=False)}),
                }
                instance_d['context_indicator'] = SequenceLabelField([1] * (context_len + 1) +
                                                                     [0] * (len(second_sent) - context_len + 1),
                                                                     instance_d['words'])
                instances.append(Instance(instance_d))
        batch = Batch(instances)
        batch.index_instances(vocab)
        tensor_dict = batch.as_tensor_dict(for_training=self.train)

        # instances_mask = torch.LongTensor(np.stack([np.array([len(sub_g) > 0 for sub_g in g], dtype=np.int64)
        #                                             for g in selected_gens]))
        return {
            'lm_feats': feats,
            'inds': inds,
            'ending_word_ids': tensor_dict['words']['tokens'].view(inds.shape[0], -1,
                                                                   tensor_dict['words']['tokens'].size(1)),
            'postags_word_ids': tensor_dict['postags']['pos'].view(inds.shape[0], -1,
                                                                   tensor_dict['postags']['pos'].size(1)),
            'ctx_indicator': tensor_dict['context_indicator'].view(inds.shape[0], -1,
                                                                   tensor_dict['context_indicator'].size(1)),
        }

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        """
        :param index: index into the list of examples. ps: they are of the form

        sent1: List[str] of tokens for the first sentence
        startphrase: List[str] of tokens for the first part of the 2nd sentence
        generations: List[List[str]] of tokenized responses. The first one is GT.
        postags: List[List[str]] of POSTags and some lexicalization for startphrase+generations.

        They're all of the same size (1024)
        :return: index
                 second_sentences List[List[str]] full s2's
                 pos_tags List[List[str]] full PosTags of S2's
                 feats [#ex, dim] np array of features
                 context_len length of context size in second_sentences and pos_tags
        """
        this_ex = self.instances[index]

        second_sentences = [this_ex['startphrase'] + gen for gen in this_ex['generations']]
        context_len = len(this_ex['startphrase'])

        feats_vals = this_ex['scores'].values
        if np.isinf(feats_vals).any():
            feats_vals[np.isinf(feats_vals)] = 1e17

        feats = np.column_stack((
            np.log(feats_vals),
            np.array([len(gen) for gen in this_ex['generations']], dtype=np.float32),
            np.ones(feats_vals.shape[0], dtype=np.float32) * context_len,
            np.ones(feats_vals.shape[0], dtype=np.float32) * len(this_ex['sent1']),
        ))
        return index, second_sentences, this_ex['postags'], feats, context_len

    @classmethod
    def splits(cls, assignments):
        """ if assignments is none we initialize by looking at topN"""

        s_idx = 0
        train_instances = []
        val_instances = []
        test_instances = []

        train_indices = []
        test_indices = []
        print("loading the data!", flush=True)

        def _load_from_examples(example_list, offset):
            idx = np.random.permutation(len(example_list))
            train_idx = np.sort(idx[:int(TRAIN_PERC * idx.shape[0])])
            val_idx = np.sort(idx[int(TRAIN_PERC * idx.shape[0]):])

            train_indices.append(offset + train_idx)
            test_indices.append(offset + val_idx)

            for i in tqdm(train_idx):
                item_copy = example_list[i]
                item_copy['generations'] = [example_list[i]['generations'][j] for j in assignments[i + offset]]
                item_copy['postags'] = [example_list[i]['postags'][j] for j in assignments[i + offset]]
                item_copy['scores'] = example_list[i]['scores'].iloc[assignments[i + offset]]
                train_instances.append(item_copy)

            for i in tqdm(val_idx):
                item_copy = deepcopy(example_list[i])
                item_copy['generations'] = [example_list[i]['generations'][j] for j in assignments[i + offset]]
                item_copy['postags'] = [example_list[i]['postags'][j] for j in assignments[i + offset]]
                item_copy['scores'] = example_list[i]['scores'].iloc[assignments[i + offset]]
                val_instances.append(item_copy)
                test_instances.append(example_list[i])
            return len(ex_this_fold)

        folds2use = range(5) if fold == -1 else [fold]
        for fold_no in folds2use:
            print("loading data from fold {}".format(fold_no), flush=True)
            with open('examples{}-of-5.pkl'.format(fold_no), 'rb') as f:
                ex_this_fold = pkl.load(f)
            s_idx += _load_from_examples(ex_this_fold, s_idx)

        train_indices = np.concatenate(train_indices, 0)
        test_indices = np.concatenate(test_indices, 0)

        return cls(train_instances, train_indices, train=True), cls(val_instances, test_indices, train=False), cls(
            test_instances, test_indices, train=False, recompute_assignments=True)


def _iter():
    train, val, test = AssignmentsDataLoader.splits(assignments)
    model = Ensemble(vocab)
    model.cuda()

    val_results = model.fit(train.dataloader, val.dataloader, num_epoch=10)
    # Now get predictions for the best thing
    best_scoring_model_name = pd.Series(val_results).argmax()
    print("We will rebalance with {}".format(best_scoring_model_name))

    test_results, all_predictions = model.validate(test.dataloader)

    n2chs = []
    for val_ind, pred in zip(test.inds, all_predictions[best_scoring_model_name]):
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
            pass
        else:
            # change a random index
            ind_loc = np.random.choice(easy_inds, replace=False, size=num2change)
            adv_loc = np.random.choice(adversarial_examples, replace=False, size=num2change)
            assignments[val_ind, ind_loc] = adv_loc
            # Change the first index over.
            # ind_loc = easy_inds[0]
            # assignments[val_ind, ind_loc] = adversarial_examples[0]
    val_results['n2chs'] = np.mean(n2chs)
    return pd.Series(val_results)

all_results = []
for i in range(50):
    all_results.append(_iter())
    if fold == -1:
        pd.DataFrame(all_results).to_csv('ensemble-accs.csv', index=False)
        np.save('assignments-{}.npy'.format(i), assignments)
    else:
        pd.DataFrame(all_results).to_csv('ensemble-accs-fold-{}.csv'.format(fold), index=False)
        np.save('assignments-fold-{}-{}.npy'.format(fold, i), assignments)

#
# # To extract some things (maybe this is useful? idk)
# from nltk.tokenize.moses import MosesDetokenizer
# def _extract():
#     detokenizer = MosesDetokenizer()
#     with open('examples0-of-5.pkl', 'rb') as f:
#         ex_this_fold = pkl.load(f)
#     assignments = np.load('assignments-4.npy')
#
#     selected_examples = []
#     for ind, (item, assign_i) in enumerate(zip(tqdm(ex_this_fold), assignments)):
#         context = pd.Series([detokenizer.detokenize(item['sent1'], return_str=True)] * len(assign_i))
#         completions = pd.Series(
#             [detokenizer.detokenize(item['startphrase'] + item['generations'][i], return_str=True) for i in
#              assign_i.tolist()])
#         dataset = pd.Series([item['dataset']] * len(assign_i))
#         ids = pd.Series([item['id']] * len(assign_i))
#         duration = pd.Series([item['duration']] * len(assign_i))
#         inds = pd.Series([ind] * len(assign_i))
#
#         df_this_ex = pd.DataFrame(
#             data={'inds': inds, 'selections': assign_i, 'context': context, 'completions': completions,
#                   'is_gold': (assign_i == 0),
#                   'choice': np.arange(NUM_DISTRACTORS + 1),
#                   'dataset': dataset, 'ids': ids, 'duration': duration},
#             columns=['inds', 'context', 'completions', 'selections', 'is_gold', 'choice', 'dataset', 'ids', 'duration'])
#
#         df_with_extra_feats = pd.concat(
#             (df_this_ex, item['scores'].iloc[assign_i].reset_index(drop=True)), axis=1)
#         selected_examples.append(df_with_extra_feats)
#     return pd.concat(selected_examples, 0).reset_index(drop=True)
#
#
# _extract().to_csv('dataset.csv', sep='\t', index=False)
