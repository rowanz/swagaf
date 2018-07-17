# First make the vocabulary, etc.

import os
import pickle as pkl
import random

import simplejson as json
from allennlp.common.util import get_spacy_model
from allennlp.data import Instance
from allennlp.data import Token
from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from raw_data.events import DATA_PATH
from pytorch_misc import pairwise
from create_swag.lm.config import NUM_FOLDS

def load_lm_data(fold=None, mode='train'):
    """
    Turns the sequential data into instances.
    :param split:
    :return:
    """
    # Get or make vocab
    spacy_model = get_spacy_model("en_core_web_sm", pos_tags=False, parse=False, ner=False)
    if os.path.exists('vocabulary'):
        print("Loading cached vocab. caution if you're building the dataset again!!!!", flush=True)
        vocab = Vocabulary.from_files('vocabulary')

        with open(os.path.join(DATA_PATH, 'events-3.json'), 'r') as f:
            lm_data = json.load(f)
        lm_data = [data_item for s in ('train', 'val', 'test') for data_item in lm_data[s]]
    else:
        assert fold is None
        with open(os.path.join(DATA_PATH, 'events-3.json'), 'r') as f:
            lm_data = json.load(f)
        lm_data = [data_item for s in ('train', 'val', 'test') for data_item in lm_data[s]]
        # Manually doing this because I don't want to double count things
        vocab = Vocabulary.from_instances(
            [Instance({'story': TextField(
                [Token(x) for x in ['@@bos@@'] + [x.orth_ for x in spacy_model(sent)] + ['@@eos@@']], token_indexers={
                    'tokens': SingleIdTokenIndexer(namespace='tokens', lowercase_tokens=True)})}) for data_item in
             lm_data for sent in
             data_item['sentences']], min_count={'tokens': 3})

        vocab.get_index_to_token_vocabulary('tokens')
        vocab.save_to_files('vocabulary')
        print("VOCABULARY HAS {} ITEMS".format(vocab.get_vocab_size(namespace='tokens')))

    if all([os.path.exists('lm-{}-of-{}.pkl'.format(i, NUM_FOLDS)) for i in range(NUM_FOLDS)]):
        print("LOADING CACHED DATASET", flush=True)
        if mode == 'val':
            with open('lm-{}-of-{}.pkl'.format(fold, NUM_FOLDS), 'rb') as f:
                print("Loading split{} for {}".format(fold, mode))
                instances = pkl.load(f)
        else:
            instances = []
            for other_fold in range(NUM_FOLDS):
                if other_fold != fold:
                    with open('lm-{}-of-{}.pkl'.format(other_fold, NUM_FOLDS), 'rb') as f:
                        print("Loading split{} for {}".format(other_fold, mode))
                        instances += pkl.load(f)
        return instances, vocab

    print("MAKING THE DATASET", flush=True)
    assert fold is None
    for item in tqdm(lm_data):
        item['sentences_tokenized'] = [[st.orth_ for st in spacy_model(sent)] for sent in item['sentences']]

    def _to_instances(data):
        # flatten this
        instances = []
        for item in data:
            for s1, s2 in pairwise(item['sentences_tokenized']):
                instances.append((
                    Instance({'story': TextField([Token(x) for x in ['@@bos@@'] + s1 + s2 + ['@@eos@@']],
                                                 token_indexers={
                                                     'tokens': SingleIdTokenIndexer(namespace='tokens',
                                                                                    lowercase_tokens=True)})}),
                    s1,
                    s2,
                    item,
                ))
        return instances

    random.seed(123456)
    random.shuffle(lm_data)
    all_sets = []
    for fold_ in range(NUM_FOLDS):
        val_set = _to_instances(lm_data[len(lm_data) * fold_ // NUM_FOLDS:len(lm_data) * (fold_ + 1) // NUM_FOLDS])
        with open('lm-{}-of-{}.pkl'.format(fold_, NUM_FOLDS), 'wb') as f:
            pkl.dump(val_set, f)
        all_sets.extend(val_set)
    return all_sets, vocab


class RawPassages(Dataset):
    def __init__(self, fold, mode):
        self.mode = mode
        self.fold = fold
        self.instances, self.vocab = load_lm_data(fold=self.fold, mode=self.mode)
        self.dataloader = DataLoader(dataset=self, batch_size=32,
                                     shuffle=self.mode == 'train', num_workers=0,
                                     collate_fn=self.collate, drop_last=self.mode == 'train')
        self.indexer = ELMoTokenCharactersIndexer()

    def collate(self, instances_l):
        batch = Batch([x[0] for x in instances_l])
        batch.index_instances(self.vocab)

        batch_dict = {k: v['tokens'] for k, v in batch.as_tensor_dict().items()}

        batch_dict['story_tokens'] = [instance[0].fields['story'].tokens for instance in instances_l]
        batch_dict['story_full'] = [x[1] + x[2] for x in instances_l]
        batch_dict['items'] = [x[3] for x in instances_l]
        return batch_dict

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        """
        :param index:
        :return: * raw rocstories
                 * entities
                 * entity IDs + sentences
                 * Instance. to print use r3.fields['verb_phrase'].field_list[5].tokens
        """
        return self.instances[index]

    @classmethod
    def splits(cls, fold):
        return cls(fold, mode='train'), cls(fold, mode='val')


if __name__ == '__main__':
    instances, vocab = load_lm_data()
    # train, val = RawPassages.splits()
    # for item in train.dataloader:
    #     for story in item['story_tokens']:
    #         tok_text = [x.text.lower() for x in story]
    #         remapped_text = [vocab.get_token_from_index(vocab.get_token_index(x)) for x in tok_text]
    #         print('({}) {} -> {}'.format('D' if tok_text != remapped_text else ' ',
    #                                      ' '.join(tok_text), ' '.join(remapped_text)), flush=True)
