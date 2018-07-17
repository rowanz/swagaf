import os

import pandas as pd
import torch
from allennlp.data import Instance
from allennlp.data import Token
from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from torch import optim

from create_swag.lm.simple_bilm import SimpleBiLM
from raw_data.events import _postprocess
from pytorch_misc import clip_grad_norm, print_para, time_batch
from create_swag.lm.config import PRETRAIN_TXT
assert os.path.exists('../vocabulary')

vocab = Vocabulary.from_files('../vocabulary')
indexer = ELMoTokenCharactersIndexer()


def batcher(inp_list):
    """ batches, asumming everything is padded and tokenized."""
    instances = [Instance({'story': TextField([Token(x) for x in ['@@bos@@'] + subl + ['@@eos@@']], token_indexers={
        'tokens': SingleIdTokenIndexer(namespace='tokens', lowercase_tokens=True), 'char_encoding': indexer}),
                           }) for subl in inp_list]
    batch = Batch(instances)
    batch.index_instances(vocab)
    result_dict = batch.as_tensor_dict()['story']
    result_dict['story'] = inp_list
    return result_dict


def data_runner(start_point=0, minlength=4):
    print("starting at {}".format(start_point))
    with open(PRETRAIN_TXT, 'r') as f:
        f.seek(start_point)
        f.readline()  # Clear the partial line
        for i, line in enumerate(f):
            yield _postprocess(line)


def _sample_a_good_pair(gen, seq_length, min_length=3):
    cur_status = []
    eos_idxs = [i for i, x in enumerate(cur_status) if x in ('.', '!', '?')]
    while len(eos_idxs) < 2:
        cur_status.extend([x for x in next(gen).split(' ') if x is not '\n'])
        eos_idxs = [i for i, x in enumerate(cur_status) if x in ('.', '!', '?')]
    if eos_idxs[1] >= seq_length:
        return _sample_a_good_pair(gen, seq_length, min_length=min_length)
    elif (eos_idxs[0] < min_length) or (eos_idxs[1] - eos_idxs[0]) < min_length:  # Too short
        return _sample_a_good_pair(gen, seq_length, min_length=min_length)

    return cur_status[:eos_idxs[1] + 1]


def looped_data_runner(batch_size=128, seq_length=50):
    offset = 0
    TOTAL_BYTES_TRAIN = 4343022454
    generators = [data_runner(start_point=TOTAL_BYTES_TRAIN * i // batch_size + offset, minlength=0) for i in
                  range(batch_size)]
    while True:
        for g_i, gen in enumerate(generators):
            yield _sample_a_good_pair(gen, seq_length=seq_length, min_length=5)


def bucketed_data_runner(batch_size=64, seq_length=50):
    length2batch = [[] for i in range(seq_length + 1)]
    # Get diverse samples
    for batch in looped_data_runner(batch_size=128, seq_length=seq_length):
        length2batch[len(batch)].append(batch)
        if len(length2batch[len(batch)]) >= batch_size:
            # print("Yielding now of size {}".format(len(batch)))
            yield batcher(length2batch[len(batch)])
            length2batch[len(batch)] = []


# Dataloader
model = SimpleBiLM(vocab=vocab, recurrent_dropout_probability=0.2, embedding_dropout_probability=0.2)
model.cuda()

tr = []
model.train()
for epoch_num in range(2):
    if epoch_num == 0:
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], weight_decay=1e-6, lr=1e-3)
    else:
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], weight_decay=1e-6, lr=1e-4)

    print(print_para(model))
    for b, (time_per_batch, batch) in enumerate(time_batch(bucketed_data_runner())):
        batch['tokens'] = batch['tokens'].cuda(async=True)

        model_forward = model(batch['tokens'])
        losses = {key: model_forward[key] for key in ['forward_loss', 'reverse_loss']}
        tr.append(pd.Series({k: v.data[0] for k, v in losses.items()}))

        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()

        if b % 100 == 0 and b > 0:
            df_cat = pd.concat(tr[-100:], axis=1).mean(1)
            print("b{:8d} {:.3f}s/batch, fwd loss {:.3f} rev loss {:.3f} ".format(b, time_per_batch,
                                                                                  df_cat['forward_loss'],
                                                                                  df_cat['reverse_loss']), flush=True)
        clip_grad_norm(
            [(n, p) for n, p in model.named_parameters() if p.grad is not None],
            max_norm=1.0, verbose=b % 1000 == 1, clip=True)
        optimizer.step()
        if b % 10000 == 0 and b > 0:
            torch.save({'state_dict': model.state_dict()}, 'e{}-tbooks-pretrained-ckpt-{}.tar'.format(epoch_num, b))
