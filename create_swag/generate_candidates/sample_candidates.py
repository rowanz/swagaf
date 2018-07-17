import pickle as pkl
from argparse import ArgumentParser
from copy import deepcopy
from time import time

import numpy as np
import pandas as pd
import torch
from allennlp.commands.predict import Predictor
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from tqdm import tqdm

from create_swag.lm.config import NUM_FOLDS
from create_swag.lm.simple_bilm import SimpleBiLM
from pytorch_misc import optimistic_restore
from spacy.tokens.doc import Doc
from allennlp.common.util import get_spacy_model
import spacy

BATCH_SIZE = 1024

# ARGUMENTS
parser = ArgumentParser(description='which fold to use')
parser.add_argument('-fold', dest='fold', help='Which fold to use', type=int, default=0)
fold = parser.parse_args().fold
assert fold in set(range(NUM_FOLDS))
print("~~~~~~~~~USING SPLIT#{}~~~~~~~~~~~~~".format(fold))

# SETUP
spacy_model = get_spacy_model("en_core_web_sm", pos_tags=True, parse=False, ner=False)
spacy_model.tokenizer = lambda x: Doc(spacy_model.vocab, x)

archive = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz')
constituency_predictor = Predictor.from_archive(archive, 'constituency-parser')
# This is hella hacky! but it's tokenized already
constituency_predictor._tokenizer.spacy.tokenizer = lambda x: Doc(constituency_predictor._tokenizer.spacy.vocab, x)

vocab = Vocabulary.from_files('../lm/vocabulary')

pos_vocab = Vocabulary(counter={'tokens': {name: i + 9000 for i, name in enumerate(
    [vocab.get_token_from_index(x) for x in range(100)] + [pos for pos in spacy.parts_of_speech.NAMES.values() if
                                                           len(pos) > 0]
)}})

model = SimpleBiLM(vocab=vocab, recurrent_dropout_probability=0.2, embedding_dropout_probability=0.2)
optimistic_restore(model,
                   torch.load('../lm/best-{}.tar'.format(fold))['state_dict'])  # <- NEED TO DO THIS ON A FOLD LEVEL
# include if not necessary
model.register_buffer('invalid_tokens', torch.LongTensor([vocab.get_token_index(tok) for tok in
                                                          ['@@UNKNOWN@@', '@@PADDING@@', '@@bos@@', '@@eos@@',
                                                           '@@NEWLINE@@']]))
model.cuda()
model.eval()

with open('../lm/lm-{}-of-{}.pkl'.format(fold, NUM_FOLDS), 'rb') as f:
    stories_tokenized = pkl.load(f)


########

# We want to recurse until we find verb phrases
def find_VP(tree):
    """
    Recurse on the tree until we find verb phrases
    :param tree: constituency parser result
    :return:
    """

    # Recursion is annoying because we need to check whether each is a list or not
    def _recurse_on_children():
        assert 'children' in tree
        result = []
        for child in tree['children']:
            res = find_VP(child)
            if isinstance(res, tuple):
                result.append(res)
            else:
                result.extend(res)
        return result

    if 'VP' in tree['attributes']:
        # # Now we'll get greedy and see if we can find something better
        # if 'children' in tree and len(tree['children']) > 1:
        #     recurse_result = _recurse_on_children()
        #     if all([x[1] in ('VP', 'NP', 'CC') for x in recurse_result]):
        #         return recurse_result
        return [(tree['word'], 'VP')]
    # base cases
    if 'NP' in tree['attributes']:
        return [(tree['word'], 'NP')]
    # No children
    if not 'children' in tree:
        return [(tree['word'], tree['attributes'][0])]

    # If a node only has 1 child then we'll have to stick with that
    if len(tree['children']) == 1:
        return _recurse_on_children()
    # try recursing on everything
    return _recurse_on_children()


def split_on_final_vp(sentence):
    """ Splits a sentence on the final verb phrase"""
    try:
        res = constituency_predictor.predict_json({'sentence': sentence})
    except:
        return None, None
    res_chunked = find_VP(res['hierplane_tree']['root'])
    is_vp = [i for i, (word, pos) in enumerate(res_chunked) if pos == 'VP']
    if len(is_vp) == 0:
        return None, None
    vp_ind = max(is_vp)
    not_vp = [token for x in res_chunked[:vp_ind] for token in x[0].split(' ')]
    is_vp = [token for x in res_chunked[vp_ind:] for token in x[0].split(' ')]
    return not_vp, is_vp


good_examples = []
for (instance, s1_toks, s2_toks, item) in tqdm(stories_tokenized):

    eos_bounds = [i + 1 for i, x in enumerate(s1_toks) if x in ('.', '?', '!')]
    if len(eos_bounds) == 0:
        s1_toks.append('.')  # Just in case there's no EOS indicator.

    context_len = len(s1_toks)
    if context_len < 6 or context_len > 100:
        print("skipping on {} (too short or long)".format(' '.join(s1_toks + s2_toks)))
        continue

    # Something I should have done: make sure that there aren't multiple periods, etc. in s2 or in the middle
    eos_bounds_s2 = [i + 1 for i, x in enumerate(s2_toks) if x in ('.', '?', '!')]
    if len(eos_bounds_s2) > 1 or max(eos_bounds_s2) != len(s2_toks):
        continue
    elif len(eos_bounds_s2) == 0:
        s2_toks.append('.')


    # Now split on the VP
    startphrase, endphrase = split_on_final_vp(s2_toks)
    if startphrase is None or len(startphrase) == 0 or len(endphrase) < 5 or len(endphrase) > 25:
        print("skipping on {}->{},{}".format(' '.join(s1_toks + s2_toks), startphrase, endphrase), flush=True)
        continue

    # if endphrase contains unk then it's hopeless
    if any(vocab.get_token_index(tok.lower()) == vocab.get_token_index(vocab._oov_token) for tok in endphrase):
        print("skipping on {} (unk!)".format(' '.join(s1_toks + s2_toks)))
        continue

    context = s1_toks + startphrase
    tic = time()
    gens0, fwd_scores, ctx_scores = model.conditional_generation(context, gt_completion=endphrase,
                                                                 batch_size=2 * BATCH_SIZE,
                                                                 max_gen_length=25)
    if len(gens0) < BATCH_SIZE:
        print("Couldnt generate enough candidates so skipping")
        continue
    gens0 = gens0[:BATCH_SIZE]
    fwd_scores = fwd_scores[:BATCH_SIZE]

    # Now get the backward scores.
    full_sents = [context + gen for gen in gens0]  # NOTE: #1 is GT
    result_dict = model(model.batch_to_ids(full_sents), use_forward=False, use_reverse=True, compute_logprobs=True)
    ending_lengths = (fwd_scores < 0).sum(1)
    ending_lengths_float = ending_lengths.astype(np.float32)
    rev_scores = result_dict['reverse_logprobs'].data.cpu().numpy()

    forward_logperp_ending = -fwd_scores.sum(1) / ending_lengths_float
    reverse_logperp_ending = -rev_scores[:, context_len:].sum(1) / ending_lengths_float
    forward_logperp_begin = -ctx_scores.mean()
    reverse_logperp_begin = -rev_scores[:, :context_len].mean(1)
    eos_logperp = -fwd_scores[np.arange(fwd_scores.shape[0]), ending_lengths - 1]
    print("Time elapsed {:.3f}".format(time() - tic), flush=True)

    scores = np.exp(np.column_stack((
        forward_logperp_ending,
        reverse_logperp_ending,
        reverse_logperp_begin,
        eos_logperp,
        np.ones(forward_logperp_ending.shape[0], dtype=np.float32) * forward_logperp_begin,
    )))

    # PRINTOUT
    low2high = scores[:, 2].argsort()
    print("\n\n Dataset={} ctx: {} (perp={:.3f})\n~~~\n".format(item['dataset'], ' '.join(context),
                                                                np.exp(forward_logperp_begin)), flush=True)
    for i, ind in enumerate(low2high.tolist()):
        gen_i = ' '.join(gens0[ind])
        if (ind == 0) or (i < 128):
            print("{:3d}/{:4d}) ({}, end|ctx:{:5.1f} end:{:5.1f} ctx|end:{:5.1f} EOS|(ctx, end):{:5.1f}) {}".format(
                i, len(gens0), 'GOLD' if ind == 0 else '    ', *scores[ind][:-1], gen_i), flush=True)
    gt_score = low2high.argsort()[0]

    item_full = deepcopy(item)
    item_full['sent1'] = s1_toks
    item_full['startphrase'] = startphrase
    item_full['context'] = context
    item_full['generations'] = gens0
    item_full['postags'] = [  # parse real fast
        [x.orth_.lower() if pos_vocab.get_token_index(x.orth_.lower()) != 1 else x.pos_ for x in y]
        for y in spacy_model.pipe([startphrase + gen for gen in gens0], batch_size=BATCH_SIZE)]
    item_full['scores'] = pd.DataFrame(data=scores, index=np.arange(scores.shape[0]),
                                       columns=['end-from-ctx', 'end', 'ctx-from-end', 'eos-from-ctxend', 'ctx'])
    good_examples.append(item_full)

with open('examples{}-of-{}.pkl'.format(fold, NUM_FOLDS), 'wb') as f:
    pkl.dump(good_examples, f)
