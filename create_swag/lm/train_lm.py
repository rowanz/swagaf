import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from create_swag.lm.config import NUM_FOLDS
from create_swag.lm.load_data import load_lm_data, RawPassages
from create_swag.lm.simple_bilm import SimpleBiLM
from pytorch_misc import clip_grad_norm, optimistic_restore, print_para, time_batch

if not os.path.exists('vocabulary') or not all(
        [os.path.exists('lm-{}-of-{}.pkl'.format(i, NUM_FOLDS)) for i in range(NUM_FOLDS)]):
    print("MAKING THE VOCABULARY / DATA AGAIN", flush=True)
    _, vocab = load_lm_data(None)

# ARGUMENTS
parser = ArgumentParser(description='which fold to use')
parser.add_argument('-fold', dest='fold', help='Which fold to use', type=int, default=0)
fold = parser.parse_args().fold
assert fold in set(range(NUM_FOLDS))
if not os.path.exists('checkpoints-{}'.format(fold)):
    os.mkdir('checkpoints-{}'.format(fold))
print("~~~~~~~~~USING SPLIT#{}~~~~~~~~~~~~~".format(fold))

train, val = RawPassages.splits(fold=fold)
model = SimpleBiLM(
    vocab=train.vocab,
    recurrent_dropout_probability=0.2,
    embedding_dropout_probability=0.2,
)

model.cuda()
optimistic_restore(model, torch.load('e1-tbooks-pretrained-ckpt-370000.tar')['state_dict'])
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], weight_decay=1e-6, lr=1e-3)
# scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1,
#                               verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
print(print_para(model))

for epoch_num in range(15):
    tr = []
    model.train()
    for b, (time_per_batch, batch) in enumerate(time_batch(train.dataloader)):
        # batch['char_encoding'] = batch['char_encoding'].cuda(async=True)
        batch['story'] = batch['story'].cuda(async=True)

        model_forward = model(batch['story'])
        losses = {key: model_forward[key] for key in ['forward_loss', 'reverse_loss']}
        tr.append(pd.Series({k: v.data[0] for k, v in losses.items()}))
        optimizer.zero_grad()
        loss = sum(losses.values())
        loss.backward()

        if b % 100 == 0 and b > 0:
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train.dataloader), time_per_batch,
                len(train.dataloader) * time_per_batch / 60))
            print(pd.concat(tr[-100:], axis=1).mean(1))
            print('-----------', flush=True)
        clip_grad_norm(
            [(n, p) for n, p in model.named_parameters() if p.grad is not None],
            max_norm=1, verbose=b % 1000 == 1, clip=True)
        optimizer.step()

    # Get the validation perplexity
    perplexity = []
    model.eval()
    for batch in tqdm(val.dataloader):
        # batch['char_encoding'] = batch['char_encoding'].cuda(async=True)
        batch['story'] = batch['story'].cuda(async=True)

        model_forward = model(batch['story'])
        losses = {key: model_forward[key] for key in ['forward_loss', 'reverse_loss']}
        perplexity.append(pd.Series({k: v.data[0] for k, v in losses.items()}))

    df_cat = pd.DataFrame(perplexity).mean(0)
    print("Epoch {}, fwd loss {:.3f} perplexity {:.3f} bwd loss {:.3f} perplexity {:.3f}".format(
        epoch_num, df_cat['forward_loss'], np.exp(df_cat['forward_loss']), df_cat['reverse_loss'],
        np.exp(df_cat['reverse_loss'])), flush=True)

    scheduler.step(df_cat['forward_loss'] + df_cat['reverse_loss'])
    torch.save({'state_dict': model.state_dict()}, 'checkpoints-{}/ckpt-{}.tar'.format(fold, epoch_num))
