"""
The big idea will be to add in the worst scoring one. But we want to use a MULTILAYER PERCEPTRON.
Also not using word features for now

"""

import torch
from allennlp.common import Params
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.token_embedders.embedding import Embedding
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import pandas as pd
from model.pytorch_misc import clip_grad_norm, optimistic_restore, print_para, time_batch
from torch import optim
import numpy as np


#################### Model types

def reshape(f):
    def wrapper(self, *args, **kwargs):
        sizes = [x.size() for x in args] + [x.size() for x in kwargs.values()]
        batch_size, num_ex = sizes[0][:2]

        res = f(self, *[x.view((-1,) + x.size()[2:]) for x in args],
                **{k: v.view((-1,), + v.size()[2:]) for k, v in kwargs})
        if isinstance(res, tuple):
            return tuple([x.view((batch_size, num_ex,) + x.size()[1:]) for x in res])
        return res.view((batch_size, num_ex,) + res.size()[1:])

    return wrapper


class LMFeatsModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=1024):
        """
        Averaged embeddings of ending -> label
        :param embed_dim: dimension to use
        """
        super(LMFeatsModel, self).__init__()
        self.mapping = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.SELU(),
            nn.AlphaDropout(p=0.2),
        )
        self.prediction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SELU(),
            nn.AlphaDropout(p=0.2),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    @reshape
    def forward(self, feats):
        """
        :param words: [batch, dim] indices
        :return: [batch] scores of real-ness.
        """
        inter_feats = self.mapping(feats)
        preds = self.prediction(inter_feats).squeeze(1)
        return preds, inter_feats

    def fit(self, data, val_data=None, num_epoch=10):
        self.train()
        optimizer = optim.Adam(self.parameters(), weight_decay=1e-4, lr=1e-3)
        best_val = 0.0
        for epoch_num in range(num_epoch):
            tr = []
            for b, (time_per_batch, batch) in enumerate(time_batch(data, reset_every=100)):
                results = self(batch['lm_feats'].cuda(async=True))[0]
                loss = F.cross_entropy(results, Variable(results.data.new(results.size(0)).long().fill_(0)))
                summ_dict = {'loss': loss.data[0], 'acc': (results.max(1)[1] == 0).float().mean().data[0]}

                tr.append(pd.Series(summ_dict))
                optimizer.zero_grad()
                loss.backward()

                clip_grad_norm(
                    [(n, p) for n, p in self.named_parameters() if p.grad is not None],
                    max_norm=1.0, verbose=False, clip=True)
                optimizer.step()

            mean_stats = pd.concat(tr, axis=1).mean(1)
            if val_data is not None:
                val_acc, val_results = self.validate(val_data)
                print("e{:2d}: train loss {:.3f} train acc {:.3f} val acc {:.3f}".format(epoch_num, mean_stats['loss'],
                                                                          mean_stats['acc'], val_acc), flush=True)
                if val_acc < best_val or epoch_num == (num_epoch - 1):
                    return {'mlp': val_acc, 'fasttext': 0, 'cnn': 0, 'lstm_pos': 0, 'ensemble': 0}
                best_val = val_acc

    def validate(self, data):
        self.eval()
        all_predictions = []
        for b, (time_per_batch, batch) in enumerate(time_batch(data, reset_every=100)):
            results = self(batch['lm_feats'].cuda(async=True))[0]
            all_predictions.append(results.data.cpu().numpy())
        all_predictions = np.concatenate(all_predictions, 0)
        acc = (all_predictions.argmax(1) == 0).mean()
        return acc, {'ensemble': all_predictions}


class BoWModel(nn.Module):
    def __init__(self, vocab, use_mean=True, embed_dim=100):
        """
        Averaged embeddings of ending -> label
        :param embed_dim: dimension to use
        """
        super(BoWModel, self).__init__()
        assert embed_dim == 100
        self.embeds = Embedding.from_params(
            vocab,
            Params({'vocab_namespace': 'tokens',
                    'embedding_dim': embed_dim,
                    'trainable': True,
                    'padding_index': 0,
                    'pretrained_file':
                        'https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz'
                    }))

        self.embed_dim = embed_dim
        self.use_mean = use_mean
        self.embedding_to_label = nn.Linear(self.embed_dim, 1, bias=False)

    @reshape
    def forward(self, word_ids):
        """
        :param word_ids: [batch, length] ids
        :return: [batch] scores of real-ness.
        """
        embeds = self.embeds(word_ids)
        mask = (word_ids.data != 0).long()
        seq_lengths = mask.sum(-1, keepdim=True).float()
        seq_lengths[seq_lengths < 1] = 1.0

        inter_feats = embeds.sum(1) / Variable(seq_lengths) if self.use_mean else embeds.max(1)[0]
        preds = self.embedding_to_label(inter_feats).squeeze(1)
        return preds, inter_feats


class CNNModel(nn.Module):
    def __init__(self, vocab, embed_dim=100, window_sizes=(2, 3, 4, 5), num_filters=128):
        super(CNNModel, self).__init__()

        self.embeds = Embedding.from_params(
            vocab,
            Params({'vocab_namespace': 'tokens',
                    'embedding_dim': embed_dim,
                    'trainable': True,
                    'padding_index': 0,
                    'pretrained_file':
                        'https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz'
                    }))
        self.binary_feature_embedding = Embedding(2, embed_dim)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim * 2, num_filters, kernel_size=window_size, padding=window_size - 1) for window_size in
            window_sizes
        ])
        self.fc = nn.Linear(num_filters * len(window_sizes), 1, bias=False)

    @reshape
    def forward(self, word_ids, indicator_ids):
        """
        :param word_ids: [batch, length] ids
        :param indicator_ids: [batch, length] ids
        """
        embeds = torch.cat((self.embeds(word_ids), self.binary_feature_embedding(indicator_ids)), 2)
        # mask = (word_ids != 0).long()

        embeds_t = embeds.transpose(1, 2)  # [B, D, L]

        conv_reps = []
        for conv in self.convs:
            conv_reps.append(F.relu(conv(embeds_t)).max(2)[0])  # Now it's [B, D]

        inter_feats = torch.cat(conv_reps, 1)
        preds = self.fc(inter_feats).squeeze(1)
        return preds, inter_feats


class BLSTMModel(nn.Module):
    def __init__(self, vocab, use_postags_only=True, embed_dim=100, hidden_size=200, recurrent_dropout_probability=0.3,
                 use_highway=False,
                 maxpool=True):
        super(BLSTMModel, self).__init__()

        self.embeds = Embedding.from_params(
            vocab,
            Params({'vocab_namespace': 'pos' if use_postags_only else 'tokens',
                    'embedding_dim': embed_dim,
                    'trainable': True,
                    'padding_index': 0,
                    'pretrained_file': None if use_postags_only else 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz',
                    }))
        self.binary_feature_embedding = Embedding(2, embed_dim)

        self.fwd_lstm = PytorchSeq2SeqWrapper(AugmentedLstm(
            input_size=embed_dim * 2, hidden_size=hidden_size, go_forward=True,
            recurrent_dropout_probability=recurrent_dropout_probability,
            use_input_projection_bias=False, use_highway=use_highway), stateful=False)

        self.bwd_lstm = PytorchSeq2SeqWrapper(AugmentedLstm(
            input_size=embed_dim * 2, hidden_size=hidden_size, go_forward=False,
            recurrent_dropout_probability=recurrent_dropout_probability,
            use_input_projection_bias=False, use_highway=use_highway), stateful=False)

        self.maxpool = maxpool
        self.fc = nn.Linear(hidden_size * 2, 1, bias=False)

    @reshape
    def forward(self, word_ids, indicator_ids):
        """
        :param word_ids: [batch, length] ids
        :param indicator_ids: [batch, length] ids
        """
        embeds = torch.cat((self.embeds(word_ids), self.binary_feature_embedding(indicator_ids)), 2)
        mask = (word_ids != 0).long()

        fwd_activation = self.fwd_lstm(embeds, mask)  # [B, L, D]
        bwd_activation = self.bwd_lstm(embeds, mask)
        if self.maxpool:
            reps = torch.cat((fwd_activation.max(1)[0], bwd_activation.max(1)[0]), 1)  # [B*N, 2D]
        else:
            # Forward and last.
            reps = torch.cat((
                fwd_activation[torch.arange(0, mask.size(0), out=mask.data.new(mask.size(0))), mask.sum(1) - 1],
                bwd_activation[:, 0]
            ), 1)
        return self.fc(reps).squeeze(1), reps


class Ensemble(nn.Module):
    def __init__(self, vocab):
        super(Ensemble, self).__init__()

        self.fasttext_model = BoWModel(vocab, use_mean=True, embed_dim=100)
        self.mlp_model = LMFeatsModel(input_dim=8, hidden_dim=1024)
        self.lstm_pos_model = BLSTMModel(vocab, use_postags_only=True, maxpool=True)
        # self.lstm_lex_model = BLSTMModel(vocab, use_postags_only=False, maxpool=True)
        self.cnn_model = CNNModel(vocab)

        self.mlp = nn.Sequential(
            nn.Linear(100 + 1024 + 400 + 4 * 128, 2048, bias=True),
            # nn.SELU(),
            # nn.AlphaDropout(p=0.2),
            # nn.Linear(2048, 2048, bias=True),
            nn.SELU(),
            nn.AlphaDropout(p=0.2),
            nn.Linear(2048, 1, bias=False),
        )

    def forward(self, lm_feats, ending_word_ids, postags_word_ids, ctx_indicator, inds):
        """
        :param lm_feats: [batch_size, #options, dim]
        :param ending_word_ids: [batch_size, #options, L] word ids
        :param postags_word_ids: [batch_size, #options, L] word ids
        :param ctx_indicator: [batch_size, #options, L] indicator
        :param inds: [batch_size] indices (not needed)
        :return:
        """
        results = {}
        results['mlp'], mlp_feats = self.mlp_model(lm_feats)
        results['fasttext'], fasttext_feats = self.fasttext_model(ending_word_ids)
        results['cnn'], cnn_feats = self.cnn_model(ending_word_ids, ctx_indicator)
        results['lstm_pos'], lstm_feats = self.lstm_pos_model(postags_word_ids, ctx_indicator)
        # results['lstm_lex'], _ = self.lstm_lex_model(ending_word_ids, ctx_indicator)
        results['ensemble'] = self.mlp(
            torch.cat((mlp_feats, fasttext_feats, cnn_feats, lstm_feats), 2)).squeeze(2)
        return results

    def predict(self, lm_feats, ending_word_ids, postags_word_ids, ctx_indicator, inds):
        """ Predict a distribution of probabilities
        :return: Dict from model type -> prob dist
        """
        results = self.forward(lm_feats, ending_word_ids, postags_word_ids, ctx_indicator, inds)
        results = {k: F.softmax(v, 1).data.cpu().numpy() for k, v in results.items()}
        return results

    def validate(self, val_dataloader):
        """
        :param val_dataloader: Dataloader
        :return: Accuracies: dict from model -> accuracy
                 All predictions: Dict from model -> [batch, #ex] distribution.
        """
        # Compute the validation performance
        self.eval()
        all_predictions = {'mlp': [], 'fasttext': [], 'cnn': [], 'lstm_pos': [], #'lstm_lex': [],
                           'ensemble': []}
        for b, (time_per_batch, batch) in enumerate(time_batch(val_dataloader, reset_every=100)):
            batch = {k: v.cuda(async=True) if hasattr(v, 'cuda') else v for k, v in batch.items()}
            if b % 100 == 0 and b > 0:
                print("\nb{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                    b, len(val_dataloader), time_per_batch,
                    len(val_dataloader) * time_per_batch / 60), flush=True)
            for k, v in self.predict(**batch).items():
                all_predictions[k].append(v)
        all_predictions = {k: np.concatenate(v, 0) for k, v in all_predictions.items()}
        accuracies = {k: np.mean(v.argmax(1) == 0) for k, v in all_predictions.items()}
        return accuracies, all_predictions

    def fit(self, train_dataloader, val_dataloader, num_epoch=5):
        """
        :param train_dataloader: Dataloader
        :param num_epoch number of epochs to use
        """
        print_every = 100
        optimizer = optim.Adam([p for p in self.parameters() if p.requires_grad], weight_decay=1e-6, lr=1e-3)
        best_val = 0.0
        for epoch_num in range(num_epoch):
            tr = []
            self.train()
            for b, (time_per_batch, batch) in enumerate(time_batch(train_dataloader, reset_every=print_every)):
                batch = {k: v.cuda(async=True) if hasattr(v, 'cuda') else v for k, v in batch.items()}
                results = self(**batch)
                losses = {'{}-loss'.format(k): F.cross_entropy(
                    v, Variable(v.data.new(v.size(0)).long().fill_(0))) for k, v in results.items()}
                if any([np.isnan(x.data.cpu().numpy()) for x in losses.values()]):
                    import ipdb
                    ipdb.set_trace()
                loss = sum(losses.values())
                summ_dict = {k: v.data[0] for k, v in losses.items()}
                summ_dict.update(
                    {'{}-acc'.format(k): (v.max(1)[1] == 0).float().mean().data[0] for k, v in results.items()})

                tr.append(pd.Series(summ_dict))
                optimizer.zero_grad()
                loss.backward()

                if b % print_every == 0 and b > 0:
                    print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                        epoch_num, b, len(train_dataloader), time_per_batch,
                        len(train_dataloader) * time_per_batch / 60))
                    print(pd.concat(tr[-print_every:], axis=1).mean(1))
                    print('-----------', flush=True)

                # clip_grad_norm([(n, p) for n, p in self.named_parameters() if
                #                 p.grad is not None and n.startswith('lstm_lex_model')], max_norm=1.0,
                #                verbose=b % 100 == 1, clip=True)
                clip_grad_norm([(n, p) for n, p in self.named_parameters() if
                                p.grad is not None and not n.startswith('lstm_lex_model')], max_norm=1.0,
                               verbose=b % 100 == 1, clip=True)
                optimizer.step()
            val_results, _ = self.validate(val_dataloader)
            val_acc = val_results['ensemble']
            if val_acc < best_val or epoch_num == (num_epoch - 1):
                print("Stopping on epoch={} with\n{}".format(epoch_num, pd.Series(val_results)), flush=True)
                return val_results
            else:
                print("Continuing on epoch={} with\n{}".format(epoch_num, pd.Series(val_results)), flush=True)
            best_val = val_acc
