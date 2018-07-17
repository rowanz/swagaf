"""
A wrapper around ai2s elmo LM to allow for an lm objective...
"""

from typing import Optional, Tuple
from typing import Union, List, Dict

import numpy as np
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.nn.util import sequence_cross_entropy_with_logits
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence


def _de_duplicate_generations(generations):
    """
    Given a list of list of strings, filter out the ones that are duplicates. and return an idx corresponding
    to the good ones
    :param generations:
    :return:
    """
    dup_set = set()
    unique_idx = []
    for i, gen_i in enumerate(generations):
        gen_i_str = ' '.join(gen_i)
        if gen_i_str not in dup_set:
            unique_idx.append(i)
            dup_set.add(gen_i_str)
    return [generations[i] for i in unique_idx], np.array(unique_idx)


class StackedLstm(torch.nn.Module):
    """
    A stacked LSTM.

    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    hidden_size : int, required
        The dimension of the outputs of the LSTM.
    num_layers : int, required
        The number of stacked LSTMs to use.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .
    use_input_projection_bias : bool, optional (default = True)
        Whether or not to use a bias on the input projection layer. This is mainly here
        for backwards compatibility reasons and will be removed (and set to False)
        in future releases.

    Returns
    -------
    output_accumulator : PackedSequence
        The outputs of the interleaved LSTMs per timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 recurrent_dropout_probability: float = 0.0,
                 use_highway: bool = True,
                 use_input_projection_bias: bool = True,
                 go_forward: bool = True) -> None:
        super(StackedLstm, self).__init__()

        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        layers = []
        lstm_input_size = input_size
        for layer_index in range(num_layers):
            layer = AugmentedLstm(lstm_input_size, hidden_size, go_forward,
                                  recurrent_dropout_probability=recurrent_dropout_probability,
                                  use_highway=use_highway,
                                  use_input_projection_bias=use_input_projection_bias)
            lstm_input_size = hidden_size
            self.add_module('layer_{}'.format(layer_index), layer)
            layers.append(layer)
        self.lstm_layers = layers

    def forward(self,  # pylint: disable=arguments-differ
                inputs: PackedSequence,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (1, batch_size, output_dimension).

        Returns
        -------
        output_sequence : PackedSequence
            The encoded sequence of shape (batch_size, sequence_length, hidden_size)
        final_states: torch.Tensor
            The per-layer final (state, memory) states of the LSTM, each with shape
            (num_layers, batch_size, hidden_size).
        """
        if not initial_state:
            hidden_states = [None] * len(self.lstm_layers)
        elif initial_state[0].size()[0] != len(self.lstm_layers):
            raise ConfigurationError("Initial states were passed to forward() but the number of "
                                     "initial states does not match the number of layers.")
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0),
                                     initial_state[1].split(1, 0)))

        output_sequence = inputs
        final_states = []
        for i, state in enumerate(hidden_states):
            layer = getattr(self, 'layer_{}'.format(i))
            # The state is duplicated to mirror the Pytorch API for LSTMs.
            output_sequence, final_state = layer(output_sequence, state)
            final_states.append(final_state)

        final_state_tuple = tuple(torch.cat(state_list, 0) for state_list in zip(*final_states))
        return output_sequence, final_state_tuple


class SimpleBiLM(torch.nn.Module):
    def __init__(self,
                 vocab: Vocabulary,
                 recurrent_dropout_probability: float = 0.0,
                 embedding_dropout_probability: float = 0.0,
                 input_size=512,
                 hidden_size=512) -> None:
        """
        :param options_file: for initializing elmo BiLM
        :param weight_file: for initializing elmo BiLM
        :param requires_grad: Whether or not to finetune the LSTM layers
        :param recurrent_dropout_probability: recurrent dropout to add to LSTM layers
        """
        super(SimpleBiLM, self).__init__()

        self.forward_lm = PytorchSeq2SeqWrapper(StackedLstm(
            input_size=input_size, hidden_size=hidden_size, num_layers=2, go_forward=True,
            recurrent_dropout_probability=recurrent_dropout_probability,
            use_input_projection_bias=False, use_highway=True), stateful=True)
        self.reverse_lm = PytorchSeq2SeqWrapper(StackedLstm(
            input_size=input_size, hidden_size=hidden_size, num_layers=2, go_forward=False,
            recurrent_dropout_probability=recurrent_dropout_probability,
            use_input_projection_bias=False, use_highway=True), stateful=True)

        # This will also be the encoder
        self.decoder = torch.nn.Linear(512, vocab.get_vocab_size(namespace='tokens'))

        self.vocab = vocab
        self.register_buffer('eos_tokens', torch.LongTensor([vocab.get_token_index(tok) for tok in
                                                             ['.', '!', '?', '@@UNKNOWN@@', '@@PADDING@@', '@@bos@@',
                                                              '@@eos@@']]))
        self.register_buffer('invalid_tokens', torch.LongTensor([vocab.get_token_index(tok) for tok in
                                                                 ['@@UNKNOWN@@', '@@PADDING@@', '@@bos@@', '@@eos@@',
                                                                  '@@NEWLINE@@']]))
        self.embedding_dropout_probability = embedding_dropout_probability

    def embed_words(self, words):
        assert words.dim() == 2
        if not self.training:
            return F.embedding(words, self.decoder.weight)
        # Embedding dropout
        vocab_size = self.decoder.weight.size(0)
        mask = Variable(
            self.decoder.weight.data.new(vocab_size, 1).bernoulli_(1 - self.embedding_dropout_probability).expand_as(
                self.decoder.weight) / (1 - self.embedding_dropout_probability))

        padding_idx = 0
        embeds = self.decoder._backend.Embedding.apply(words, mask * self.decoder.weight, padding_idx, None,
                                                       2, False, False)
        return embeds

    def timestep_to_ids(self, timestep_tokenized: List[str]):
        """ Just a single timestep (so dont add BOS or EOS"""
        return Variable(torch.LongTensor([self.vocab.get_token_index(x) for x in timestep_tokenized])[:, None],
                        volatile=not self.training).cuda(async=True)

    def batch_to_ids(self, stories_tokenized: List[List[str]]):
        """
        Simple wrapper around _elmo_batch_to_ids
        :param batch: A list of tokenized sentences.
        :return: A tensor of padded character ids.
        """
        batch = Batch([Instance(
            {'story': TextField([Token('@@bos@@')] + [Token(x) for x in story] + [Token('@@eos@@')],
                                token_indexers={
                                    'tokens': SingleIdTokenIndexer(namespace='tokens', lowercase_tokens=True)})})
            for story in stories_tokenized])
        batch.index_instances(self.vocab)
        words = {k: v['tokens'] for k, v in batch.as_tensor_dict(for_training=self.training).items()}['story'].cuda(
            async=True)
        return words

    def conditional_generation(self, context, gt_completion, batch_size=128, max_gen_length=25,
                               same_length_as_gt=False):
        """
        Generate conditoned on the context. While we're at it we'll score the GT going forwards
        :param context: List of tokens to condition on. We'll add the BOS marker to it
        :param gt_completion: The GT completion
        :param batch_size: Number of sentences to generate
        :param max_gen_length: Max length for genertaed sentences (irrelvant if same_length_as_gt=True)
        :param same_length_as_gt: set to True if you want all the sents to have the same length as the gt_completion
        :return:
        """
        # Forward condition on context, then repeat to be the right batch size:
        #  (layer_index, batch_size, fwd hidden dim)
        forward_logprobs = self(self.batch_to_ids([context]), use_forward=True,
                                use_reverse=False, compute_logprobs=True)['forward_logprobs']
        self.forward_lm._states = tuple(x.repeat(1, batch_size, 1).contiguous() for x in self.forward_lm._states)
        # Each item will be (token, score)
        generations = [[(context[-1], 0.0)] for i in range(batch_size)]
        mask = Variable(forward_logprobs.data.new(batch_size).long().fill_(1))

        gt_completion_padded = [self.vocab.get_token_index(gt_token) for gt_token in
                                [x.lower() for x in gt_completion] + ['@@PADDING@@'] * (
                                        max_gen_length - len(gt_completion))]

        for index, gt_token_ind in enumerate(gt_completion_padded):
            embeds = self.embed_words(self.timestep_to_ids([gen[-1][0] for gen in generations]))
            next_dists = F.softmax(self.decoder(self.forward_lm(embeds, mask[:, None]))[:, 0], 1).data

            # Perform hacky stuff on the distribution (disallowing BOS, EOS, that sorta thing
            sampling_probs = next_dists.clone()
            sampling_probs[:, self.invalid_tokens] = 0.0

            # fix first row!!!
            sampling_probs[0].zero_()
            sampling_probs[0, gt_token_ind] = 1

            if same_length_as_gt:
                if index == (len(gt_completion) - 1):
                    sampling_probs.zero_()
                    sampling_probs[:, gt_token_ind] = 1
                else:
                    sampling_probs[:, self.eos_tokens] = 0.0

            sampling_probs = sampling_probs / sampling_probs.sum(1, keepdim=True)

            next_preds = torch.multinomial(sampling_probs, 1).squeeze(1)
            next_scores = np.log(next_dists[
                                     torch.arange(0, next_dists.size(0),
                                                  out=mask.data.new(next_dists.size(0))),
                                     next_preds,
                                 ].cpu().numpy())
            for i, (gen_list, pred_id, score_i, mask_i) in enumerate(
                    zip(generations, next_preds.cpu().numpy(), next_scores, mask.data.cpu().numpy())):
                if mask_i:
                    gen_list.append((self.vocab.get_token_from_index(pred_id), score_i))
            is_eos = (next_preds[:, None] == self.eos_tokens[None]).max(1)[0]
            mask[is_eos] = 0
            if mask.sum().data[0] == 0:
                break
        generation_scores = np.zeros((len(generations), max([len(g) - 1 for g in generations])), dtype=np.float32)
        for i, gen in enumerate(generations):
            for j, (_, v) in enumerate(gen[1:]):
                generation_scores[i, j] = v

        generation_toks, idx = _de_duplicate_generations([[tok for (tok, score) in gen[1:]] for gen in generations])
        return generation_toks, generation_scores[idx], forward_logprobs.data.cpu().numpy()

    def _chunked_logsoftmaxes(self, activation, word_targets, chunk_size=256):
        """
        do the softmax in chunks so memory doesnt explode
        :param activation: [batch, T, dim]
        :param targets: [batch, T] indices
        :param chunk_size: you might need to tune this based on GPU specs
        :return:
        """
        all_logprobs = []
        num_chunks = (activation.size(0) - 1) // chunk_size + 1
        for activation_chunk, target_chunk in zip(torch.chunk(activation, num_chunks, dim=0),
                                                  torch.chunk(word_targets, num_chunks, dim=0)):
            assert activation_chunk.size()[:2] == target_chunk.size()[:2]
            targets_flat = target_chunk.view(-1)
            time_indexer = torch.arange(0, targets_flat.size(0),
                                        out=target_chunk.data.new(targets_flat.size(0))) % target_chunk.size(1)
            batch_indexer = torch.arange(0, targets_flat.size(0),
                                         out=target_chunk.data.new(targets_flat.size(0))) / target_chunk.size(1)
            all_logprobs.append(F.log_softmax(self.decoder(activation_chunk), 2)[
                                    batch_indexer, time_indexer, targets_flat].view(*target_chunk.size()))
        return torch.cat(all_logprobs, 0)

    def forward(self, words: torch.Tensor, use_forward=True, use_reverse=True, compute_logprobs=False) -> Dict[
        str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        use this for training the LM
        :param words: [batch_size, N] words. assuming you're starting with BOS and ending with EOS here
        :return:
        """
        encoded_inputs = self.embed_words(words)
        mask = (words != 0).long()[:, 2:]
        word_targets = words[:, 1:-1].contiguous()

        result_dict = {
            'mask': mask,
            'word_targets': word_targets,
        }
        # TODO: try to reduce duplicate code here
        if use_forward:
            self.forward_lm.reset_states()
            forward_activation = self.forward_lm(encoded_inputs[:, :-2], mask)

            if compute_logprobs:
                # being memory efficient here is critical if the input tensors are large
                result_dict['forward_logprobs'] = self._chunked_logsoftmaxes(forward_activation,
                                                                             word_targets) * mask.float()
            else:

                result_dict['forward_logits'] = self.decoder(forward_activation)
                result_dict['forward_loss'] = sequence_cross_entropy_with_logits(result_dict['forward_logits'],
                                                                                 word_targets,
                                                                                 mask)
        if use_reverse:
            self.reverse_lm.reset_states()
            reverse_activation = self.reverse_lm(encoded_inputs[:, 2:], mask)
            if compute_logprobs:
                result_dict['reverse_logprobs'] = self._chunked_logsoftmaxes(reverse_activation,
                                                                             word_targets) * mask.float()
            else:
                result_dict['reverse_logits'] = self.decoder(reverse_activation)
                result_dict['reverse_loss'] = sequence_cross_entropy_with_logits(result_dict['reverse_logits'],
                                                                                 word_targets,
                                                                                 mask)
        return result_dict
