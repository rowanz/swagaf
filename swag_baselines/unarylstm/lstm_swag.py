from typing import Dict, List, TextIO, Optional

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, ElmoTokenEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.training.metrics import CategoricalAccuracy

@Model.register("lstm_swag")
class LstmSwag(Model):
    """
    This model performs semantic role labeling using BIO tags using Propbank semantic roles.
    Specifically, it is an implmentation of `Deep Semantic Role Labeling - What works
    and what's next <https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`_ .

    This implementation is effectively a series of stacked interleaved LSTMs with highway
    connections, applied to embedded sequences of words concatenated with a binary indicator
    containing whether or not a word is the verbal predicate to generate predictions for in
    the sentence. Additionally, during inference, Viterbi decoding is applied to constrain
    the predictions to contain valid BIO sequences.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    binary_feature_dim : int, required.
        The dimensionality of the embedding of the binary verb predicate features.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    label_smoothing : ``float``, optional (default = 0.0)
        Whether or not to use label smoothing on the labels when computing cross entropy loss.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 # binary_feature_dim: int,
                 embedding_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(LstmSwag, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder

        # For the span based evaluation, we don't want to consider labels
        # for verb, because the verb index is provided to the model.
        self.encoder = encoder
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self.output_prediction = Linear(self.encoder.get_output_dim(), 1, bias=False)

        check_dimensions_match(text_field_embedder.get_output_dim(),
                               encoder.get_input_dim(),
                               "text embedding dim", "eq encoder input dim")
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,  # type: ignore
                hypothesis0: Dict[str, torch.LongTensor],
                hypothesis1: Dict[str, torch.LongTensor],
                hypothesis2: Dict[str, torch.LongTensor],
                hypothesis3: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        logits = []
        for tokens in [hypothesis0, hypothesis1, hypothesis2, hypothesis3]:
            if isinstance(self.text_field_embedder, ElmoTokenEmbedder):
                self.text_field_embedder._elmo._elmo_lstm._elmo_lstm.reset_states()

            embedded_text_input = self.embedding_dropout(self.text_field_embedder(tokens))
            mask = get_text_field_mask(tokens)

            batch_size, sequence_length, _ = embedded_text_input.size()

            encoded_text = self.encoder(embedded_text_input, mask)

            logits.append(self.output_prediction(encoded_text.max(1)[0]))

        logits = torch.cat(logits, -1)
        class_probabilities = F.softmax(logits, dim=-1).view([batch_size, 4])
        output_dict = {"label_logits": logits, "label_probs": class_probabilities}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            self._accuracy(logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'accuracy': self._accuracy.get_metric(reset),
                }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'LstmSwag':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   encoder=encoder,
                   initializer=initializer,
                   regularizer=regularizer)
