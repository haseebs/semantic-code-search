import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from astlib.tensor_utils.analyze import node_incidence_matrix
from rtp_transformer.utils import generate_tree_relative_movements

from .encoder_base import EncoderBase
from .relative_multihead_attention import RelativeMultiheadSelfAttention
from .positional_encoder import PositionalEncoder
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TreeTransformerEncoderLayer(TransformerEncoderLayer):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        num_rpr_embeddings=None,
    ):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn = RelativeMultiheadSelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            num_rpr_embeddings=num_rpr_embeddings,
            key_rpr=True,
            value_rpr=False,
            pad_idx=1,
            self_attention=True,
        )

    def forward(
        self, src, src_mask=None, src_key_padding_mask=None, relative_distances=None
    ):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2, _ = self.self_attn(
            query=src,
            key=src,
            value=src,
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask,
            relative_distances=relative_distances,
            tree_attn_mask=None,
            incidences=None,
            nonterminal_mask=None,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TreeTransformerEncoder(TransformerEncoder):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    _constants_ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None, clamping_distance=3):
        super().__init__(encoder_layer, num_layers, norm)
        self.clamping_distance = clamping_distance

    def forward(
        self,
        src,
        mask=None,
        src_key_padding_mask=None,
        src_descendants=None,
        src_nonterminal_mask=None,
    ):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src

        # TODO check this if vocab changes
        incidences = node_incidence_matrix(
            src_descendants, pad_idx=0, pad_mask=src_key_padding_mask,
        )
        relative_distances = generate_tree_relative_movements(
            node_incidences=incidences,
            pad_idx=0,
            max_relative_distance=self.clamping_distance,
            pad_mask=src_key_padding_mask,
        )

        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                relative_distances=relative_distances,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TreeAttentionEncoder(EncoderBase):
    def __init__(
        self,
        ntoken,
        ninp,
        nhead,
        nhid,
        nlayers,
        dropout,
        vocab_count_threshold,
        use_bpe,
        vocab_pct_bpe,
        clamping_distance,
        use_sinusoidal_positional_embeddings: bool,
    ):
        super().__init__(ntoken, vocab_count_threshold, use_bpe, vocab_pct_bpe)
        self.src_mask = None
        encoder_layers = TreeTransformerEncoderLayer(
            ninp,
            nhead,
            nhid,
            dropout,
            num_rpr_embeddings=2 * ((clamping_distance + 1) ** 2),
        )
        self.transformer_encoder = TreeTransformerEncoder(
            encoder_layers, nlayers, clamping_distance=clamping_distance
        )
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=0)
        self.ninp = ninp

        self.pos_encoder = (
            PositionalEncoder(ninp, dropout)
            if use_sinusoidal_positional_embeddings
            else None
        )

        self.scale = math.sqrt(self.ninp)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, seq_tokens_mask, seq_len, src_descendants):

        seq_tokens_mask = seq_tokens_mask == 0  # BoolTensor [B, N]
        src = src.T  # [B, N] -> [N, B]

        # embed
        src = self.encoder(src)  # [N, B, D]

        # scale by sqrt(d)
        src *= self.scale

        # maybe embed sinusoidal positions
        if self.pos_encoder is not None:
            src = self.pos_encoder(src)

        output = self.transformer_encoder(
            src, src_key_padding_mask=seq_tokens_mask, src_descendants=src_descendants
        )  # [N,B,D]
        seq_token_embeddings_sum = output.sum(dim=0)  # [N,D]
        seq_lengths = seq_len.to(dtype=torch.float32).unsqueeze(dim=-1)  # [N,1]
        return seq_token_embeddings_sum / seq_lengths
