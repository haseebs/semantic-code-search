import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from astlib.tensor_utils.analyze import node_incidence_matrix
from rtp_transformer.utils import generate_tree_relative_movements
from typing import Tuple

from .encoder_base import EncoderBase
from .relative_multihead_attention import RelativeMultiheadSelfAttention
from .positional_encoder import PositionalEncoder, LevelPositionalEmbedding
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TreeTransformerEncoderLayer(TransformerEncoderLayer):
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
    def forward(
        self,
        src,
        mask=None,
        src_key_padding_mask=None,
        relative_distances=None,
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


def init_params(module):
    """
    Initialize the weights.
    This overrides the default initializations depending on the specified arguments.
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


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
        use_level_positional_embeddings: bool,
        ancestor_prediction: bool
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
            encoder_layers, nlayers
        )
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=0)
        self.ninp = ninp

        self.pos_encoder = None
        if use_sinusoidal_positional_embeddings:
            self.pos_encoder = PositionalEncoder(ninp)

        self.level_pos_encoder = None
        if use_level_positional_embeddings:
            self.level_pos_encoder = LevelPositionalEmbedding(embedding_dim=ninp)

        self.dropout = nn.Dropout(p=dropout)
        self.clamping_distance = clamping_distance

        self.scale = math.sqrt(self.ninp)

        self.ancestor_prediction_head = LCAPredictionHead(
            embed_dim=ninp,
            activation_fn="relu"
        ) if ancestor_prediction else None

        self.apply(init_params)

    def forward(self, src, seq_tokens_mask, seq_len, src_descendants):

        seq_tokens_mask = seq_tokens_mask == 0  # BoolTensor [B, N]
        src = src.T  # [B, N] -> [N, B]

        # embed
        src_embed = self.encoder(src)  # [N, B, D]

        # scale by sqrt(d)
        src_embed *= self.scale

        # maybe embed positions
        if self.pos_encoder is not None:
            src_embed = self.pos_encoder(src_embed)

        # compute node incidences here         # TODO check this if vocab changes
        incidences = node_incidence_matrix(
            src_descendants, pad_idx=0, pad_mask=seq_tokens_mask,
        )
        relative_distances = generate_tree_relative_movements(
            node_incidences=incidences,
            pad_idx=0,
            max_relative_distance=self.clamping_distance,
            pad_mask=seq_tokens_mask,
        )

        if self.level_pos_encoder is not None:
            src_embed = self.level_pos_encoder(src_embed, incidences)

        # dropout
        src_embed = self.dropout(src_embed)

        output = self.transformer_encoder(
            src_embed, src_key_padding_mask=seq_tokens_mask, relative_distances=relative_distances
        )  # [N, B, D]

        # compute code embeddings
        seq_token_embeddings_sum = output.sum(dim=0)  # [B, D]
        seq_lengths = seq_len.to(seq_token_embeddings_sum)[:, None]  # [B, 1]
        return seq_token_embeddings_sum / seq_lengths, output

    def predict_ancestors(self, features, node_pairs):
        # compute ancestor predictions
        ancestor_logits =  self.ancestor_prediction_head(
                features=features,
                pair_indices=node_pairs
            )
        return ancestor_logits


class Linear(nn.Module):
    def __init__(self, embed_dim, output_dim, activation_fn="relu"):
        super().__init__()
        self.dense = nn.Linear(embed_dim, output_dim)
        self.activation_fn = nn.GELU() if activation_fn == "gelu" else nn.ReLU()

    def forward(self, features, **kwargs):
        # Features = B x T x H
        x = self.dense(features)
        x = self.activation_fn(x)

        return x


class LCAPredictionHead(nn.Module):
    """Head for lca prediction based on the tokens embeddings."""

    def __init__(self, embed_dim, activation_fn):
        super().__init__()

        self.linear = Linear(
            embed_dim=embed_dim * 2,
            output_dim=embed_dim,
            activation_fn=activation_fn
        )

    @staticmethod
    def gather_feature_pairs(features: torch.Tensor, pair_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # features: B x T x H
        # pair_indices: B x V x 2
        features_1 = torch.gather(
            features,
            1,
            pair_indices[:, :, 0, None].expand(-1, -1, features.size(-1))
        )  # B x V x H
        features_2 = torch.gather(
            features,
            1,
            pair_indices[:, :, 1, None].expand(-1, -1, features.size(-1))
        )  # B x V x H

        return features_1, features_2

    def forward(self, features: torch.Tensor, pair_indices: torch.Tensor, **kwargs):
        """
        # features: N x B x H  as returned from the encoder
        # pair_indices: B x V x 2  (V=number of ancestor predictions)

        :return:
        """
        features = features.transpose(0, 1)  # [B, N, H]

        encoded1, encoded2 = self.gather_feature_pairs(features, pair_indices)  # both B x V x H
        encoded = torch.cat((encoded1, encoded2), dim=-1)  # B x V x 2H

        # project
        x = self.linear(encoded)  # B x V x H
        x = torch.bmm(x, features.transpose(1, 2))  # B, V, T
        return x

