# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq.modules import MultiheadAttention
from torch import Tensor, nn
import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules.sinusoidal_positional_embedding import (
    SinusoidalPositionalEmbedding,
)

from typing import *
from rtp_transformer.utils import accumulate_nodes


def generate_relative_distances_matrix(
    length_q: int, length_k: int, max_relative_distance: int
):
    """
    Generates matrix of relative positions between inputs.
    adapted from tensor2tensor
    """
    if length_q == length_k:
        range_vec_q = range_vec_k = torch.arange(length_q)
    else:
        range_vec_k = torch.arange(length_k)
        range_vec_q = range_vec_k[-length_q:]

    distance_mat = range_vec_k[None, :] - range_vec_q[:, None]

    distance_mat_clipped = torch.clamp(
        distance_mat, min=-max_relative_distance, max=max_relative_distance
    )
    # Shift values to be >= 0. Each integer still uniquely identifies a relative
    # position difference.
    final_mat = distance_mat_clipped + max_relative_distance
    return final_mat


## key attention
def compute_key_attention_gather(
    embedding: Union[torch.Tensor, nn.Embedding],
    q: torch.Tensor,
    relative_distances: torch.Tensor,
) -> torch.Tensor:
    """
    Computes relative position representations, by multiplying the
     query with the rpr-embedding and gathering the desired scores
     afterwards (2.), instead of embedding every token and multiplying
     that with every key (1., as described in Shaw et al.):

    1. [S, B*H, HD] x [S, HD, S] -> [B, H, S, S]
    2. [B*H, S, HD] x [HD, M] -> [B, H, S, M]

    S = source length
    B = batch_size
    H = number of heads
    HD = head_dim
    M = length of key embedding (number of relative positions)

    :param embedding:           shape [M, HD]
    :param q:                   shape [B, H, S, HD]
    :param relative_distances:  shape [S, S]; or [B, S, S], if saved state then [1, S] or [B, 1, S]
    :return: a_k                shape [B*H, S, S]
    """
    (
        batch_size,
        num_heads,
        q_len,
        head_dim,
    ) = q.shape  # [B, H, S, HD]  - saved [B, H, 1, HD]

    # just some checks
    if isinstance(embedding, nn.Embedding):
        embedding = embedding.weight

    if relative_distances.ndimension() == 2:
        relative_distances = relative_distances[  # [S, S]        - saved [1, S]
            None, None, :, :
        ].expand(  # [1, 1, S, S]  - saved [1, 1, 1, S]
            (batch_size, num_heads, -1, -1)
        )  # [B, H, S, S]  - saved [B, H, 1, S]
    elif relative_distances.ndimension() == 3:
        relative_distances = relative_distances[  # [B, S, S]      - saved [B, 1, S]
            :, None, :, :
        ].expand(  # [B, 1, S, S]   - saved [B, 1, 1, S]
            (-1, num_heads, -1, -1)
        )  # [B, H, S, S]   - saved [B, H, 1, S]

    rd_bz, rd_nh, rd_rows, rd_cols, = relative_distances.shape
    assert batch_size == rd_bz
    assert rd_rows == q_len
    assert rd_rows == 1 or rd_rows == rd_cols

    # then the work
    q_pos = torch.matmul(
        q, embedding.t()  # [B, H, S, HD]  - saved [B, H, 1, HD]  # [HD, M]
    )  # [B, H, S, M]   - saved [B, H, 1, M]

    k_rpr = torch.gather(
        q_pos,  # [B, H, S, M]  - saved [B, H, 1, M]
        dim=3,
        index=relative_distances,  # [B, H, S, S]  - saved [B, H, 1, S]
    )
    k_rpr = k_rpr.view(-1, rd_rows, rd_cols)  # [B*H, S, S]  - saved [B*H, 1, S]

    return k_rpr


def compute_value_attention(
    embedding: nn.Embedding, attn_probs: torch.Tensor, relative_positions: torch.Tensor
) -> torch.Tensor:
    return compute_value_attention_shaw(embedding, attn_probs, relative_positions)


def compute_key_attention_shaw(
    embedding: nn.Embedding, q: torch.Tensor, relative_distances: torch.Tensor
) -> torch.Tensor:
    """
    Computes relative position representations as described by Shaw et al. (2018) (Equation 5).

    S = source length
    B = batch_size
    H = number of heads
    HD = head_dim

    :param q:                   shape [B, H, S, HD]  (saved [B, H, 1, HD])
    :param relative_distances:  shape [S, S]         (saved [1, S])
    :return: a_k                shape [B*H, S, S]    (saved [B, H, 1, S])
    """
    k_rpr_embed = embedding(
        relative_distances  # [S, S]        - saved [1, S]
    ).transpose(  # [S, S, HD]    - saved [1, S, HD]
        1, 2
    )  # [S, HD, S]    - saved [1, HD, S]
    k_rpr = torch.bmm(
        (
            q.view(
                -1, q.size(2), q.size(3)  # B*H  # S  # HD
            ).transpose(  # [B*H, S, HD]
                0, 1
            )
        ),  # [S, B*H, HD]  - saved [1, B*H, HD]
        k_rpr_embed,  # [S, HD, S]    - saved [1, HD, S]
    ).transpose(  # [S, B*H, S]   - saved [1, B*H, S]
        0, 1
    )  # [B*H, S, S]   - saved [B*H, 1, S]
    return k_rpr  # [B*H, S, S]  - saved [B*H, 1, S]


def compute_value_attention_shaw(
    embedding: nn.Embedding, attn_probs: torch.Tensor, relative_positions: torch.Tensor
) -> torch.Tensor:
    """
    Computes relative position representations as described in (Shaw et al. 2018)
    S = source length
    B = batch_size
    H = number of heads
    HD = head_dim

    :param attn_probs:          shape [B*H, S, S]
    :param relative_positions:  shape [S, S]
    :return: a_v                shape [B*H, S, HD]
    """
    v_rpr_embed = embedding(relative_positions)  # [S, S]  # [S, S, HD]
    v_rpr = (
        torch.bmm(
            attn_probs.transpose(0, 1), v_rpr_embed  # [S, B*H, S]  # [S, S, HD]
        )  # [S, B*H, HD]
    ).transpose(
        0, 1
    )  # [B*H, S, HD]

    return v_rpr  # [B*H, S, HD]


class RelativeMultiheadSelfAttention(MultiheadAttention):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        key_rpr: bool = True,
        value_rpr: bool = True,
        num_rpr_embeddings: int = None,  # without padding offset
        pad_idx: Optional[int] = None,  # for padded relative_distances
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim,
            vdim,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            self_attention,
            encoder_decoder_attention,
        )

        assert (
            self.self_attention
        ), "RelativeMultiheadAttention only with self-attention "

        # relative positional params
        self.pad_idx = pad_idx

        if pad_idx is not None and num_rpr_embeddings is not None:
            num_rpr_embeddings += pad_idx + 1

        self.key_rpr_embedding = (
            nn.Embedding(
                num_embeddings=num_rpr_embeddings,
                embedding_dim=self.head_dim,
                padding_idx=pad_idx,
            )
            if key_rpr
            else None
        )

        self.value_rpr_embedding = (
            nn.Embedding(
                num_embeddings=num_rpr_embeddings,
                embedding_dim=self.head_dim,
                padding_idx=pad_idx,
            )
            if value_rpr
            else None
        )

        self.num_rpr_embeddings = num_rpr_embeddings

        if not (value_rpr or key_rpr):
            import warnings

            warnings.warn(
                "Neither value or key RPR are set, you might want to use regular transformer"
                "without relative position representations instead."
            )

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        # if self.key_rpr_embedding is not None:
        #     nn.init.xavier_uniform_(self.key_rpr_embedding.weight, gain=1 / math.sqrt(2))
        #
        # if self.value_rpr_embedding is not None:
        #     nn.init.xavier_uniform_(self.value_rpr_embedding.weight, gain=1 / math.sqrt(2))

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        key_attention_func=compute_key_attention_gather,  # todo make this fix
        value_attention_func=compute_value_attention_shaw,
        relative_distances: Optional[torch.LongTensor] = None,
        tree_attn_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return
            relative_distances (LongTensor, optional): Pre-compute relative
                distances between elements in a batch. If None relative
                distances will be computed based on query and key sequences.
                Should be of shape [q_len, k_len] if relative distances are
                the same for all elements in a batch, otherwise [bsz, q_len, k_len].
            tree_attn_mask (BoolTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len, src_len)`, where
                padding elements are indicated by 1s.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)  # [S, B, D] -> [S, B, D']   (SAVED: [1, B, D])
            k = self.k_proj(query)  # [S, B, D] -> [S, B, D']   (SAVED: [1, B, D])
            v = self.v_proj(query)  # [S, B, D] -> [S, B, D']   (SAVED: [1, B, D])
        elif self.encoder_decoder_attention:
            raise ValueError("Not supported in RPR")
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)  # [S, B, D] -> [S, B, D']
            k = self.k_proj(key)  # [S, B, D] -> [S, B, D']
            v = self.v_proj(value)  # [S, B, D] -> [S, B, D']
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        # HD = D // num_heads = head_dim
        q = (
            q.contiguous()  # [S, B, D']
            .view(-1, bsz * self.num_heads, self.head_dim)  # [S, B*H, HD]
            .transpose(0, 1)  # [B*H, S, HD]
        )  # [B*H, S, HD] (SAVED: [B*H, 1, D'])
        if k is not None:
            k = (
                k.contiguous()  # [S, B, D]
                .view(-1, bsz * self.num_heads, self.head_dim)  # [S, B*H, HD]
                .transpose(0, 1)  # [B*H, S, HD]
            )  # S,B,D ->  [B*H, S, head_dim]  # (SAVED: [B*H, 1, D'])
        if v is not None:
            v = (
                v.contiguous()  # [S, B, D]
                .view(-1, bsz * self.num_heads, self.head_dim)  # [S, B*H, HD]
                .transpose(0, 1)  # [B*H, S, HD]
            )  # S,B,D ->  [B*H, S, head_dim]  # (SAVED: [B*H, 1, D'])

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

        ## if saved state exists (when we're decoding)
        # the previous output tokens have been preprended to key and value.
        # q = [B*H, 1, HD]
        # k = [B*H, S, HD]
        # v = [B*H, S, HD]

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        # [B*H, S, head_dim] x [B*H, head_dim, S]
        attn_weights = torch.bmm(
            q, k.transpose(1, 2)
        )  # [B*H, S, S]  (saved: [B*H, 1, S])
        attn_weights = MultiheadAttention.apply_sparse_mask(
            attn_weights, tgt_len, src_len, bsz
        )

        #### rpr attention ####
        if relative_distances is None and (
            self.key_rpr_embedding or self.value_rpr_embedding is not None
        ):
            with torch.no_grad():
                relative_distances = generate_relative_distances_matrix(
                    length_q=q.size(1),
                    length_k=k.size(1),
                    max_relative_distance=(self.num_rpr_embeddings // 2),
                ).to(
                    q.device
                )  # [S, S]
            # raise ValueError("Compute relative distances before!")

        assert relative_distances.ndimension() in {2, 3}
        assert relative_distances.size(-2) == q.size(1)
        assert relative_distances.size(-1) == src_len

        if self.key_rpr_embedding is not None:
            k_rpr = key_attention_func(  # equation 5 in paper
                embedding=self.key_rpr_embedding,
                q=q.view(bsz, self.num_heads, -1, self.head_dim),
                relative_distances=relative_distances,
            )  # [B*H, S, S]
            attn_weights += k_rpr  # [B*H, S, S]

        if tree_attn_mask is not None:
            # mask
            assert tree_attn_mask.size(0) == bsz
            assert tree_attn_mask.size(1) == tree_attn_mask.size(2) == src_len

            attn_weights = attn_weights.view(bsz, self.num_heads, src_len, src_len)
            attn_weights = attn_weights.masked_fill(
                tree_attn_mask.unsqueeze(1).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, src_len, src_len)
        ##### /rpr ######

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )  # [B*H, S, S]

        attn = torch.bmm(attn_probs, v)  # [B*H, S, S]  # [B*H, S, HD]  # [B*H, S, HD]

        #### rpr attention ####
        # value relative positions
        if self.value_rpr_embedding is not None:
            v_rpr = value_attention_func(
                embedding=self.value_rpr_embedding,
                attn_probs=attn_probs,
                relative_positions=relative_distances,
            )
            attn += v_rpr
        #### /rpr ####

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights


def print_rounded(x, msg: str = "", decimals=3):
    print(msg, ((x * 10 ** decimals).round() / (10 ** decimals)))
