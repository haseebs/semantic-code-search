import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_base import EncoderBase
from .positional_encoder import PositionalEncoder
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SelfAttentionEncoder(EncoderBase):
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
    ):
        super().__init__(ntoken, vocab_count_threshold, use_bpe, vocab_pct_bpe)
        self.src_mask = None
        self.pos_encoder = PositionalEncoder(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        # self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, seq_tokens_mask, seq_len):
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #    device = src.device
        #    mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #    self.src_mask = mask

        seq_tokens_mask = (1 - seq_tokens_mask).T > 0
        src = src.T
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=seq_tokens_mask)
        # output = self.decoder(output)
        seq_token_embeddings_sum = output.sum(dim=1)
        seq_lengths = seq_len.to(dtype=torch.float32).unsqueeze(dim=-1)
        return seq_token_embeddings_sum / seq_lengths
