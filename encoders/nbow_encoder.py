import torch
from torch import nn
from .encoder_base import EncoderBase
from typing import Dict, Any


class NbowEncoder(EncoderBase):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        dropout: float,
        vocab_count_threshold: int,
        use_bpe: bool,
        vocab_pct_bpe: float,
    ):
        super().__init__(vocab_size, vocab_count_threshold, use_bpe, vocab_pct_bpe)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, seq_tokens, seq_tokens_mask, seq_len):
        seq_token_embeddings = self.dropout(self.embeddings(seq_tokens))
        seq_token_embeddings_masked = seq_token_embeddings * seq_tokens_mask.unsqueeze(
            dim=-1
        )
        seq_token_embeddings_sum = seq_token_embeddings_masked.sum(dim=1)
        seq_lengths = seq_len.to(dtype=torch.float32).unsqueeze(dim=-1)
        return seq_token_embeddings_sum / seq_lengths
