import torch
from torch import nn
from .encoder_base import EncoderBase
from typing import Dict, Any


class NbowEncoder(EncoderBase):
    def __init__(self, hyperparameters: Dict[str, Any]):
        super().__init__(hyperparameters)
        self.embeddings = nn.Embedding(
            self.hypers["vocab_size"], self.hypers["embedding_dim"]
        )
        self.dropout = nn.Dropout(p=self.hypers["dropout_prob"])

    def forward(self, seq_tokens, seq_tokens_mask, seq_len):
        seq_token_embeddings = self.dropout(self.embeddings(seq_tokens))
        seq_token_embeddings_masked = (
            seq_token_embeddings * seq_tokens_mask.unsqueeze(dim=-1)
        )
        seq_token_embeddings_sum = seq_token_embeddings_masked.sum(dim=1)
        seq_lengths = seq_len.to(dtype=torch.float32).unsqueeze(dim=-1)
        return seq_token_embeddings_sum / seq_lengths
