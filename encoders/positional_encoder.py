import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from astlib.tensor_utils.analyze import level


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoder, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """ x should have dimensions [N, B, D] """

        x = x + self.pe[: x.size(0), :]
        return x


class LevelPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, num_embeddings: int = 200):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.pos_embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)

    def forward(self, x, node_incidences):
        """
            x should have dimensions [N, B, D]
            num_descendants is expected to be of size [B x N x N].
        """

        # root has level 0 and padding tokens should have level -1
        levels = level(node_incidences)  # [B, N]
        levels += 1  # shift everything so that padding tokens have level 0
        levels = levels.t()  # [N, B]

        x = x + self.pos_embedding(levels)  # [N, B, D]
        return x
