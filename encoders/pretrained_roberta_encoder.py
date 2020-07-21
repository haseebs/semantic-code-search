import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaTokenizer, RobertaForMaskedLM


class PretrainedRobertaEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaForMaskedLM.from_pretrained(
            "roberta-base", output_hidden_states=True
        )
        self.linear = torch.nn.Linear(
            self.model.base_model.embeddings.word_embeddings.weight.shape[1], 128
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, seq_tokens_mask):
        output = self.model(src, seq_tokens_mask)[1][
            0
        ]  # TODO make sure this is last hidden layer and not the first
        output = self.linear(output)

        seq_token_embeddings_sum = output.sum(
            dim=1
        )  # TODO Use seq_tokens_mask to exclude padding idxes
        seq_lengths = seq_tokens_mask.sum(dim=1).unsqueeze(1)
        return seq_token_embeddings_sum / seq_lengths
