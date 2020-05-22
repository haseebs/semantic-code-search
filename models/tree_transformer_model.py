import random
import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from encoders.encoder_factory import EncoderFactory
from .transformer_model import TransformerModel


class TreeTransformerModel(TransformerModel):
    def __init__(
        self,
        hypers: Dict[str, Any],
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
    ):
        super(TreeTransformerModel, self).__init__(
            hypers, train_dataset, valid_dataset, test_dataset
        )

    def forward(self, batch):
        code_embs = self.code_encoder(
            src=batch["encoded_code"],
            seq_tokens_mask=batch["encoded_code_mask"],
            seq_len=batch["encoded_code_length"],
            src_descendants=batch["code_ast_descendants"],
        )
        query_embs = self.query_encoder(
            src=batch["encoded_query"],
            seq_tokens_mask=batch["encoded_query_mask"],
            seq_len=batch["encoded_query_length"],
        )
        return code_embs, query_embs
