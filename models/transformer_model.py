import random
import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from encoders.encoder_factory import EncoderFactory
from .model_base import ModelBase


class TransformerModel(ModelBase):
    def __init__(
        self,
        hparams: Dict[str, Any],
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
    ):
        super(TransformerModel, self).__init__(
            hparams, train_dataset, valid_dataset, test_dataset
        )

    def forward(self, batch):
        code_embs = self.code_encoder(
            seq_tokens=batch["encoded_code"],
            seq_tokens_mask=batch["encoded_code_mask"],
            seq_len=batch["encoded_code_length"],
        )
        query_embs = self.query_encoder(
            seq_tokens=batch["encoded_query"],
            seq_tokens_mask=batch["encoded_query_mask"],
            seq_len=batch["encoded_query_length"],
        )
        return code_embs, query_embs
