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


class PretrainedModel(ModelBase):
    def __init__(
        self,
        hparams: Dict[str, Any],
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
    ):
        super(PretrainedModel, self).__init__(
            hparams, train_dataset, valid_dataset, test_dataset
        )
        if self.hparams["freeze_pretrained_embeddings"]:
            self.query_encoder.model.roberta.embeddings.requires_grad_(False)

    def forward(self, batch):
        code_embs = self.code_encoder(
            src=batch["encoded_code"],
            seq_tokens_mask=batch["encoded_code_mask"],
            seq_len=batch["encoded_code_length"],
        )
        query_embs = self.query_encoder(
            src=batch["encoded_query"], seq_tokens_mask=batch["encoded_query_mask"],
        )
        return code_embs, query_embs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])

    def on_after_backward(self):
        for param in self.query_encoder.parameters():
            param.grad *= self.hparams["query_grad_multiplier"]
        #from IPython import embed; embed()
        #for k, v in self.state_dict().items():
        #    if 'query_encoder' in k:
        #        self.state_dict()[k] *= self.hparams["query_grad_multiplier"]

    def init_encoders(self):
        # TODO cleanup this mess
        print("Building vocabulary...")
        for sample in self.train_dataset.original_data:
            if self.hparams["code_encoder_type"] == "pretrained_roberta_encoder":
                pass
            elif self.hparams["code_encoder_type"] == "tree_attention_encoder":
                self.code_encoder.update_tokens_from_sample(sample["code_ast_tokens"])
            else:
                self.code_encoder.update_tokens_from_sample(sample["code_tokens"])
            if self.hparams["query_encoder_type"] == "pretrained_roberta_encoder":
                continue
            else:
                self.query_encoder.update_tokens_from_sample(
                    [t.lower() for t in sample[self.hparams["key_docstring_tokens"]]]
                )
        if self.hparams["code_encoder_type"] != "pretrained_roberta_encoder":
            self.code_encoder.build_vocabulary()
        if self.hparams["query_encoder_type"] != "pretrained_roberta_encoder":
            self.query_encoder.build_vocabulary()
