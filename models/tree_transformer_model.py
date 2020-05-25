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
        hparams: Dict[str, Any],
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
    ):
        super(TreeTransformerModel, self).__init__(
            hparams, train_dataset, valid_dataset, test_dataset
        )

    def forward(self, batch):

        code_embs, code_features = self.code_encoder(
            src=batch["encoded_code"],
            seq_tokens_mask=batch["encoded_code_mask"],
            seq_len=batch["encoded_code_length"],
            src_descendants=batch["code_ast_descendants"],
        )

        ancestor_logits = None
        if self.hparams["tree_transformer_ancestor_prediction"]:
            ancestor_source_node1 = batch["ancestor_source_node1"]  # [B, V]  (V == #'tree_transformer_ancestor_prediction')
            ancestor_source_node2 = batch["ancestor_source_node2"]  # [B, V]
            node_pairs = torch.cat(
                (
                    ancestor_source_node1[:, :, None],
                    ancestor_source_node2[:, :, None],
                ),
                dim=-1
            )  # [B, V, 2]
            ancestor_logits = self.code_encoder.predict_ancestors(code_features, node_pairs)

        query_embs = self.query_encoder(
            src=batch["encoded_query"],
            seq_tokens_mask=batch["encoded_query_mask"],
            seq_len=batch["encoded_query_length"],
        )
        return code_embs, query_embs, ancestor_logits

    def training_step(self, batch, batch_idx):
        code_embs, query_embs, ancestor_logits = self.forward(batch)

        return {
            "code_embs": code_embs,
            "query_embs": query_embs,
            "ancestor_logits": ancestor_logits,
            "ancestor_target": batch.get("ancestor_target", None)
        }

    def training_end(self, out: Dict):

        loss, mrr, _, _ = self.get_eval_metrics(out["code_embs"], out["query_embs"])

        tqdm_dict = {"train_mrr": mrr}
        log_dict = {"train_loss": loss, "train_mrr": mrr}

        if out.get("ancestor_logits", None) is not None:
            ancestor_loss = self.compute_ancestor_loss(
                ancestor_logits=out["ancestor_logits"],
                ancestor_target=out["ancestor_target"],
            )
            # todo make configurable
            loss += 0.3 * ancestor_loss
            log_dict["train_loss"] = loss
            log_dict["ancestor_loss"] = ancestor_loss
            tqdm_dict["anc_loss"] = ancestor_loss

        return {"loss": loss, "progress_bar": tqdm_dict, "log": log_dict}

    def validation_step(self, batch, batch_idx):
        code_embs, query_embs, ancestor_logits = self.forward(batch)
        loss, mrr, _, _ = self.get_eval_metrics(code_embs, query_embs)
        log_dict = {"loss": loss, "mrr": mrr}

        if ancestor_logits is not None:
            ancestor_loss = self.compute_ancestor_loss(
                ancestor_logits=ancestor_logits,
                ancestor_target=batch["ancestor_target"],
            )
            # todo make configurable
            loss += 0.3 * ancestor_loss
            log_dict["loss"] = loss
            log_dict["ancestor_loss"] = ancestor_loss

        return log_dict

    def validation_epoch_end(self, out):
        # TODO mean of means != total mean
        avg_loss = torch.stack([x["loss"] for x in out]).mean()
        log_dict = {
            "val_loss": avg_loss,
            "val_mrr": torch.stack([x["mrr"] for x in out]).mean()
        }

        if "ancestor_loss" in out:
            log_dict["val_ancestor_loss"] = torch.stack([x["ancestor_loss"] for x in out]).mean()

        return {"val_loss": avg_loss, "progress_bar": log_dict, "log": log_dict}

    @staticmethod
    def compute_ancestor_loss(ancestor_logits, ancestor_target):
        lprobs = F.log_softmax(ancestor_logits, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        target = ancestor_target.view(-1)

        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=-1,
            reduction='mean'
        )

        return loss
