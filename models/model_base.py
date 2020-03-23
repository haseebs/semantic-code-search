import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from encoders.encoder_factory import EncoderFactory


class ModelBase(pl.LightningModule):
    def __init__(
        self,
        hypers: Dict[str, Any],
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
    ):
        super().__init__()
        self.hypers = hypers
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        self.init_encoders()

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

    def training_step(self, batch, batch_idx):
        code_embs, query_embs = self.forward(batch)
        return {"code_embs": code_embs, "query_embs": query_embs}

    def training_end(self, out):
        loss, mrr = self.get_eval_metrics(out["code_embs"], out["query_embs"])

        tqdm_dict = {"train_mrr": mrr}
        log_dict = {"train_loss": loss, "train_mrr": mrr}

        return {"loss": loss, "progress_bar": tqdm_dict, "log": log_dict}

    def validation_step(self, batch, batch_idx):
        code_embs, query_embs = self.forward(batch)
        loss, mrr = self.get_eval_metrics(code_embs, query_embs)
        return {"loss": loss, "mrr": mrr}

    def validation_epoch_end(self, out):
        avg_loss = torch.stack([x["loss"] for x in out]).mean()
        avg_mrr = torch.stack(
            [x["mrr"] for x in out]
        ).mean()  # TODO mean of means != total mean

        log_dict = {"val_loss": avg_loss, "val_mrr": avg_mrr}

        return {"val_loss": avg_loss, "progress_bar": log_dict, "log": log_dict}

    def test_step(self, batch, batch_idx):
        code_embs, query_embs = self.forward(batch)
        _, mrr = self.get_eval_metrics(code_embs, query_embs)
        return {"mrr": mrr}

    def test_epoch_end(self, out):
        avg_mrr = torch.stack([x["mrr"] for x in out]).mean()
        log_dict = {"test_mrr": avg_mrr}

        wandb_path = f"haseebs/{self.logger.experiment.project}/{self.logger.experiment.id}"
        run = wandb.Api().run(wandb_path)
        run.summary["test_mrr"] = avg_mrr
        run.summary.update()

        return {"test_mrr": avg_mrr, "progress_bar": log_dict, "log": log_dict}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hypers["learning_rate"])

    def init_encoders(self):
        encoder_factory = EncoderFactory(self.hypers)
        self.code_encoder = encoder_factory.get_encoder(self.hypers["code_encoder_type"])
        self.query_encoder = encoder_factory.get_encoder(self.hypers["query_encoder_type"])

        # TODO cleanup this mess
        for sample in self.train_dataset.original_data:
            self.code_encoder.update_tokens_from_sample(sample["code_tokens"])
            self.query_encoder.update_tokens_from_sample(
                [t.lower() for t in sample["docstring_tokens"]]
            )
        self.code_encoder.build_vocabulary()
        self.query_encoder.build_vocabulary()

    def get_eval_metrics(self, code_embs: torch.tensor, query_embs: torch.tensor):
        query_norm = F.normalize(query_embs, dim=-1) + 1e-10
        code_norm = F.normalize(code_embs, dim=-1) + 1e-10
        similarity_scores = code_norm @ query_norm.T
        # similarity_scores = query_embs/query_norm @ (code_embs/code_norm).T
        neg_matrix = torch.diag(torch.tensor([float("-inf")] * similarity_scores.shape[0]))
        per_sample_loss = torch.max(
            torch.tensor(0.0).cuda(),
            self.hypers["margin"]
            - similarity_scores.diagonal()
            + torch.max(F.relu(similarity_scores + neg_matrix.cuda()), dim=-1)[0],
        )
        total_loss = per_sample_loss.mean()
        correct_scores = similarity_scores.diagonal().detach()
        compared_scores = similarity_scores >= correct_scores.unsqueeze(dim=-1)
        mrr = (1 / compared_scores.sum(dim=1).to(dtype=torch.float)).mean()

        return total_loss, mrr

    def train_dataloader(self):
        if (
            len(self.train_dataset.encoded_data) == 0
        ):  # TODO Why is this being called more than once
            self.train_dataset.encode_data(self.query_encoder, self.code_encoder)
            del self.train_dataset.original_data  # save memory
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hypers["batch_size"],
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if len(self.valid_dataset.encoded_data) == 0:
            self.valid_dataset.encode_data(self.query_encoder, self.code_encoder)
            del self.valid_dataset.original_data  # save memory
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.hypers["batch_size"],
            shuffle=True,
            drop_last=True,
        )

    def test_dataloader(self):
        self.test_dataset.encode_data(self.query_encoder, self.code_encoder)
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hypers["batch_size"],
            shuffle=False,
            drop_last=True,
        )


if __name__ == "__main__":
    from IPython import embed

    embed()
