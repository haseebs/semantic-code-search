import random
import wandb
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_metric_learning import losses

from typing import Dict, Any
from more_itertools import sliced
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from encoders.encoder_factory import EncoderFactory


class ModelBase(pl.LightningModule):
    def __init__(
        self,
        hparams: Dict[str, Any],
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
    ):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.get_encoders()

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
        loss, mrr, _, _ = self.get_eval_metrics(out["code_embs"], out["query_embs"])

        tqdm_dict = {"train_mrr": mrr}
        log_dict = {"train_loss": loss, "train_mrr": mrr}

        return {"loss": loss, "progress_bar": tqdm_dict, "log": log_dict}

    def validation_step(self, batch, batch_idx):
        # TODO randomize
        code_embs, query_embs = self.forward(batch)
        loss, mrr, _, _ = self.get_eval_metrics(code_embs, query_embs)
        return {"loss": loss, "mrr": mrr}

    def validation_epoch_end(self, out):
        avg_loss = torch.stack([x["loss"] for x in out]).mean()
        # TODO mean of means != total mean
        avg_mrr = torch.stack([x["mrr"] for x in out]).mean()
        log_dict = {"val_loss": avg_loss, "val_mrr": avg_mrr}

        return {"val_loss": avg_loss, "progress_bar": log_dict, "log": log_dict}

    def test_step(self, batch, batch_idx):
        code_embs, query_embs = self.forward(batch)
        return {"batch": batch, "code_embs": code_embs, "query_embs": query_embs}

    def test_epoch_end(self, out):
        sz = out[0]["code_embs"].shape[1]
        all_code_embs = torch.stack([x["code_embs"] for x in out]).view(-1, sz)
        all_query_embs = torch.stack([x["query_embs"] for x in out]).view(
            -1, sz
        )

        code_embs_batched = sliced(all_code_embs, 1000)
        query_embs_batched = sliced(all_query_embs, 1000)
        all_mrr, all_similarity_scores, all_ranks = [], [], []
        for code_embs, query_embs in tqdm(zip(code_embs_batched, query_embs_batched)):
            if code_embs.shape[0] < 1000:
                break
            _, mrr, similarity_scores, ranks = self.get_eval_metrics(
                code_embs.to(self.hparams["test_device"]),
                query_embs.to(self.hparams["test_device"]),
            )
            all_mrr.append(mrr)
            all_similarity_scores.append(similarity_scores)
            all_ranks.append(ranks)

        # self.make_examples(batch, similarity_scores, ranks)

        avg_mrr = torch.stack(all_mrr).mean()
        log_dict = {"test_mrr": avg_mrr}
        wandb.run.summary["test_mrr"] = avg_mrr.item()  # shouldnt have to do this
        return {"test_mrr": avg_mrr, "progress_bar": log_dict, "log": log_dict}

    def get_eval_metrics(self, code_embs: torch.tensor, query_embs: torch.tensor):
        query_norm = F.normalize(query_embs, dim=-1) + 1e-10
        code_norm = F.normalize(code_embs, dim=-1) + 1e-10
        similarity_scores = code_norm @ query_norm.T
        # neg_matrix = torch.diag(
        #    torch.tensor([float("-inf")] * similarity_scores.shape[0])
        # )
        # per_sample_loss = torch.max(
        #    torch.tensor(0.0).cuda(),
        #    self.hparams["margin"]
        #    - similarity_scores.diagonal()
        #    + torch.max(F.relu(similarity_scores + neg_matrix.cuda()), dim=-1)[0],
        # )
        # total_loss = per_sample_loss.mean()
        embs = torch.cat([query_embs, code_embs])
        labels = torch.arange(1, query_embs.shape[0] + 1)
        labels = torch.cat([labels, labels])
        loss_func = losses.TripletMarginLoss(
            margin=self.hparams["margin"], triplets_per_anchor="all"
        )
        total_loss = loss_func(embs, labels)

        correct_scores = similarity_scores.diagonal().detach()
        compared_scores = similarity_scores >= correct_scores.unsqueeze(dim=-1)
        ranks = compared_scores.sum(dim=1)
        mrr = (1 / ranks.to(dtype=torch.float)).mean()

        return total_loss, mrr, similarity_scores, ranks

    def make_examples(self, batch, similarity_scores, ranks):
        # max_examples = 250 if 250 < self.hparams["batch_size"] else self.hparams["batch_size"]
        max_examples = 100
        language = self.test_dataset.original_data[0]["language"]
        predictions = torch.argmax(similarity_scores, dim=1)
        r = random.sample(range(self.hparams["batch_size"]), max_examples)

        selected_ranks = ranks[r]
        selected_predictions = predictions[r]
        predicted_original_idx = batch["original_data_idx"][r]
        predicted_original_data = [
            self.test_dataset.original_data[i]["code"] for i in predicted_original_idx
        ]

        query_original_idx = batch["original_data_idx"][r]
        query_original_data = [
            self.test_dataset.original_data[i]["docstring"] for i in query_original_idx
        ]

        examples_table = []
        examples_table_columns = ["Rank", "Language", "Query", "Code"]

        for idx in range(max_examples):
            markdown_code = (
                "```%s\n" % language
                + predicted_original_data[idx].strip("\n")
                + "\n```"
            )
            examples_table.append(
                [
                    selected_ranks[idx].item(),
                    language,
                    query_original_data[idx],
                    markdown_code,
                ]
            )

        self.logger.experiment.log(
            {
                "Test Examples": wandb.Table(
                    columns=examples_table_columns, rows=examples_table
                )
            }
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])

    def get_encoders(self):
        encoder_factory = EncoderFactory(self.hparams)
        if self.hparams["encoder_sharing_mode"] == "per_code_language":
            raise NotImplementedError
        elif self.hparams["encoder_sharing_mode"] == "per_input_source":
            self.code_encoder = encoder_factory.get_encoder(
                self.hparams["code_encoder_type"]
            )
            self.query_encoder = encoder_factory.get_encoder(
                self.hparams["query_encoder_type"]
            )
        elif self.hparams["encoder_sharing_mode"] == "all":
            self.code_encoder = encoder_factory.get_encoder(
                self.hparams["code_encoder_type"]
            )
            self.query_encoder = self.code_encoder

    def init_encoders(self):
        # TODO cleanup this mess
        print("Building vocabulary...")
        for sample in self.train_dataset.original_data:
            if self.hparams["code_encoder_type"] == "tree_attention_encoder":
                self.code_encoder.update_tokens_from_sample(sample["code_ast_tokens"])
            else:
                self.code_encoder.update_tokens_from_sample(sample["code_tokens"])
            self.query_encoder.update_tokens_from_sample(
                [t.lower() for t in sample[self.hparams["key_docstring_tokens"]]]
            )
        self.code_encoder.build_vocabulary()
        self.query_encoder.build_vocabulary()

    def prepare_data(self):
        # do not prepare data again when testing
        if len(self.train_dataset.original_data) == 0:
            return

        self.init_encoders()

        print("Tokenizing data...")
        self.train_dataset.encode_data(self.query_encoder, self.code_encoder)
        self.valid_dataset.encode_data(self.query_encoder, self.code_encoder)
        self.test_dataset.encode_data(self.query_encoder, self.code_encoder)

        # free up memory
        self.train_dataset.original_data = []
        self.valid_dataset.original_data = []

        # dirty hack to not let pytorch_lightning finalize the session before testing
        self.logger.finalize = lambda *args: None

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            drop_last=True,
        )


if __name__ == "__main__":
    from IPython import embed

    embed()
