import os
import sys

sys.path.append("/home/haseebs/workspace/CSN/semantic_code_search")

import wandb
from pytorch_lightning import loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset import CSNDataset
from models.model_base import ModelBase


def run():
    hypers = {
        "code_encoder_type": "nbow_encoder",
        "query_encoder_type": "nbow_encoder",
        "code_max_num_tokens": 200,
        "query_max_num_tokens": 30,
        "learning_rate": 0.01,
        "batch_size": 1000,
        "loss": "cosine",
        "vocab_size": 10000,
        "embedding_dim": 128,
        "dropout_prob": 0.1,
        "gradient_clip": 1,
        "margin": 1,
        "max_epochs": 300,
        "patience": 5,
        "use_bpe": True,
        "vocab_pct_bpe": 0.5,
        "vocab_count_threshold": 10,
    }

    keep_keys = set(["language", "docstring_tokens", "code_tokens"])
    keep_keys_test = set(
        ["language", "docstring_tokens", "code_tokens", "docstring", "code"]
    )

    logger = loggers.WandbLogger(
        experiment=wandb.init(project="semantic-code-search")
    )

    train_dataset = CSNDataset(
        hypers=hypers, keep_keys=keep_keys, data_split="train"
    )
    valid_dataset = CSNDataset(
        hypers=hypers, keep_keys=keep_keys, data_split="valid"
    )
    test_dataset = CSNDataset(
        hypers=hypers, keep_keys=keep_keys_test, data_split="test"
    )

    model = ModelBase(hypers, train_dataset, valid_dataset, test_dataset)

    early_stop_callback = EarlyStopping(
        monitor="val_mrr",
        min_delta=0.00,
        patience=hypers["patience"],
        verbose=True,
        mode="max",
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=wandb.run.dir + "/{epoch:02d}_best_checkpoint",
        monitor="val_mrr",
        verbose=True,
        mode="max",
    )

    trainer = Trainer(
        max_nb_epochs=hypers["max_epochs"],
        gradient_clip_val=hypers["gradient_clip"],
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        progress_bar_refresh_rate=10,
        logger=logger,
        # train_percent_check=0.1,
        gpus=1,
    )

    trainer.fit(model)
    trainer.test(model)

    # close session since we used the hack
    wandb.join(0)


if __name__ == "__main__":
    run()
