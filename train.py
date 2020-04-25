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
    logger = loggers.WandbLogger(experiment=wandb.init(project="semantic-code-search"))

    train_dataset = CSNDataset(
        hypers=wandb.config, keep_keys=wandb.config["keep_keys"], data_split="train"
    )
    valid_dataset = CSNDataset(
        hypers=wandb.config, keep_keys=wandb.config["keep_keys"], data_split="valid"
    )
    test_dataset = CSNDataset(
        hypers=wandb.config, keep_keys=wandb.config["keep_keys_test"], data_split="test"
    )

    model = ModelBase(wandb.config, train_dataset, valid_dataset, test_dataset)

    early_stop_callback = EarlyStopping(
        monitor="val_mrr",
        min_delta=0.00,
        patience=wandb.config["patience"],
        verbose=False,
        mode="max",
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=wandb.run.dir + "/{epoch:02d}_best_checkpoint",
        monitor="val_mrr",
        verbose=False,
        mode="max",
    )

    trainer = Trainer(
        max_nb_epochs=wandb.config["max_epochs"],
        gradient_clip_val=wandb.config["gradient_clip"],
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        progress_bar_refresh_rate=10,
        logger=logger,
        # train_percent_check=0.1,
        gpus=1,
    )

    trainer.fit(model)
    trainer.test(model)

    # close session manually since we used the hack
    wandb.join(0)


if __name__ == "__main__":
    run()
