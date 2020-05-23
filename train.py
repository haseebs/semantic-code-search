import os
import sys
import yaml

sys.path.append("/home/haseebs/workspace/CSN/semantic_code_search")

import wandb
import torch
import argparse
import numpy as np
from pytorch_lightning import loggers
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset import CSNDataset
from models.model_factory import ModelFactory


def run():

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true", help="whether to train")
    parser.add_argument(
        "-e", "--evaluate", action="store_true", help="whether to evaluate only"
    )
    parser.add_argument(
        "-l", "--load", action="store", type=str, help="path to checkpoint"
    )
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        default="config-default.yaml",
        type=str,
        help="path to config",
    )
    args = parser.parse_args()
    if (not args.train and not args.evaluate) or (args.evaluate and not args.load):
        print("wrong args passed")
        exit(0)

    print(f"Loading config parameters from {args.config}")
    cfg_file = yaml.safe_load(open(args.config))

    run_id = None
    if args.load:
        run_id = args.load.split("/")[1].split("-")[2]
    logger = loggers.WandbLogger(
        experiment=wandb.init(
            project="semantic-code-search", resume=run_id, config=cfg_file
        )
    )

    seed_everything(wandb.config["seed"])

    train_dataset = CSNDataset(
        hparams=wandb.config, keep_keys=wandb.config["keep_keys"], data_split="train"
    )
    valid_dataset = CSNDataset(
        hparams=wandb.config, keep_keys=wandb.config["keep_keys"], data_split="valid"
    )
    test_dataset = CSNDataset(
        hparams=wandb.config,
        keep_keys=wandb.config["keep_keys_test"],
        data_split="test",
    )

    model_factory = ModelFactory(
        {k: wandb.config.get(k) for k in wandb.config.keys()},
        train_dataset,
        valid_dataset,
        test_dataset,
    )
    model = model_factory.get_model(wandb.config["model_type"])

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

    if args.load and args.train:
        print(f"Loading checkpoint from #{args.load} and evaluating")

    trainer = Trainer(
        max_nb_epochs=wandb.config["max_epochs"],
        gradient_clip_val=wandb.config["gradient_clip"],
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        progress_bar_refresh_rate=10,
        logger=logger,
        deterministic=True,
        resume_from_checkpoint=args.load,
        # train_percent_check=0.01,
        gpus=1,
    )
    # from IPython import embed; embed()

    if args.train:
        trainer.fit(model)
    if args.evaluate and args.load:
        print(f"Loading checkpoint from #{args.load} and evaluating on test set")
        model = model.load_from_checkpoint(
            args.load,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )

    trainer.test(model)

    # close session manually since we used the hack
    wandb.join(0)


if __name__ == "__main__":
    run()
