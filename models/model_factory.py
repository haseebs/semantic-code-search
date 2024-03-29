from typing import Dict, Any
from torch.utils.data import Dataset
from .model_base import ModelBase
from .transformer_model import TransformerModel
from .tree_transformer_model import TreeTransformerModel
from .pretrained_model import PretrainedModel
from .es_model import EsModel


class ModelFactory:
    def __init__(
        self,
        hparams: Dict[str, Any],
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
    ):
        self.hparams = hparams
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

    def get_model(self, model_type: str = "nbow_model"):
        if model_type == "nbow_model":
            return ModelBase(
                self.hparams, self.train_dataset, self.valid_dataset, self.test_dataset,
            )
        elif model_type == "transformer_model":
            return TransformerModel(
                self.hparams, self.train_dataset, self.valid_dataset, self.test_dataset,
            )
        elif model_type == "tree_transformer_model":
            return TreeTransformerModel(
                self.hparams, self.train_dataset, self.valid_dataset, self.test_dataset,
            )
        elif model_type == "pretrained_model":
            return PretrainedModel(
                self.hparams, self.train_dataset, self.valid_dataset, self.test_dataset,
            )
        elif model_type == "es_model":
            return EsModel(
                self.hparams, self.train_dataset, self.valid_dataset, self.test_dataset,
            )
        else:
            print(f"Model: {model_type} is not implemented!")
            raise NotImplementedError
