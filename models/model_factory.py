from nbow_model import NbowModel
from typing import Dict, Any
from torch.utils.data import Dataset


class ModelFactory:
    def __init__(
        self,
        hyperparameters: Dict[str, Any],
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
    ):
        self.hypers = hyperparameters
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

    def get_model(self, model_type: str = "nbow_model"):
        if model_type == "nbow_model":
            return NbowModel(
                self.hypers, self.train_dataset, self.valid_dataset, self.test_dataset
            )
        else:
            print(f"Model: {model_type} is not implemented!")
            raise NotImplementedError
