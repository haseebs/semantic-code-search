from model_base import ModelBase
from typing import List, Dict, Any
from torch.utils.data import Dataset


class NBowModel(ModelBase):
    def __init__(self, hyperparameters: Dict[str, Any], dataset: Dataset):
        super.__init__(hyperparameters, dataset)
        self.init_encoders()

    def init_encoders(self):
        self.code_encoder = self.encoder_factory.get_encoder(
            self.hparams["code_encoder_type"]
        )
        self.query_encoder = self.encoder_factory.get_encoder(
            self.hparams["query_encoder_type"]
        )
