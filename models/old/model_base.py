from abc import ABC, abstractmethod
from typing import Dict, Any
from torch.utils.data import Dataset, SequentialSampler, RandomSampler, BatchSampler
from encoders.encoder_factory import EncoderFactory


class ModelBase(ABC):
    def __init__(
        self,
        hyperparameters: Dict[str, Any],
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
    ):
        self.hypers = hyperparameters
        self.train_sampler = get_data_sampler(train_dataset, random=True)
        self.valid_sampler = get_data_sampler(valid_dataset, random=True)
        self.test_sampler = get_data_sampler(test_dataset, random=False)

    @abstractmethod
    def training_step(self, minibatch):
        pass

    @abstractmethod
    def evaluation_step(self, minibatch):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def init_encoders(self, dataset: Dataset):
        encoder_factory = EncoderFactory(self.hypers)
        pass

    def get_data_sampler(dataset: Dataset, random: bool) -> BatchSampler:
        if random:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        return BatchSampler(sampler, self.hypers["batch_size"], drop_last=True)
