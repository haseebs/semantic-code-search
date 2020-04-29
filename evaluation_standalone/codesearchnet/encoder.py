from abc import ABC, abstractmethod
from enum import Enum


class Category(Enum):
    DOC_STR = 'docstring'
    CODE = 'code'


class EncoderInterface(ABC):

    @abstractmethod
    def apply(self, samples, category):
        """
        :param samples: a list of samples, each sample is a dictionary, which contains among others the code of its
            method and its doc string
        :param category: could be set either to Category.CODE or to Category.DOC_STR.
            is given to use one Encoder for both cases and distinguish though the category.
        :return: a generator of vectors as representation of the code or the doc string

        When applying samples to the encoder, a generator should be returned which calculates a vector representation
        for each sample.
        """
        pass
