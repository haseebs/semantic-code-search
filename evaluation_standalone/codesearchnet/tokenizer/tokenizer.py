from abc import ABC, abstractmethod


class TokenizerInterface(ABC):

    @abstractmethod
    def format_tokens(self, tokens):
        pass

    @abstractmethod
    def token_intersection(self, tokens1, tokens2):
        pass
