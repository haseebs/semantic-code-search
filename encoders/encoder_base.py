from torch import nn
from typing import Dict, Any, List
from collections import Counter

from utils.bpevocabulary import BpeVocabulary


class EncoderBase(nn.Module):
    def __init__(self, hyperparameters: Dict[str, Any]):
        super().__init__()
        self.hypers = hyperparameters
        self.token_counter = Counter()
        self.vocabulary = None

    def update_tokens_from_sample(self, sample_tokens: List[str]) -> None:
        self.token_counter.update(sample_tokens)

    def build_vocabulary(self) -> None:
        if self.hypers["use_bpe"]:
            self.vocabulary = BpeVocabulary(
                vocab_size=self.hypers["vocab_size"],
                pct_bpe=self.hypers["vocab_pct_bpe"],
            )
            self.vocabulary.fit(self.token_counter)
        else:
            self.vocabulary = Vocabulary.create_vocabulary(
                tokens=self.token_counter,
                max_size=self.hypers["vocab_size"],
                count_threshold=self.hypers["vocab_count_threshold"],
            )
