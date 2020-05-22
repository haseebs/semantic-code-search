from torch import nn
from typing import Dict, Any, List
from collections import Counter

from utils.bpevocabulary import BpeVocabulary


class EncoderBase(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        vocab_count_threshold: int,
        use_bpe: bool,
        vocab_pct_bpe: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.vocab_count_threshold = vocab_count_threshold
        self.use_bpe = use_bpe
        self.vocab_pct_bpe = vocab_pct_bpe
        self.token_counter = Counter()
        self.vocabulary = None

    def update_tokens_from_sample(self, sample_tokens: List[str]) -> None:

        self.token_counter.update(sample_tokens)

    def build_vocabulary(self) -> None:
        if self.vocabulary != None:
            return
        if self.use_bpe:
            required_tokens = [
                t for t in self.token_counter if t.startswith("<[") and t.endswith("]>")
            ]
            self.vocabulary = BpeVocabulary(
                vocab_size=self.vocab_size,
                pct_bpe=self.vocab_pct_bpe,
                required_tokens=required_tokens,
            )
            self.vocabulary.fit(self.token_counter)
        else:
            self.vocabulary = Vocabulary.create_vocabulary(
                tokens=self.token_counter,
                max_size=self.vocab_size,
                count_threshold=self.vocab_count_threshold,
            )
