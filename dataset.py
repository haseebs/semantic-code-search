import os
import gzip
import json
import random
import numpy as np
from typing import List, Dict, Any, Iterable
from torch.utils.data import Dataset

from encoders.encoder_base import EncoderBase
from utils.utils import convert_and_pad_token_sequence
from utils.utils import convert_and_pad_tree_sequence


class CSNDataset(Dataset):
    def __init__(
        self, hparams: Dict[str, Any], keep_keys: set(), data_split: str = "train"
    ):
        self.hparams = hparams
        self.keep_keys = keep_keys
        self.original_data = []
        self.encoded_data = []
        self.data_split = data_split
        self.read_data(data_split)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

    def read_jsonl(self, path: str) -> Iterable[Dict[str, Any]]:
        jsonl_file = gzip.open(path, mode="rt", encoding="utf-8")
        for line in jsonl_file:
            yield {k: v for k, v in json.loads(line).items() if k in self.keep_keys}

    def read_data(self, data_split: str = "train") -> None:
        # data_dirs = open("data_dirs.txt", "rt", encoding="utf-8")
        paths = [os.path.join(path, data_split) for path in self.hparams["data_dirs"]]
        for path in paths:
            data_files = sorted(os.listdir(path))
            for data_file in data_files:
                if data_file.endswith(".jsonl.gz"):
                    self.original_data.extend(
                        self.read_jsonl(path=os.path.join(path, data_file))
                    )

    def encode_data(
        self, query_encoder: EncoderBase, code_encoder: EncoderBase
    ) -> None:
        # TODO may need to move to encoder class to handle encoders that come with their own tokenizers
        count_empty_code = 0
        for idx, sample in enumerate(self.original_data):

            enc_query, enc_query_mask = convert_and_pad_token_sequence(
                query_encoder.vocabulary,
                [t.lower() for t in sample["docstring_tokens"]],
                self.hparams["query_max_num_tokens"],
            )

            if self.hparams["code_encoder_type"] == "tree_attention_encoder":
                (
                    enc_code,
                    enc_code_mask,
                    code_ast_descendants,
                ) = convert_and_pad_tree_sequence(
                    code_encoder.vocabulary,
                    sample["code_ast_tokens"],
                    sample["code_ast_descendants"],
                    self.hparams["code_max_num_tokens"],
                )
            else:
                enc_code, enc_code_mask = convert_and_pad_token_sequence(
                    code_encoder.vocabulary,
                    sample["code_tokens"],
                    self.hparams["code_max_num_tokens"],
                )

            enc_query_length = int(np.sum(enc_query_mask))
            enc_code_length = int(np.sum(enc_code_mask))
            # TODO what to do about empty stuff
            if not (enc_code_length > 0):
                count_empty_code += 1
                continue
            assert enc_query_length > 0  # and enc_code_length > 0

            encoded_data_item = {
                "original_data_idx": idx,
                "encoded_query": enc_query,
                "encoded_query_mask": enc_query_mask,
                "encoded_query_length": enc_query_length,
                "encoded_code": enc_code,
                "encoded_code_mask": enc_code_mask,
                "encoded_code_length": enc_code_length,
            }
            if self.hparams["code_encoder_type"] == "tree_attention_encoder":
                encoded_data_item["code_ast_descendants"] = code_ast_descendants
            self.encoded_data.append(encoded_data_item)
        print("Samples rejected due to no AST built: ", count_empty_code)
        if self.data_split in ["valid", "test"]:
            random.shuffle(self.encoded_data)


if __name__ == "__main__":
    dataset = CSNDataset()
    encoder_code = EncoderBase(
        {
            "use_bpe": True,
            "vocab_size": 10000,
            "vocab_pct_bpe": 0.5,
            "vocab_count_threshold": 10,
        }
    )
    encoder_query = EncoderBase(
        {
            "use_bpe": True,
            "vocab_size": 10000,
            "vocab_pct_bpe": 0.5,
            "vocab_count_threshold": 10,
        }
    )

    for sample in dataset:
        encoder_code.update_tokens_from_sample(sample["code_tokens"])
        encoder_query.update_tokens_from_sample(
            [t.lower() for t in sample["docstring_tokens"]]
        )
    encoder_code.build_vocabulary()
    encoder_query.build_vocabulary()
