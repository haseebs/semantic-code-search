import os
import gzip
import json
import random
import numpy as np
from typing import List, Dict, Any, Iterable
from torch import nn
from torch.utils.data import Dataset
from elasticsearch import Elasticsearch
from tqdm import tqdm
import wandb

from encoders.encoder_base import EncoderBase
from utils.utils import convert_and_pad_token_sequence
from utils.utils import convert_and_pad_tree_sequence


class CSNDataset(Dataset):
    def __init__(
        self,
        hparams: Dict[str, Any],
        keep_keys: set(),
        data_split: str = "train",
        languages: List[str] = [],
        logger = None,
    ):
        self.hparams = hparams
        self.keep_keys = keep_keys
        self.languages = languages
        self.original_data = []
        self.encoded_data = []
        self.data_split = data_split
        self.logger = logger
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
            if not path.split("/")[-2] in self.languages:  # TODO fix hardcoded path idx
                continue
            data_files = sorted(os.listdir(path))
            for data_file in data_files:
                if data_file.endswith(".jsonl.gz"):
                    self.original_data.extend(
                        self.read_jsonl(path=os.path.join(path, data_file))
                    )
                    # if data_split in ["train", "valid"]:
                    #    break

    def build_es_index(self) -> None:
        if not self.hparams["use_elasticsearch"]:
            return
        print("Building Elasticsearch Index...")
        self.es = Elasticsearch()
        for idx, data in tqdm(
            enumerate(self.original_data), total=len(self.original_data)
        ):
            self.es.index(index="code_index", body={"idx": idx, "code": data["code"]})

    def encode_data(self, query_encoder: nn.Module, code_encoder: nn.Module) -> None:
        # TODO may need to move to encoder class to handle encoders that come with their own tokenizers
        count_empty_code = 0
        count_empty_docstring = 0
        count_empty_neg = 0
        examples_table = []
        examples_columns = [
            "language",
            "query",
            "code_pos",
            "code_neg",
            "query_neg",
            "score_neg",
        ]

        if self.data_split == "train":
            self.build_es_index()

        for idx, sample in tqdm(
            enumerate(self.original_data), total=len(self.original_data)
        ):

            if self.hparams["query_encoder_type"] == "pretrained_roberta_encoder":
                enc_query = query_encoder.tokenizer.encode(
                    [t.lower() for t in sample[self.hparams["key_docstring_tokens"]]],
                    is_pretokenized=True,
                    padding="max_length",
                    truncation=True,
                    max_length=self.hparams["query_max_num_tokens"],
                )
                enc_query = enc_query[: self.hparams["query_max_num_tokens"] - 1]
                if enc_query[-1] != query_encoder.tokenizer.eos_token_id:
                    enc_query.append(query_encoder.tokenizer.eos_token_id)
                while len(enc_query) < self.hparams["query_max_num_tokens"]:
                    enc_query.append(query_encoder.tokenizer.pad_token_id)
                enc_query = np.array(enc_query)
                enc_query_mask = np.int_(
                    enc_query != query_encoder.tokenizer.pad_token_id
                )

            else:
                enc_query, enc_query_mask = convert_and_pad_token_sequence(
                    query_encoder.vocabulary,
                    [t.lower() for t in sample[self.hparams["key_docstring_tokens"]]],
                    self.hparams["query_max_num_tokens"],
                )

            if self.hparams["code_encoder_type"] == "pretrained_roberta_encoder":
                raise NotImplementedError

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
            if enc_code_length == 0:
                count_empty_code += 1
                continue
            if enc_query_length == 0:  # and enc_code_length > 0
                count_empty_docstring += 1
                continue

            encoded_data_item = {
                "original_data_idx": idx,
                "language": sample["language"],
                "encoded_query": enc_query,
                "encoded_query_mask": enc_query_mask,
                "encoded_query_length": enc_query_length,
                "encoded_code": enc_code,
                "encoded_code_mask": enc_code_mask,
                "encoded_code_length": enc_code_length,
            }
            if self.hparams["code_encoder_type"] == "tree_attention_encoder":
                encoded_data_item["code_ast_descendants"] = code_ast_descendants

            if self.hparams["use_elasticsearch"] and self.data_split == "train":
                res = self.es.search(
                    index="code_index",
                    body={
                        "query": {
                            "more_like_this": {
                                "fields": ["code"],
                                "like": sample["docstring"],
                                "min_term_freq": 1,
                                "max_query_terms": 30,
                            }
                        }
                    },
                )
                code_tokens_neg = None
                for hit in res["hits"]["hits"]:
                    if hit["_source"]["idx"] != idx:
                        code_tokens_neg = self.original_data[hit["_source"]["idx"]][
                            "code_tokens"
                        ]
                        break
                if code_tokens_neg != None:
                    neg_sample = self.original_data[hit["_source"]["idx"]]
                    markdown_code_pos = (
                        "```%s\n" % sample["language"]
                        + sample["code"].strip("\n")
                        + "\n```"
                    )

                    markdown_code_neg = (
                        "```%s\n" % neg_sample["language"]
                        + neg_sample["code"].strip("\n")
                        + "\n```"
                    )
                    examples_table.append(
                        [
                            sample["language"],
                            sample["docstring"],
                            markdown_code_pos,
                            markdown_code_neg,
                            neg_sample["docstring"],
                            hit["_score"],
                        ]
                    )
                if code_tokens_neg == None:
                    count_empty_neg += 1
                    code_tokens_neg = self.original_data[
                        random.sample(range(0, len(self.original_data)), 1)[0]
                    ]["code_tokens"]

                enc_code_neg, enc_code_mask_neg = convert_and_pad_token_sequence(
                    code_encoder.vocabulary,
                    code_tokens_neg,
                    self.hparams["code_max_num_tokens"],
                )

                enc_code_length_neg = int(np.sum(enc_code_mask_neg))
                encoded_data_item.update(
                    {
                        "encoded_code_neg": enc_code_neg,
                        "encoded_code_mask_neg": enc_code_mask_neg,
                        "encoded_code_length_neg": enc_code_length_neg,
                    }
                )

            self.encoded_data.append(encoded_data_item)
        print(
            "Samples rejected due to no AST built: ",
            count_empty_code,
            "; empty docstrings:",
            count_empty_docstring,
            "; empty negatives:",
            count_empty_neg,
        )

        from IPython import embed

        embed()
        if self.hparams["use_elasticsearch"]:
            self.logger.experiment.log(
                {
                    "Elasticsearch Examples": wandb.Table(
                        columns=examples_columns, rows=examples_table
                    )
                }
            )

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
