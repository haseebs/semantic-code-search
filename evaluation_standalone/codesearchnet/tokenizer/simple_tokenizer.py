import os
import re

from codesearchnet.tokenizer.tokenizer import TokenizerInterface


class SimpleTokenizer(TokenizerInterface):

    def __init__(self, stop_words_path, split_words_path):
        """
        :param stop_words_path: path to txt file, which contains in each line one stop word
        :param split_words_path: path to txt file, which contains in each line a split word
        """
        self.stop_words = []
        self.split_words = []
        if os.path.isfile(stop_words_path):
            with open(stop_words_path, 'r') as f:
                self.stop_words = f.read().splitlines()
        if os.path.isfile(split_words_path):
            with open(split_words_path, 'r') as f:
                self.split_words = f.read().splitlines()

    def format_tokens(self, tokens):
        """
        :param tokens: list of unformatted tokens, eg: ["camelCase", "under_line"]
        :return: a new list of formatted tokens, eg: ["camel", case", "under", "line"]

        creates a new list of tokens, which contain all split tokens of the sample list
        """
        sample = self.split_remove_sep(" ", tokens)
        for sep in self.split_words:
            sample = self.split_keep_sep(sep, sample)
        sample = self.split_camelcase(sample)
        return sample

    def token_intersection(self, tokens1, tokens2):
        """
        formats tokens and returns tokens which appear in both lists
        """
        tokens1 = self.format_tokens(tokens1)
        tokens2 = self.format_tokens(tokens2)

        return list(set(tokens1).intersection(tokens2))

    def remove_stopwords(self, tokens):
        """
        :param tokens: an array of tokens
        :return: new array of tokens without stopwords

        creates a new array without stopwords of the given stopwords file
        """
        if self.stop_words:
            new_tokens = []
            for token in tokens:
                if token not in self.stop_words:
                    new_tokens.append(token)
            return new_tokens
        return tokens

    @staticmethod
    def split_remove_sep(sep, sample):
        new_sample = []
        for token in sample:
            splits = token.split(sep)
            for new_token in splits:
                if len(new_token) > 0:
                    new_sample.append(new_token)
        return new_sample

    @staticmethod
    def split_keep_sep(sep, sample):
        new_sample = []
        for token in sample:
            splits = token.split(sep)
            for i in range(len(splits)):
                if len(splits[i]) > 0:
                    new_sample.append(splits[i])
                    if i != len(splits) - 1:
                        new_sample.append(sep)
                else:
                    if i != len(splits) - 1:
                        new_sample.append(sep)
        return new_sample

    @staticmethod
    def split_camelcase(sample):
        new_sample = []
        for token in sample:
            camelcase_splits = re.sub('([a-z])([A-Z])', r'\1 \2', token).split()
            for new_token in camelcase_splits:
                if len(new_token) > 0:
                    new_sample.append(new_token.lower())
        return new_sample
