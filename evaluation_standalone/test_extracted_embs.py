import codesearchnet
import pickle
import torch
import numpy as np
from tensorboardX import SummaryWriter

SAMPLE_PATH = "/home/haseebs/workspace/CSN/CodeSearchNet_official/resources/data_extract/ruby/final/jsonl"

class Encoder(codesearchnet.EncoderInterface):
    def __init__(self):
        self.loaded_embs = pickle.load(open("codesearchnet/test/my_emb_ruby.pickle", "rb"))
        #self.loaded_embs = np.load("codesearchnet/test/embs_dict.npy")
    def apply(self, samples, category):
        for sample in samples:
            yield self.apply_one(sample, category)

    def apply_one(self, sample, category):
        """
        :param sample: one sample as dictionary
        :param category: either Category.CODE or Category.DOC_STR
        """
        if category == codesearchnet.Category.DOC_STR:
            return self.loaded_embs[sample['url']][1]
        elif category == codesearchnet.Category.CODE:
            return self.loaded_embs[sample['url']][0]
        else:
            raise TypeError('category should be a value of codesearchnet.Category')


encoder = Encoder()
writer = SummaryWriter()

search = codesearchnet.SearchNew(encoder, encoder)
search.apply(SAMPLE_PATH, writer)
