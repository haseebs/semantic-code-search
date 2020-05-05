import numpy as np
from scipy.spatial.distance import cdist
from more_itertools import ichunked
from annoy import AnnoyIndex
from codesearchnet.encoder import Category
from codesearchnet.evaluator import Evaluator
from codesearchnet.sample_dataset import SampleDataset
from tqdm import tqdm

N_TREES = 10
K = 100
BATCH_SIZE = 1000


class SearchNew:
    def __init__(self, code_encoder, doc_encoder):
        self.code_encoder = code_encoder
        self.doc_encoder = doc_encoder
        self.metric = "euclidean"  # or angular

    def compute_ranks(self, src_representations, tgt_representations):
        distances = cdist(src_representations, tgt_representations, metric="cosine")
        correct_elements = np.expand_dims(np.diag(distances), axis=-1)
        return np.sum(distances <= correct_elements, axis=-1), distances

    def apply(
        self,
        sample_path,
        writer,
        number_of_samples=None,
        tokenizer=None,
        directory="test",
        load_annoy="",
        annoy_save_name="",
        metric=None,
    ):
        """
        :param sample_path: path to directory of samples, for example "data/java/final/jsonl/"
        :param writer: summary writer which saves the result
        :param number_of_samples: number of code samples which will be comparable, by default all
        :param tokenizer: instance of class tokenizer, is used to evaluate on formatted tokens
        :param directory: sample_path + directory will build the path to the samples eg: test or valid
        :param load_annoy: if it is set, the annoy index will be loaded with the given file name
        :param annoy_save_name: if it is set, the annoy index will be saved in a file with this name
        :param metric: the distance function which will be used to calculate the vectors distance

        When applying a sample_path, for each sample two vector representations will be calculated by the code_encoder
        and doc_encoder. Each doc vector will be compared with all code vectors and ranks will be calculated. The
        results will be shown by the evaluator on tensor board.
        """
        if metric:
            self.metric = metric

        # read samples
        all_samples = SampleDataset(sample_path, directory, number_of_samples)
        batched_samples = ichunked(
            all_samples, BATCH_SIZE
        )  # TODO not Dataset class anymore?

        # TODO Last batch will be < BATCH_SIZE and thus will have lesser distractors
        sum_mrr = 0
        for samples in batched_samples:
            # create annoy index
            samples = list(samples)
            if len(samples) < BATCH_SIZE:
                break
            code_vector_gen = self.code_encoder.apply(samples, Category.CODE)
            annoy_size = len(next(code_vector_gen))
            code_vector_gen = self.code_encoder.apply(samples, Category.CODE)
            doc_vector_gen = self.doc_encoder.apply(samples, Category.DOC_STR)

            code_vecs = np.array([vec for vec in code_vector_gen])
            doc_vecs = np.array([vec for vec in doc_vector_gen])
            ranks, _ = self.compute_ranks(code_vecs, doc_vecs)
            mrr = np.mean(1.0 / ranks)
            sum_mrr += mrr
            print("Batch MRR: ", mrr)
        final_mrr = sum_mrr / (len(all_samples) // BATCH_SIZE)
        print("Final MRR: ", final_mrr)
