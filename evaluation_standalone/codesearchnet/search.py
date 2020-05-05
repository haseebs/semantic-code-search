from more_itertools import ichunked
from annoy import AnnoyIndex
from codesearchnet.encoder import Category
from codesearchnet.evaluator import Evaluator
from codesearchnet.sample_dataset import SampleDataset
from tqdm import tqdm

N_TREES = 10
K = 100
BATCH_SIZE = 1000


class Search:
    def __init__(self, code_encoder, doc_encoder):
        self.code_encoder = code_encoder
        self.doc_encoder = doc_encoder
        self.metric = "angular"  # or angular

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
        evaluator = None if not writer else Evaluator(writer, metric, tokenizer)

        # read samples
        all_samples = SampleDataset(sample_path, directory, number_of_samples)
        batched_samples = ichunked(
            all_samples, BATCH_SIZE
        )  # TODO not Dataset class anymore?

        # TODO Last batch will be < BATCH_SIZE and thus will have lesser distractors
        rank_list = []
        for samples in batched_samples:
            # create annoy index
            samples = list(samples)
            code_vector_gen = self.code_encoder.apply(samples, Category.CODE)
            annoy_size = len(next(code_vector_gen))
            code_vector_gen = self.code_encoder.apply(samples, Category.CODE)
            doc_vector_gen = self.doc_encoder.apply(samples, Category.DOC_STR)

            annoy = self.build_annoy_index(
                code_vector_gen, annoy_size, len(samples), load_annoy, annoy_save_name
            )

            # calculate ranks for each doc_vector
            for index, doc_vector in tqdm(
                enumerate(doc_vector_gen),
                desc=f"Calculating distances",
                total=len(samples),
            ):
                k_indices = annoy.get_nns_by_vector(doc_vector, K)
                rank_dict = {rank: i for rank, i in enumerate(k_indices, 1)}
                rank_list.append(rank_dict)
                if evaluator:
                    evaluator.add_sample(index, rank_dict, samples)
        evaluator.get_final_results()
        return rank_list

    def build_annoy_index(
        self, vector_generator, vector_size, samples_size, load_annoy, save_name
    ):
        index = AnnoyIndex(vector_size, self.metric)
        if load_annoy:
            index.load(load_annoy)
        else:
            for i, code_vector in tqdm(
                enumerate(vector_generator),
                desc=f"Building annoy index",
                total=samples_size,
            ):
                index.add_item(i, code_vector)
            index.build(N_TREES)

        if save_name:
            index.save(save_name)

        return index
