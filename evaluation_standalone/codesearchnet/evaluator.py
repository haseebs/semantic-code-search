from codesearchnet.tokenizer.simple_tokenizer import SimpleTokenizer
import numpy as np

NUMBER_OF_OUTPUT_SAMPLES = 100


class Evaluator:
    def __init__(self, writer, metric, tokenizer):
        self.writer = writer
        self.writer.add_text("Metric", "euclid")
        self.tokenizer = tokenizer if tokenizer else SimpleTokenizer("", "")
        self.original_ranks = []
        self.results = []

    def add_sample(self, index, rank_dict, samples):
        doc_sample = samples[index]
        bad_rank = 1000  # len(samples) #TODO

        original_code_rank = self.get_original_rank(rank_dict, index, bad_rank)
        self.original_ranks.append(original_code_rank)

        if index < NUMBER_OF_OUTPUT_SAMPLES:
            original_code_rank = (
                original_code_rank if original_code_rank != bad_rank else "very bad"
            )
            matching_tokens = self.tokenizer.token_intersection(
                doc_sample["docstring_tokens"], doc_sample["code_tokens"]
            )

            result = f"#{index}. Function is ranked {original_code_rank}\n"
            result += (
                f"- Docstring tokens: '{' '.join(doc_sample['docstring_tokens'])}'\n"
            )
            result += f"- Code tokens: '{' '.join(doc_sample['code_tokens'])}'\n"
            result += f"- Matching tokens ({len(matching_tokens)}): '{' '.join(matching_tokens)}'\n\n\n"

            for rank in range(1, 4):
                if rank in rank_dict:
                    code_sample = samples[rank_dict[rank]]
                    matching_tokens = self.tokenizer.token_intersection(
                        doc_sample["docstring_tokens"], code_sample["code_tokens"]
                    )

                    result += f"###{rank}. Rank: {rank_dict[rank]}. Function\n"
                    result += f"- matching tokens ({len(matching_tokens)}): '{' '.join(matching_tokens)}'\n"
                    result += (
                        f"- all code tokens: '{' '.join(code_sample['code_tokens'])}'\n"
                    )
            self.results.append(result)
            if index == NUMBER_OF_OUTPUT_SAMPLES - 1:
                self.writer.add_text("Results", "\n".join(self.results))
                self.results = []

        self.writer.add_scalar("Mean rank", self.mean_rank(), global_step=index)
        self.writer.add_scalar("Mean reciprocal rank", self.mrr(), global_step=index)

        if index == len(samples) - 1:
            self.writer.add_text("Hits", self.hits([3, 5, 10]))
            self.writer.add_text(
                "Mean rank",
                f"{str(self.mean_rank())} of {len(self.original_ranks)} samples",
            )
            self.writer.add_text("Mean reciprocal rank", "%2.2f" % self.mrr())

    def get_final_results(self):
        from IPython import embed

        embed()
        print("MRR final: ", self.mrr())

    def hits(self, of):
        result = ""
        ranks = np.array(self.original_ranks)
        for i in of:
            p = len(ranks[ranks <= i]) * 100.0 / len(ranks)
            result += "- %2.2f%% were ranked under the top %d.\n" % (p, i)
        return result

    def mean_rank(self):
        return int(round(sum(self.original_ranks) / len(self.original_ranks)))

    def mrr(self):
        return sum([1 / rank for rank in self.original_ranks]) / len(
            self.original_ranks
        )

    @staticmethod
    def get_original_rank(ranks, i, bad_rank):
        """
        :param ranks: dictionary of ranks
        :param i: index of doc vector
        :param bad_rank: rank which will be used if rank is out of range
        """
        if i in list(ranks.values()):
            return list(ranks.keys())[list(ranks.values()).index(i)]
        return bad_rank
