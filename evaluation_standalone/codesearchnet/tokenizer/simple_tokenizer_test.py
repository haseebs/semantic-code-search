import unittest
import os

from codesearchnet.tokenizer.simple_tokenizer import SimpleTokenizer

STOP_WORDS = os.path.join('..', 'test', 'stopwords.txt')
SPLIT_WORDS = os.path.join('..', 'test', 'splitwords.txt')


class TestTokenizer(unittest.TestCase):

    def test_format(self):
        test = ["hallo i", "bims_1", "funnyTestYeah!", "\u5206\u949f\u7ebf\u56de\u6d4b\u7684\u65f6\u5019\u7684gap"]
        tokenizer = SimpleTokenizer(STOP_WORDS, SPLIT_WORDS)
        result = tokenizer.format_tokens(test)
        self.assertEqual("hallo", result[0])
        self.assertEqual("i", result[1])
        self.assertEqual("bims", result[2])
        self.assertEqual("_", result[3])
        self.assertEqual("1", result[4])
        self.assertEqual("funny", result[5])
        self.assertEqual("test", result[6])
        self.assertEqual("yeah", result[7])
        self.assertEqual("!", result[8])
        self.assertEqual("分钟线回测的时候的gap", result[9])
        self.assertEqual(10, len(result))

    def test_remove_stopwords(self):
        tokens = ["1", "def", "keks", "void", "return", "returns"]
        tokenizer = SimpleTokenizer(STOP_WORDS, SPLIT_WORDS)
        result = tokenizer.remove_stopwords(tokens)
        self.assertEqual("keks", result[0])
        self.assertEqual("returns", result[1])
        self.assertEqual(2, len(result))

    def test_split_remove_sep(self):
        sample = ["1_test", "_I_", "bims_"]
        tokenizer = SimpleTokenizer(STOP_WORDS, SPLIT_WORDS)
        result = tokenizer.split_remove_sep("_", sample)
        self.assertEqual("1", result[0])
        self.assertEqual("test", result[1])
        self.assertEqual("I", result[2])
        self.assertEqual("bims", result[3])
        self.assertEqual(4, len(result))

    def test_split_keep_sep(self):
        sample = ["1_test", "_I_", "bims_"]
        tokenizer = SimpleTokenizer(STOP_WORDS, SPLIT_WORDS)
        result = tokenizer.split_keep_sep("_", sample)
        self.assertEqual("1", result[0])
        self.assertEqual("_", result[1])
        self.assertEqual("test", result[2])
        self.assertEqual("_", result[3])
        self.assertEqual("I", result[4])
        self.assertEqual("_", result[5])
        self.assertEqual("bims", result[6])
        self.assertEqual("_", result[7])
        self.assertEqual(8, len(result))

    def test_split_camelcase(self):
        sample = ["testIchBims", "LaLa", "FINALVAR"]
        tokenizer = SimpleTokenizer(STOP_WORDS, SPLIT_WORDS)
        result = tokenizer.split_camelcase(sample)
        self.assertEqual("test", result[0])
        self.assertEqual("ich", result[1])
        self.assertEqual("bims", result[2])
        self.assertEqual("la", result[3])
        self.assertEqual("la", result[4])
        self.assertEqual("finalvar", result[5])
        self.assertEqual(6, len(result))


if __name__ == '__main__':
    unittest.main()
