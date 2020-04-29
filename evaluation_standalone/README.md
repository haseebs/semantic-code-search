## Quick Start

- create a new project for the encoder and implement the EncoderInterface 
- apply samples to the search class
- create a venv: `python3 -m venv env-csn`
- start venv: `source env-csn/bin/activate`
    - install this project as package: `pip install -e .`
- start your project (using venv)
- take look at the results: `tensorboard --logdir=runs`

## Encoder Example
```python
from tensorboardX import SummaryWriter
import torch

SAMPLE_PATH = "/video2/codesearchnet/CodeSearchNet/java/final/jsonl/"

class Encoder(codesearchnet.EncoderInterface):
    def apply(self, samples, category):
        for sample in samples:
            yield self.apply_one(sample, category)
    
    def apply_one(self, sample, category):
        """
        :param sample: one sample as dictionary
        :param category: either Category.CODE or Category.DOC_STR
        """
        if category == codesearchnet.Category.DOC_STR:
            doc_tokens = sample['docstring_tokens']
            return torch.zeros(len(doc_tokens))  # embedding for doc string
        elif category == codesearchnet.Category.CODE:
            code_tokens = sample['code_tokens']
            return torch.zeros(len(code_tokens))  # embedding for code
        else:
            raise TypeError('category should be a value of codesearchnet.Category')


encoder = Encoder()
writer = SummaryWriter()

search = codesearchnet.Search(encoder, encoder)
search.apply(SAMPLE_PATH, writer)
```

See [BOW Encoder](https://gitlab.cs.hs-rm.de/lavis/repository_mining/codesearchnet-challenge/code-search-net-challenge/blob/master/src/encoder/one_hot/bow_encoder.py) as an additional example
