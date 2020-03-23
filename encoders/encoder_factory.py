from .nbow_encoder import NbowEncoder
from typing import Dict, Any


class EncoderFactory:
    def __init__(self, hyperparameters: Dict[str, Any]):
        self.hypers = hyperparameters

    def get_encoder(self, encoder_type: str = "nbow_encoder"):
        if encoder_type == "nbow_encoder":
            return NbowEncoder(self.hypers)
        else:
            print(f"Encoder: {encoder_type}z is not implemented!")
            raise NotImplementedError


if __name__ == "__main__":
    from IPython import embed

    embed()
    sys.path.append("/home/haseebs/workspace/CSN/semantic-code-search")
