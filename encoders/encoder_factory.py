from typing import Dict, Any
from .nbow_encoder import NbowEncoder
from .self_attention_encoder import SelfAttentionEncoder
from .tree_attention_encoder import TreeAttentionEncoder


class EncoderFactory:
    def __init__(self, hyperparameters: Dict[str, Any]):
        self.hypers = hyperparameters

    def get_encoder(self, encoder_type: str = "nbow_encoder"):
        if encoder_type == "nbow_encoder":
            return NbowEncoder(
                vocab_size=self.hypers["vocab_size"],
                embedding_dim=self.hypers["embedding_dim"],
                dropout=self.hypers["dropout_prob"],
                vocab_count_threshold=self.hypers["vocab_count_threshold"],
                use_bpe=self.hypers["use_bpe"],
                vocab_pct_bpe=self.hypers["vocab_pct_bpe"],
            )
        elif encoder_type == "self_attention_encoder":
            return SelfAttentionEncoder(
                ntoken=self.hypers["vocab_size"],
                ninp=self.hypers["embedding_dim"],
                nhead=self.hypers["self_attention_nheads"],
                nhid=self.hypers["self_attention_nhid"],
                nlayers=self.hypers["self_attention_nlayers"],
                dropout=self.hypers["dropout_prob"],
                vocab_count_threshold=self.hypers["vocab_count_threshold"],
                use_bpe=self.hypers["use_bpe"],
                vocab_pct_bpe=self.hypers["vocab_pct_bpe"],
            )
        elif encoder_type == "tree_attention_encoder":
            return TreeAttentionEncoder(
                ntoken=self.hypers["vocab_size"],
                ninp=self.hypers["embedding_dim"],
                nhead=self.hypers["self_attention_nheads"],
                nhid=self.hypers["self_attention_nhid"],
                nlayers=self.hypers["self_attention_nlayers"],
                dropout=self.hypers["dropout_prob"],
                vocab_count_threshold=self.hypers["vocab_count_threshold"],
                use_bpe=self.hypers["use_bpe"],
                vocab_pct_bpe=self.hypers["vocab_pct_bpe"],
                clamping_distance=self.hypers["clamping_distance"],
                use_sinusoidal_positional_embeddings=self.hypers["tree_transformer_use_positional_embeddings"],
            )
        else:
            print(f"Encoder: {encoder_type}z is not implemented!")
            raise NotImplementedError


if __name__ == "__main__":
    from IPython import embed

    embed()
    sys.path.append("/home/haseebs/workspace/CSN/semantic-code-search")
