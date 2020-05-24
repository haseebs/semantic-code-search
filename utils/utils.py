from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
from astlib.tree import Tree

from dpu_utils.mlutils import Vocabulary
from .bpevocabulary import BpeVocabulary
from .apply_bpe_to_descendants import apply_bpe_to_descendants, remove_csn_bpe


def convert_and_pad_tree_sequence(
    token_vocab: Union[Vocabulary, BpeVocabulary],
    token_sequence: List[str],
    token_descendants: List[int],
    output_tensor_size: int,
    pad_from_left: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tensorise token sequence with padding; returning a mask for used elements as well.

    Args:
        token_vocab: Vocabulary or BPE encoder to use. We assume that token_vocab[0] is the padding symbol.
        token_sequence: List of tokens in string form
        output_tensor_size: Size of the resulting tensor (i.e., length up which we pad / down to which we truncate.
        pad_from_left: Indicate if we are padding/truncating on the left side of string. [Default: False]

    Returns:
        Pair of numpy arrays. First is the actual tensorised token sequence, the second is a masking tensor
        that is 1.0 for those token indices that are actually used.
    """

    if isinstance(token_vocab, BpeVocabulary):
        token_ids = np.array(
            list(
                token_vocab.transform([token_sequence], fixed_length=output_tensor_size)
            )[0]
        )
        token_mask = np.int_(token_ids > 0)  # fixme why is this int?

        # apply bpe to parent array
        bpe_descendants = apply_bpe_to_descendants(
            token_ids,
            token_descendants,
            sow_idx=token_vocab.bpe_vocab[token_vocab.SOW],
            eow_idx=token_vocab.bpe_vocab[token_vocab.EOW],
        )
        # adding pad index to base
        bpe_descendants[token_mask == 1] += token_vocab.word_vocab[token_vocab.PAD] + 1
        return token_ids, token_mask, bpe_descendants
    else:
        raise NotImplementedError


def convert_and_pad_token_sequence(
    token_vocab: Union[Vocabulary, BpeVocabulary],
    token_sequence: List[str],
    output_tensor_size: int,
    pad_from_left: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tensorise token sequence with padding; returning a mask for used elements as well.

    Args:
        token_vocab: Vocabulary or BPE encoder to use. We assume that token_vocab[0] is the padding symbol.
        token_sequence: List of tokens in string form
        output_tensor_size: Size of the resulting tensor (i.e., length up which we pad / down to which we truncate.
        pad_from_left: Indicate if we are padding/truncating on the left side of string. [Default: False]

    Returns:
        Pair of numpy arrays. First is the actual tensorised token sequence, the second is a masking tensor
        that is 1.0 for those token indices that are actually used.
    """
    if isinstance(token_vocab, BpeVocabulary):
        token_ids = np.array(
            list(
                token_vocab.transform([token_sequence], fixed_length=output_tensor_size)
            )[0]
        )
        # TODO
        token_mask = np.int_(token_ids > 0)

        return token_ids, token_mask

    if pad_from_left:
        token_sequence = token_sequence[-output_tensor_size:]
    else:
        token_sequence = token_sequence[:output_tensor_size]

    sequence_length = len(token_sequence)
    if pad_from_left:
        start_idx = output_tensor_size - sequence_length
    else:
        start_idx = 0

    token_ids = np.zeros(output_tensor_size, dtype=np.int32)
    token_mask = np.zeros(output_tensor_size, dtype=np.float32)
    for i, token in enumerate(token_sequence, start=start_idx):
        token_ids[i] = token_vocab.get_id_or_unk(token)
        token_mask[i] = True

    return token_ids, token_mask
