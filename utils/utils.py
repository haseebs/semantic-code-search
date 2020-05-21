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
        token_mask = np.array(
            [1 if token_ids[i] > 0 else 0 for i in range(len(token_ids))]
        )
        # TODO can be refactored to use the ids of bpe sow and eow tokens directly
        tokens_with_bpe = []
        for i in token_ids:
            if i in token_vocab.inverse_word_vocab:
                tokens_with_bpe.append(token_vocab.inverse_word_vocab[i])
            else:
                tokens_with_bpe.append([token_vocab.inverse_bpe_vocab[i], "@@"])
        c=0
        inside_bpe=False
        for idx, token_id in enumerate(token_ids):
            if token_id == token_vocab.bpe_vocab[token_vocab.SOW]:
                inside_bpe = True
            elif token_id == token_vocab.bpe_vocab[token_vocab.EOW]:
                inside_bpe = False
            #print('ye-',token_id)
            if not inside_bpe:
                #print(token_sequence[c])
                assert(tree_sequence[c] == 0, 'Split tokens shouldnt have children')
                c+=1


        tokens_with_bpe = [t if isinstance(t, str) else t[0] for t in tokens_with_bpe]
        # tokens_with_bpe_without_padding = [t for t in tokens_with_bpe if t != '__pad']
        # tokens = remove_csn_bpe(tokens_with_bpe)

        # create and pprint the original tree
        # t = Tree.from_tokens_and_descendants(tokens, descendants=token_descendants, restore_wrapped_is_terminal=True)
        # t.init_positions()
        # t.pprint(print_position=True)

        # apply bpe to parent array
        bpe_descendants = apply_bpe_to_descendants(tokens_with_bpe, token_descendants)

        # and recreate the tree with BPE
        # we can't create a tree with padding tokens (no valid tree)
        # new_tree = Tree.from_tokens_and_descendants(tokens_with_bpe_without_padding, bpe_descendants[:len(tokens_with_bpe_without_padding)])
        # new_tree.init_positions()
        # new_tree.pprint()

        # we can use this array as src_descendants for relative tree transformer
        # print("BPE'ized descendants:")
        # print(len(bpe_descendants), bpe_descendants)

        return token_ids, token_mask, bpe_descendants

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

    return token_ids, token_mask, tokens_with_bpe


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
        token_mask = np.array(
            [1 if token_ids[i] > 0 else 0 for i in range(len(token_ids))]
        )
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
