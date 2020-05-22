import numpy as np
from typing import *


from astlib.tree import Tree


def remove_csn_bpe(tokens_with_bpe):
    tokens = []
    append_to_last = False

    def add(current_token):
        if append_to_last:
            tokens[-1] += f"@@ {current_token}"
        else:
            tokens.append(current_token)

    for token in tokens_with_bpe:
        current_token = token if isinstance(token, str) else token[0]

        if current_token == "__pad":
            continue

        if current_token == "__sow":
            # current_token = ""
            add(current_token)
            append_to_last = True
        elif current_token == "__eow":
            add(current_token)
            append_to_last = False

        else:
            add(current_token)
    return tokens


def apply_bpe_to_descendants(bpe_tokens, descendants: Iterable[int]) -> List[int]:
    """ not very performant, but it works

    ."""

    bpe_descendants = np.zeros(len(bpe_tokens), dtype=np.int)
    stack_open_ancestors = []
    stack_open_descendants = []
    bpe_token_idx: int = 0

    def _add(node_descendants, is_new_node: bool):
        nonlocal stack_open_ancestors, stack_open_descendants, bpe_token_idx

#        if bpe_token_idx >= len(bpe_descendants):
#            return

        bpe_descendants[bpe_token_idx] = node_descendants

        # if we have a new node we increment the descendants of its ancestors
        if is_new_node:
            bpe_descendants[stack_open_ancestors] += 1

        # otherwise push the node to the stack
        else:
            stack_open_ancestors.append(bpe_token_idx)
            stack_open_descendants.append(node_descendants + 1)

            # and reduce the stack (removes nodes without descendants)

            # FIXME performance
            #  maybe make stack_open_descendants and stack_open_ancestor
            #  numpy array somehow so we dont need a list comprehension here
            stack_open_descendants = [
                d - 1 for d in stack_open_descendants if (d - 1) > 0
            ]
            stack_open_ancestors = stack_open_ancestors[: len(stack_open_descendants)]

        # increment bpe counter
        bpe_token_idx += 1
        if bpe_token_idx >= len(bpe_tokens):
            bpe_descendants[stack_open_ancestors] - stack_open_descendants
            return True
        return False

    # walk through the original_tokens (and keep a matching index for the bpe tokens)
    for original_idx, original_descendants in enumerate(descendants):
#        if bpe_token_idx >= len(bpe_tokens):
#            from IPython import embed; embed()
#            break
        current_token = bpe_tokens[bpe_token_idx]

        # the start of x bpe tokens, which previously were a single token
        if current_token == "__sow":
            if original_descendants != 0:
                from IPython import embed; embed()
            assert (
                original_descendants == 0
            ), "trying to bpe'ize a token with descendants"

            # add tokens before __EOW as new tokens (by incrementing the descendants
            # of its parents by 1)
            while current_token != "__eow":
                # TODO skipping truncated bpe tokens here currently
                is_done = _add(node_descendants=0, is_new_node=True)
                if is_done:
                    assert bpe_descendants[0] < len(bpe_descendants)
                    return bpe_descendants
#                if bpe_token_idx + 1 >= len(bpe_tokens):
#                    bpe_descendants[stack_open_ancestors] - stack_open_descendants
#                    current_token = "__eow"
#                    break
                current_token = bpe_tokens[
                    bpe_token_idx
                ]  # bpe_token_idx is updated inside _add

        # _eow is the only "known" token (replaces) the original token),
        # thus we can add it as any other token.
        is_done = _add(original_descendants, is_new_node=False)
        if is_done:
            assert bpe_descendants[0] < len(bpe_descendants)
            return bpe_descendants

        assert len(stack_open_ancestors) == len(stack_open_descendants)

    assert len(bpe_descendants) == len(bpe_tokens)
    return bpe_descendants


"""
samples = np.load('out.npy', allow_pickle=True)

# for sample in samples:
sample = samples[0]

tokens_with_bpe = [t if isinstance(t, str) else t[0] for t in sample["tokens_with_bpe"]]
tokens_with_bpe_without_padding = [t for t in tokens_with_bpe if t != '__pad']
descendants = sample["descendants"]
token_ids = sample["token_ids"]
tokens = remove_csn_bpe(sample["tokens_with_bpe"])

print(len(tokens), tokens)
print(len(descendants), descendants)
print(len(tokens_with_bpe), tokens_with_bpe)

# create and pprint the original tree
t = Tree.from_tokens_and_descendants(tokens, descendants=descendants, restore_wrapped_is_terminal=True)
t.init_positions()
t.pprint(print_position=True)

# apply bpe to parent array
bpe_descendants = apply_bpe_to_descendants(tokens_with_bpe, descendants)

# and recreate the tree with BPE
# we can't create a tree with padding tokens (no valid tree)
new_tree = Tree.from_tokens_and_descendants(tokens_with_bpe_without_padding, bpe_descendants[:len(tokens_with_bpe_without_padding)])
new_tree.init_positions()
new_tree.pprint()

# we can use this array as src_descendants for relative tree transformer
print("BPE'ized descendants:")
print(len(bpe_descendants), bpe_descendants)
"""
