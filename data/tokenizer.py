from __future__ import annotations

import pickle
from collections.abc import Iterable

import regex as re

_PRETOKEN_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
_BYTE_TO_BYTES = [bytes([i]) for i in range(256)]


def init_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)


def _process_chunk(text: str, special_tokens_set: set[str] | None) -> list[list[bytes]]:
    if special_tokens_set and text in special_tokens_set:
        return [[text.encode("utf-8")]]
    pattern = _PRETOKEN_PATTERN
    byte_table = _BYTE_TO_BYTES
    result = []
    for m in pattern.finditer(text):
        encoded_bytes = m.group(0).encode("utf-8")
        result.append([byte_table[b] for b in encoded_bytes])
    return result


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.reverse_vocab = {v: k for k, v in vocab.items()}  # dict[bytes, int]
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}  # dict[tuple[bytes, bytes], int]
        if special_tokens:
            # Sort by length (longest first) to handle overlapping tokens
            sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
            special_tokens_pat = "|".join([re.escape(s) for s in sorted_special_tokens])
            self._special_tokens_re = re.compile(f"({special_tokens_pat})")
            self._special_tokens_set = set(special_tokens)
        else:
            self._special_tokens_re = None
            self._special_tokens_set = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """Loads a tokenizer from a file."""
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def pretokenize_text(self, text: str) -> list[list[bytes]]:
        if self._special_tokens_re:
            text_parts = self._special_tokens_re.split(text)
        else:
            text_parts = [text]
        pretokens = []
        for part in text_parts:
            if not part:
                continue
            pretokens.extend(_process_chunk(text=part, special_tokens_set=self._special_tokens_set))
        return pretokens

    """
    def merge_single_pretoken(self, pretoken: list[bytes]) -> list[bytes]:
        for merge in self.merges:
            new_pretoken = []
            i = 0
            while i < len(pretoken):
                if i < len(pretoken) - 1 and (pretoken[i], pretoken[i+1]) == merge:
                    new_pretoken.append(pretoken[i] + pretoken[i+1])
                    i += 2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1
            pretoken = new_pretoken
            if len(pretoken) == 1:
                break
        return pretoken
    """

    def find_best_merge(self, pretoken: list[bytes]) -> tuple[int | None, int | None]:
        best_rank, merge_idx = None, None
        for i in range(len(pretoken) - 1):
            candidate_pair = (pretoken[i], pretoken[i + 1])
            rank = self.merge_ranks.get(candidate_pair)
            if rank is None:
                continue
            if best_rank is None or rank < best_rank:
                best_rank = rank
                merge_idx = i
        return best_rank, merge_idx

    def merge_single_pretoken(self, pretoken: list[bytes]) -> list[bytes]:
        while True:
            _, merge_idx = self.find_best_merge(pretoken)
            if merge_idx is None:
                break
            merged = pretoken[merge_idx] + pretoken[merge_idx + 1]
            pretoken = pretoken[:merge_idx] + [merged] + pretoken[merge_idx + 2 :]
        return pretoken

    def merge_pretokens(self, pretokens: list[list[bytes]]) -> list[bytes]:
        final_list = []
        for pretoken in pretokens:
            final_list.extend(self.merge_single_pretoken(pretoken))
        return final_list

    def convert_to_token_ids(self, token_list: list[bytes]) -> list[int]:
        return [self.reverse_vocab[t] for t in token_list]

    def encode(self, text: str) -> list[int]:
        pretokens = self.pretokenize_text(text)
        token_list = self.merge_pretokens(pretokens)
        token_ids = self.convert_to_token_ids(token_list)
        return token_ids

    def decode(self, ids: list[int]) -> str:
        output = b"".join(self.vocab[id] for id in ids)
        return output.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]):
        """Encode an iterable of strings, yielding token IDs one at a time."""
        for item in iterable:
            yield from self.encode(item)
