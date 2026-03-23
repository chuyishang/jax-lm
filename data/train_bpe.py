import os
from collections import Counter, defaultdict
from functools import partial
from heapq import heapify, heappop, heappush
from multiprocessing import Pool, cpu_count

import regex as re
from tqdm import tqdm

_PRETOKEN_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
_BYTE_TO_BYTES = [bytes([i]) for i in range(256)]
_PRETOKEN_CHUNK_BYTES = 4 * 1024 * 1024
_PRETOKEN_MIN_MP_BYTES = 8 * 1024 * 1024


def _process_chunk(text: str, special_tokens_pat: str | None = None) -> Counter[tuple[bytes]]:
    """Helper function for multiprocessing in load_and_pretokenize."""
    pretoken_counts = Counter()
    pattern = _PRETOKEN_PATTERN
    byte_table = _BYTE_TO_BYTES
    parts = [text]
    if special_tokens_pat:
        parts = re.split(special_tokens_pat, text)
    for part in parts:
        if not part:
            continue
        for m in pattern.finditer(part):
            encoded_bytes = m.group(0).encode("utf-8")
            byte_array = tuple(byte_table[b] for b in encoded_bytes)
            pretoken_counts[byte_array] += 1
    return pretoken_counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    show_progress: bool = False,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.
        show_progress (bool): Whether to display progress bars during pretokenization and training.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.

    Notes:
        - We first initialize the vocabulary. Initial vocab size = 256 since we are training a byte-level tokenizer.
        - Then we pre-tokenize the input dataset. In this step, we:
            - Use the regex pattern to split the input corpus into tokens
            - Represent this as dict[tuple[bytes], int]
            - Sum all byte-pair counts
        - Building blocks:
            - Symbol ((1,)): single byte
            - Word (tuple[bytes]): a sequence of bytes
    """

    def load_and_pretokenize(input_path: str | os.PathLike, special_tokens: list[str]) -> Counter[tuple[bytes]]:
        """
        Loads a corpus from input path and pretokenizes it.

        Args:
            input_path: Path to corpus to load and pretokenize.
            special_tokens: List of special tokens to handle during tokenization.

        Returns:
            Counter object corresponding to each pretoken and their counts.
            Each key is a tuple of bytes e.g. (b't', b'e', b's', b't')
        """

        if not show_progress:
            print("PRETOKENIZING...")

        special_tokens_pat = None
        if special_tokens:
            special_tokens_pat = "|".join([re.escape(s) for s in special_tokens])

        file_size = os.path.getsize(input_path)

        def iter_text_chunks(path: str | os.PathLike, chunk_bytes: int):
            with open(path, encoding="utf-8") as f:
                buffer = []
                buffer_len = 0
                for line in f:
                    buffer.append(line)
                    buffer_len += len(line)
                    if buffer_len >= chunk_bytes:
                        yield "".join(buffer)
                        buffer.clear()
                        buffer_len = 0
                if buffer:
                    yield "".join(buffer)

        chunk_iter = iter_text_chunks(input_path, _PRETOKEN_CHUNK_BYTES)
        est_chunks = max(1, (file_size + _PRETOKEN_CHUNK_BYTES - 1) // _PRETOKEN_CHUNK_BYTES)

        # Use multiprocessing for larger corpora regardless of special-token presence.
        use_multiprocessing = file_size >= _PRETOKEN_MIN_MP_BYTES and cpu_count() > 1

        final_counts = Counter()
        if use_multiprocessing:
            cpu_cores = max(1, cpu_count())
            num_processes = min(cpu_cores, est_chunks)
            chunksize = max(1, est_chunks // (num_processes * 4))
            process_fn = partial(_process_chunk, special_tokens_pat=special_tokens_pat)
            with Pool(num_processes) as pool:
                counter_iter = pool.imap_unordered(process_fn, chunk_iter, chunksize=chunksize)
                if show_progress:
                    counter_iter = tqdm(
                        counter_iter,
                        total=est_chunks,
                        desc="Pretokenizing",
                        unit="chunk",
                    )
                for counter in counter_iter:
                    final_counts.update(counter)
        else:
            part_iter = chunk_iter
            if show_progress:
                part_iter = tqdm(
                    part_iter,
                    total=est_chunks,
                    desc="Pretokenizing",
                    unit="chunk",
                )
            for part in part_iter:
                final_counts.update(_process_chunk(part, special_tokens_pat=special_tokens_pat))

        if not show_progress:
            print("FINISHED PRETOKENIZING...")
        return final_counts

    def build_pair_stats(
        pretokens: Counter[tuple[bytes]],
    ) -> tuple[list[list[bytes]], list[int], Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], set[int]]]:
        """Build word symbol lists, frequencies, pair counts, and pair->word index map."""
        word_symbols: list[list[bytes]] = []
        word_freqs: list[int] = []
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

        for idx, (word, count) in enumerate(pretokens.items()):
            symbols = list(word)
            word_symbols.append(symbols)
            word_freqs.append(count)
            if len(symbols) < 2:
                continue
            prev = symbols[0]
            for sym in symbols[1:]:
                pair = (prev, sym)
                pair_counts[pair] += count
                pair_to_words[pair].add(idx)
                prev = sym
        return word_symbols, word_freqs, pair_counts, pair_to_words

    token_key_cache: dict[bytes, tuple[int, ...]] = {}

    def token_key(token: bytes) -> tuple[int, ...]:
        key = token_key_cache.get(token)
        if key is None:
            # Reverse byte-wise lexicographic order for heap tie-breaks.
            # The trailing 256 sentinel handles prefix ties correctly, so
            # b" a" sorts ahead of b" " when counts are equal.
            key = tuple(255 - b for b in token) + (256,)
            token_key_cache[token] = key
        return key

    def pair_key(pair: tuple[bytes, bytes]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (token_key(pair[0]), token_key(pair[1]))

    def merge_word(
        symbols: list[bytes], pair: tuple[bytes, bytes], merged_token: bytes
    ) -> tuple[list[bytes], bool]:
        new_symbols: list[bytes] = []
        i = 0
        changed = False
        left, right = pair
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == left and symbols[i + 1] == right:
                new_symbols.append(merged_token)
                i += 2
                changed = True
            else:
                new_symbols.append(symbols[i])
                i += 1
        return new_symbols, changed

    # Vocab initialization
    vocab = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = bytes(special_token, "utf-8")
    merges = []

    # Load and pretokenize
    pretoken_counts = load_and_pretokenize(input_path, special_tokens)

    # Build initial pair statistics
    word_symbols, word_freqs, bp_counts, pair_to_words = build_pair_stats(pretoken_counts)

    # Heap of (-count, reversed_pair_key, pair) for fast max selection with GPT-2 tie-break.
    heap = [(-count, pair_key(pair), pair) for pair, count in bp_counts.items()]
    heapify(heap)

    # Optimized training loop with incremental updates
    merge_steps = max(0, vocab_size - len(vocab))
    merge_iter = range(merge_steps)
    if show_progress:
        merge_iter = tqdm(
            merge_iter,
            total=merge_steps,
            desc="Training BPE",
            unit="merge",
        )
    for _ in merge_iter:
        # Find the most frequent valid pair (lazy heap invalidation).
        while heap:
            neg_count, _, best_pair = heappop(heap)
            count = -neg_count
            if count == bp_counts.get(best_pair, 0):
                break
        else:
            raise RuntimeError("No valid pairs to merge; pair counts are empty or stale.")

        merged_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged_token
        merges.append(best_pair)

        affected_words = pair_to_words.get(best_pair)
        if not affected_words:
            bp_counts.pop(best_pair, None)
            continue

        for word_idx in list(affected_words):
            symbols = word_symbols[word_idx]
            new_symbols, changed = merge_word(symbols, best_pair, merged_token)
            if not changed:
                # Stale index; remove from mapping.
                affected_words.discard(word_idx)
                continue

            freq = word_freqs[word_idx]
            old_pairs = list(zip(symbols, symbols[1:]))
            new_pairs = list(zip(new_symbols, new_symbols[1:]))

            # Update pair counts (subtract old, add new).
            for p in old_pairs:
                new_count = bp_counts[p] - freq
                if new_count > 0:
                    bp_counts[p] = new_count
                else:
                    bp_counts.pop(p, None)
            for p in new_pairs:
                bp_counts[p] += freq

            # Update pair -> word index map.
            for p in set(old_pairs):
                word_set = pair_to_words.get(p)
                if word_set is not None:
                    word_set.discard(word_idx)
                    if not word_set:
                        pair_to_words.pop(p, None)
            for p in set(new_pairs):
                pair_to_words.setdefault(p, set()).add(word_idx)

            # Push updated pairs into the heap (lazy invalidation handles duplicates).
            touched_pairs = set(old_pairs)
            touched_pairs.update(new_pairs)
            for p in touched_pairs:
                count = bp_counts.get(p)
                if count:
                    heappush(heap, (-count, pair_key(p), p))

            word_symbols[word_idx] = new_symbols

        pair_to_words.pop(best_pair, None)
    return vocab, merges
