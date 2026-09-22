
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re

def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input file.

    Args:
        input_path: str  Path to a text file with BPE tokenizer training data.
        vocab_size: int  A positive integer that defines the maximum final vocabulary size (including
        the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: list[str]  A list of strings to add to the vocabulary. During training, treat
        them as hard boundaries that prevent merges across their spans, but do not include them when
        computing merge statistics.

    Returns:
        vocab: dict[int, bytes]  The tokenizer vocabulary, a mapping from int (token ID in the
        vocabulary) to bytes (token bytes).
        merges: list[tuple[bytes, bytes]]  A list of BPE merges produced from training. Each list
        item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
        <token2>. The merges should be ordered by order of creation.
    """
    vocab = init_vocab()
    merges = []

    corpus_counts = prepare_corpus_counts(input_path, special_tokens)
    max_merges = vocab_size - len(vocab) - len(special_tokens)

    while len(merges) < max_merges:
        pair_counts = get_pair_counts(corpus_counts)
        if not pair_counts:
            break

        # Find the most frequent pair
        most_frequent_pair = max(pair_counts, key=pair_counts.get)
        merges.append(most_frequent_pair)

        # Update the vocabulary with the new merged token
        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[len(vocab)] = new_token

        # Merge the pair in the corpus counts
        corpus_counts = merge_pair_in_corpus(corpus_counts, most_frequent_pair)

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    return vocab, merges

def init_vocab():
    """
    Initialize the vocabulary with all possible byte values (0-255) and any special tokens.

    Returns:
        vocab: dict[int, bytes]  The initial vocabulary mapping from int (token ID) to bytes (token bytes).
    """
    # Initialize the vocabulary with all possible byte values (0-255)
    return {i:bytes([i])for i in range(256)}

def prepare_corpus_counts(input_path: str, special_tokens: list[str]) -> dict[tuple, int]:
    """
    Prepare the corpus counts for BPE training.

    Args:
        input_path: str  Path to a text file with BPE tokenizer training data.
        special_tokens: list[str]  A list of strings to add to the vocabulary. During training, treat
        them as hard boundaries that prevent merges across their spans, but do not include them when
        computing merge statistics.
    
    Returns:
        corpus_counts: dict[tuple, int]  A dictionary mapping from pre-token tuples to their counts in the corpus.
    """
    # Read the input text file
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        corpus_counts = {}
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list = re.split("|".join(re.escape(token) for token in special_tokens), chunk) if special_tokens else [chunk]

            for sub_chunk in chunk_list:
                for x in re.finditer(PAT, sub_chunk):
                    token = x.group(0)
                    pretoken_tuple = tuple(bytes([c]) for c in token.encode("utf-8"))
                    corpus_counts[pretoken_tuple] = corpus_counts.get(pretoken_tuple, 0) + 1

    return corpus_counts

def get_pair_counts(corpus_counts: dict[tuple, int]) -> dict[tuple[bytes, bytes], int]:
    """
    Get the counts of adjacent byte pairs in the corpus.

    Args:
        corpus_counts: dict[tuple, int]  A dictionary mapping from pre-token tuples to their counts in the corpus.

    Returns:
        pair_counts: dict[tuple[bytes, bytes], int]  A dictionary mapping from adjacent byte pairs to their counts in the corpus.
    """
    pair_counts = {}
    for pretoken_tuple, count in corpus_counts.items():
        for i in range(len(pretoken_tuple) - 1):
            pair = (pretoken_tuple[i], pretoken_tuple[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    return pair_counts


def merge_pair_in_corpus(corpus_counts: dict[tuple, int], pair_to_merge: tuple[bytes, bytes]) -> dict[tuple, int]:
    """
    Merge a specific byte pair in the corpus counts.

    Args:
        corpus_counts: dict[tuple, int]  A dictionary mapping from pre-token tuples to their counts in the corpus.
        pair_to_merge: tuple[bytes, bytes]  The byte pair to merge.
    
    Returns:
        merged_counts: dict[tuple, int]  A dictionary mapping from pre-token tuples to their counts in the corpus after merging the specified byte pair.    
    """
    merged_counts = {}
    merged_token = pair_to_merge[0] + pair_to_merge[1]
    for pretoken_tuple, count in corpus_counts.items():
        merged_tuple = []
        i = 0
        while i < len(pretoken_tuple):
            if i < len(pretoken_tuple) - 1 and (pretoken_tuple[i], pretoken_tuple[i + 1]) == pair_to_merge:
                merged_tuple.append(merged_token)
                i += 2  # Skip the next token since it's part of the merged pair
            else:
                merged_tuple.append(pretoken_tuple[i])
                i += 1
        merged_counts[tuple(merged_tuple)] = count
    return merged_counts