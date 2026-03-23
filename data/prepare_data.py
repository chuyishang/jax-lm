"""
Prepare training data by training a BPE tokenizer and converting text files to binary format.

This script:
1. Trains a BPE tokenizer on the training data
2. Tokenizes both train and validation sets
3. Saves tokenized data as binary files (np.uint16)
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

# from jax_impl.tokenizer import Tokenizer
from data.tokenizer import Tokenizer
from data.train_bpe import train_bpe


def train_tokenizer(
    train_path: str,
    vocab_size: int = 10000,
    special_tokens: list[str] | None = None,
    output_dir: str = "tokenizer",
    show_progress: bool = False,
):
    """Train a BPE tokenizer on the training data.

    Args:
        train_path: Path to training text file
        vocab_size: Size of vocabulary to train
        special_tokens: List of special tokens
        output_dir: Directory to save tokenizer files
        show_progress: Whether to display progress bars during training

    Returns:
        Tokenizer instance
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    print(f"\n{'=' * 80}")
    print("Training BPE Tokenizer")
    print(f"{'=' * 80}")
    print(f"Training data: {train_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")

    # Train BPE
    print("\nTraining BPE (this may take a while for large files)...")
    vocab, merges = train_bpe(
        input_path=train_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=show_progress,
    )

    print("✓ Training complete!")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Number of merges: {len(merges)}")

    # Enforce uint16 compatibility early (training uses uint16 memmap)
    # NOTE: shouldn't this be earlier? we can check if this will overflow using `vocab_size`.
    max_token_id = max(vocab.keys()) if vocab else -1
    if max_token_id >= 65536:
        raise ValueError(
            f"Max token id {max_token_id} exceeds uint16 range. Use a smaller vocab size or reduce special tokens."
        )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save vocab and merges
    vocab_file = output_path / "vocab.pkl"
    merges_file = output_path / "merges.pkl"

    with open(vocab_file, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_file, "wb") as f:
        pickle.dump(merges, f)

    print(f"\n✓ Tokenizer saved to {output_dir}/")
    print("  - vocab.pkl")
    print("  - merges.pkl")

    # Create tokenizer instance
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    return tokenizer


def tokenize_file(
    input_path: str | Path,
    output_path: str | Path,
    tokenizer: Tokenizer,
    flush_tokens: int = 1_000_000,
    show_progress: bool = False,
):
    """
    Tokenize a text file and save as binary.

    Args:
        input_path: Path to input text file
        output_path: Path to output binary file
        tokenizer: Trained tokenizer
        flush_tokens: Number of token ids to buffer before writing to disk
        show_progress: Whether to display a progress bar during tokenization
    """
    print(f"\nTokenizing {input_path}...")

    # Read file size
    file_size = Path(input_path).stat().st_size
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")

    total_tokens = 0
    max_token_id = -1
    buffer: list[int] = []
    report_every = flush_tokens * 10
    next_report = report_every
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(
            total=file_size,
            desc=f"Tokenizing {Path(input_path).name}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )

    # Stream lines to avoid splitting tokens across chunk boundaries.
    with open(input_path, encoding="utf-8") as f, open(output_path, "wb") as out_f:
        for line_num, line in enumerate(f, start=1):
            token_ids = tokenizer.encode(line)
            # TODO: this block seems pretty unnecessary lol, given that we check for overflow vocab when training
            if token_ids:
                local_max = max(token_ids)
                if local_max >= 65536:
                    raise ValueError(
                        f"Max token id {local_max} exceeds uint16 range at line {line_num}. "
                        "Use a smaller vocab size or reduce special tokens."
                    )
                if local_max > max_token_id:
                    max_token_id = local_max

            buffer.extend(token_ids)
            total_tokens += len(token_ids)

            if len(buffer) >= flush_tokens:
                np.asarray(buffer, dtype=np.uint16).tofile(out_f)
                buffer.clear()

            if progress_bar:
                progress_bar.update(len(line.encode("utf-8")))

            if not show_progress and total_tokens >= next_report:
                print(f"  Processed {total_tokens:,} tokens so far...")
                next_report += report_every

        if buffer:
            np.asarray(buffer, dtype=np.uint16).tofile(out_f)
            buffer.clear()

    if progress_bar:
        progress_bar.close()

    print("Tokenization complete!")
    print(f"Total tokens: {total_tokens:,}")

    output_size = Path(output_path).stat().st_size
    print(f"Saved to {output_path}")
    print(f"Output size: {output_size / 1024 / 1024:.2f} MB")
    if output_size:
        print(f"Compression ratio: {file_size / output_size:.2f}x")
    else:
        print("Compression ratio: N/A (empty output)")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data by tokenizing text files")
    parser.add_argument("--train-path", type=str, required=True, help="Path to training text file")
    parser.add_argument("--val-path", type=str, default=None, help="Path to validation text file (optional)")
    parser.add_argument(
        "--vocab-size", type=int, default=10000, help="Vocabulary size for BPE tokenizer (default: 10000)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data", help="Directory to save tokenized files (default: data)"
    )
    parser.add_argument(
        "--tokenizer-dir", type=str, default="tokenizer", help="Directory to save/load tokenizer (default: tokenizer)"
    )
    parser.add_argument(
        "--special-tokens", nargs="+", default=["<|endoftext|>"], help="Special tokens to add (default: <|endoftext|>)"
    )
    parser.add_argument(
        "--use-existing-tokenizer",
        action="store_true",
        help="Use existing tokenizer from tokenizer-dir instead of training new one",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show progress bars during pretokenization, training, and tokenization",
    )

    args = parser.parse_args()

    print(f"\n{'=' * 80}")
    print("Data Preparation Pipeline")
    print(f"{'=' * 80}")

    # Train or load tokenizer
    if args.use_existing_tokenizer:
        print(f"\nLoading existing tokenizer from {args.tokenizer_dir}/")
        tokenizer = Tokenizer.from_files(
            vocab_filepath=f"{args.tokenizer_dir}/vocab.pkl",
            merges_filepath=f"{args.tokenizer_dir}/merges.pkl",
            special_tokens=args.special_tokens,
        )
        max_token_id = max(tokenizer.vocab.keys()) if tokenizer.vocab else -1
        if max_token_id >= 65536:
            raise ValueError(
                f"Max token id {max_token_id} exceeds uint16 range. Use a smaller vocab size or reduce special tokens."
            )
        print("✓ Tokenizer loaded")
    else:
        tokenizer = train_tokenizer(
            train_path=args.train_path,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
            output_dir=args.tokenizer_dir,
            show_progress=args.progress,
        )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenize training data
    print(f"\n{'=' * 80}")
    print("Tokenizing Training Data")
    print(f"{'=' * 80}")
    train_output = output_dir / "train.bin"
    tokenize_file(args.train_path, train_output, tokenizer, show_progress=args.progress)

    # Tokenize validation data if provided
    if args.val_path:
        print(f"\n{'=' * 80}")
        print("Tokenizing Validation Data")
        print(f"{'=' * 80}")
        val_output = output_dir / "val.bin"
        tokenize_file(args.val_path, val_output, tokenizer, show_progress=args.progress)

    print(f"\n{'=' * 80}")
    print("Data Preparation Complete!")
    print(f"{'=' * 80}")
    print(f"\nTokenized files saved to {args.output_dir}/:")
    print("  - train.bin")
    if args.val_path:
        print("  - val.bin")
    print(f"\nTokenizer saved to {args.tokenizer_dir}/:")
    print("  - vocab.pkl")
    print("  - merges.pkl")
    print("\nYou can now use these files for training!")
    print("Update your config.yaml to use:")
    print("  train_path: train.bin")
    if args.val_path:
        print("  val_path: val.bin")


if __name__ == "__main__":
    main()
