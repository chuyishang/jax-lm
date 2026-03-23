#!/bin/bash
# Launch script for data preparation

set -e

# ---------------------------
# Config settings
# ---------------------------
PYTHON_CMD=${PYTHON_CMD:-python}
TRAIN_PATH="/scratch/current/shang/data/owt_train.txt"
VAL_PATH="/scratch/current/shang/data/owt_valid.txt"
VOCAB_SIZE=32000
OUTPUT_DIR="/scratch/current/shang/data/owt"
TOKENIZER_DIR="/scratch/current/shang/data/owt/tokenizer"
USE_EXISTING_TOKENIZER=false
SPECIAL_TOKENS=("<|endoftext|>")
PROGRESS=true

show_help() {
    cat << 'HELP'
Data Preparation Launch Script

Usage:
    ./prepare_data.sh [OPTIONS]

This script forwards all arguments to:
    python -m data.prepare_data

Common options:
    --train-path PATH           Path to training text file (required)
    --val-path PATH             Path to validation text file (optional)
    --vocab-size N              Vocabulary size for BPE tokenizer (default: 10000)
    --output-dir DIR            Directory to save tokenized files (default: data)
    --tokenizer-dir DIR         Directory to save/load tokenizer (default: tokenizer)
    --special-tokens TOKENS...  Special tokens to add (default: <|endoftext|>)
    --use-existing-tokenizer    Use tokenizer from tokenizer-dir instead of training
    --progress/--no-progress    Toggle progress bars (default: --progress via this script)

Examples:
    ./prepare_data.sh --train-path data/TinyStoriesV2-GPT4-train.txt \
      --val-path data/TinyStoriesV2-GPT4-valid.txt --vocab-size 10000

    ./prepare_data.sh --train-path data/train.txt --special-tokens "<|endoftext|>" "<|pad|>"

Tip:
    To run with uv, use: uv run bash scripts/prepare_data.sh ...
HELP
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    echo ""
    $PYTHON_CMD -m data.prepare_data --help
    exit 0
fi

CMD=("$PYTHON_CMD" -m data.prepare_data)

if [[ -n "$TRAIN_PATH" ]]; then
    CMD+=(--train-path "$TRAIN_PATH")
fi
if [[ -n "$VAL_PATH" ]]; then
    CMD+=(--val-path "$VAL_PATH")
fi
if [[ -n "$VOCAB_SIZE" ]]; then
    CMD+=(--vocab-size "$VOCAB_SIZE")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
    CMD+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "$TOKENIZER_DIR" ]]; then
    CMD+=(--tokenizer-dir "$TOKENIZER_DIR")
fi
if [[ "$USE_EXISTING_TOKENIZER" == "true" ]]; then
    CMD+=(--use-existing-tokenizer)
fi
if [[ ${#SPECIAL_TOKENS[@]} -gt 0 ]]; then
    CMD+=(--special-tokens "${SPECIAL_TOKENS[@]}")
fi
if [[ "$PROGRESS" == "true" ]]; then
    CMD+=(--progress)
elif [[ "$PROGRESS" == "false" ]]; then
    CMD+=(--no-progress)
fi

if [[ -z "$TRAIN_PATH" && $# -eq 0 ]]; then
    echo "Error: TRAIN_PATH is not set in the script and no CLI args were provided."
    echo ""
    show_help
    exit 1
fi

# CLI args override config if the same flag is provided.
CMD+=("$@")

exec "${CMD[@]}"
