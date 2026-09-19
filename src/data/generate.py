"""Synthetic sequence reversal dataset generation script.

Generates pairs of random uppercase character sequences and their reversed counterparts,
saving them to CSV files for training and validation splits.
"""

import os
import sys
import random
import string
import argparse
from typing import List, Tuple

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd


def generate_sequence_pair(min_len: int = 3, max_len: int = 8, charset: str = string.ascii_uppercase) -> Tuple[str, str]:
    """Generate a single source string and its reversed target string.

    Args:
        min_len (int): Minimum length of the random sequence.
        max_len (int): Maximum length of the random sequence.
        charset (str): Available characters to sample from.

    Returns:
        Tuple[str, str]: (source_sequence, target_sequence)
    """
    length = random.randint(min_len, max_len)
    source = "".join(random.choices(charset, k=length))
    target = source[::-1]
    return source, target


def generate_dataset(
    num_samples: int,
    min_len: int = 3,
    max_len: int = 8,
    charset: str = string.ascii_uppercase,
) -> pd.DataFrame:
    """Generate a DataFrame containing random source and reversed target sequences.

    Args:
        num_samples (int): Total number of pairs to generate.
        min_len (int): Minimum length of sequences.
        max_len (int): Maximum length of sequences.
        charset (str): Character set.

    Returns:
        pd.DataFrame: DataFrame with columns 'source' and 'target'.
    """
    pairs: List[Tuple[str, str]] = [
        generate_sequence_pair(min_len=min_len, max_len=max_len, charset=charset)
        for _ in range(num_samples)
    ]
    df = pd.DataFrame(pairs, columns=["source", "target"])
    return df


def main():
    """CLI entrypoint for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic sequence reversal dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.getenv("DATA_DIR", "./data"),
        help="Directory to save train.csv and val.csv",
    )
    parser.add_argument("--num_train", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--num_val", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--min_len", type=int, default=3, help="Minimum sequence length")
    parser.add_argument("--max_len", type=int, default=8, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")), help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.data_dir, exist_ok=True)

    print(f"Generating {args.num_train} training samples with seed {args.seed}...")
    train_df = generate_dataset(args.num_train, min_len=args.min_len, max_len=args.max_len)
    train_path = os.path.join(args.data_dir, "train.csv")
    train_df.to_csv(train_path, index=False)
    print(f"Saved training dataset to {train_path}")

    # Set seed for independent validation generation
    random.seed(args.seed + 100)
    print(f"Generating {args.num_val} validation samples with seed {args.seed + 100}...")
    val_df = generate_dataset(args.num_val, min_len=args.min_len, max_len=args.max_len)
    val_path = os.path.join(args.data_dir, "val.csv")
    val_df.to_csv(val_path, index=False)
    print(f"Saved validation dataset to {val_path}")


if __name__ == "__main__":
    main()
