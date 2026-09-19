import os
import sys
import json
import random
import argparse

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional

from src.data.tokenizer import CharacterTokenizer, PAD_ID
from src.data.dataset import SequenceReversalDataset, collate_fn
from src.model.transformer import TinySeq2SeqTransformer


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file if it exists, otherwise return defaults.

    Args:
        config_path (str): Path to JSON configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    default_config = {
        "model_config": {
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
        },
        "train_config": {
            "epochs": 10,
            "batch_size": 64,
            "lr": 0.003,
        },
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if "model_config" in loaded:
                    default_config["model_config"].update(loaded["model_config"])
                if "train_config" in loaded:
                    default_config["train_config"].update(loaded["train_config"])
        except Exception as e:
            print(f"Warning: Failed to parse {config_path} ({e}). Using default configuration.")
    return default_config


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: Optional[torch.device] = None,
) -> float:
    """Execute one epoch of training using teacher forcing.

    Args:
        model (nn.Module): Transformer model instance.
        dataloader (DataLoader): Training DataLoader.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        criterion (nn.CrossEntropyLoss): Loss criterion.
        device (Optional[torch.device]): Target compute device.

    Returns:
        float: Average epoch training loss.
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    total_loss = 0.0

    for batch in dataloader:
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)

        # Shift target to create input (tokens 0..N-1) and expected output (tokens 1..N)
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        # Create padding and causal masks
        src_padding_mask = (src == PAD_ID)
        tgt_padding_mask = (tgt_input == PAD_ID)
        tgt_mask = TinySeq2SeqTransformer.generate_square_subsequent_mask(
            tgt_input.size(1), device=device
        )

        optimizer.zero_grad()

        # Forward pass
        logits = model(
            src=src,
            tgt=tgt_input,
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

        # Compute cross entropy loss (ignoring PAD token)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_expected.reshape(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: Optional[torch.device] = None,
) -> float:
    """Evaluate model loss on validation dataset using teacher forcing.

    Args:
        model (nn.Module): Transformer model instance.
        dataloader (DataLoader): Validation DataLoader.
        criterion (nn.CrossEntropyLoss): Loss criterion.
        device (Optional[torch.device]): Target compute device.

    Returns:
        float: Average validation loss.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            src_padding_mask = (src == PAD_ID)
            tgt_padding_mask = (tgt_input == PAD_ID)
            tgt_mask = TinySeq2SeqTransformer.generate_square_subsequent_mask(
                tgt_input.size(1), device=device
            )

            logits = model(
                src=src,
                tgt=tgt_input,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_expected.reshape(-1),
            )
            total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def train_model(
    config_path: str = "./submission.json",
    data_dir: str = "./data",
    output_dir: str = "./output",
    device_str: str = "cpu",
    seed: int = 42,
) -> Tuple[TinySeq2SeqTransformer, pd.DataFrame]:
    """Orchestrate entire training pipeline across multiple epochs.

    Args:
        config_path (str): Path to JSON configuration.
        data_dir (str): Directory containing train.csv and val.csv.
        output_dir (str): Directory to save metrics.csv and checkpoint.pt.
        device_str (str): Compute device ('cpu' or 'cuda').
        seed (int): Random seed.

    Returns:
        Tuple[TinySeq2SeqTransformer, pd.DataFrame]: Trained model and metrics DataFrame.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    config = load_config(config_path)
    model_config = config.get("model_config", {})
    train_config = config.get("train_config", {})

    d_model = int(model_config.get("d_model", 64))
    nhead = int(model_config.get("nhead", 4))
    num_layers = int(model_config.get("num_layers", 2))

    epochs = int(train_config.get("epochs", 5))
    batch_size = int(train_config.get("batch_size", 64))
    lr = float(train_config.get("lr", 0.001))

    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"Training on device: {device}")

    # Initialize tokenizer and datasets
    tokenizer = CharacterTokenizer()
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Datasets not found in {data_dir}. Run src/data/generate.py first.")

    train_dataset = SequenceReversalDataset(train_path, tokenizer=tokenizer)
    val_dataset = SequenceReversalDataset(val_path, tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Initialize model
    model = TinySeq2SeqTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        pad_idx=PAD_ID,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized TinySeq2SeqTransformer with {param_count:,} trainable parameters.")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metrics_records = []

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device=device)
        val_loss = evaluate_loss(model, val_loader, criterion, device=device)

        metrics_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save metrics to output/metrics.csv
    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved training metrics to {metrics_path}")

    # Save checkpoint to output/checkpoint.pt
    checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": tokenizer.vocab_size,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "pad_idx": PAD_ID,
        },
        "train_config": train_config,
        "seed": seed,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved model checkpoint to {checkpoint_path}")

    return model, metrics_df


def main():
    """CLI entrypoint for training."""
    parser = argparse.ArgumentParser(description="Train Tiny Seq2Seq Transformer.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.getenv("CONFIG_PATH", "./submission.json"),
        help="Path to JSON config",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.getenv("DATA_DIR", "./data"),
        help="Data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getenv("OUTPUT_DIR", "./output"),
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("DEVICE", "cpu"),
        help="Compute device (cpu/cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("SEED", "42")),
        help="Random seed",
    )
    args = parser.parse_args()

    train_model(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
