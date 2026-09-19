"""Integration and contract verification tests for dataset, training metrics, and evaluation results."""

import os
import json
import math
import pandas as pd
import torch
import pytest


def test_submission_config_contract():
    """Verify submission.json adheres to the required hyperparameter schema."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "submission.json")
    assert os.path.exists(config_path), "submission.json must exist at root"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert "model_config" in config, "model_config key missing in submission.json"
    assert "train_config" in config, "train_config key missing in submission.json"

    assert "d_model" in config["model_config"]
    assert "nhead" in config["model_config"]
    assert "num_layers" in config["model_config"]

    assert "epochs" in config["train_config"]
    assert config["train_config"]["epochs"] >= 3, "epochs must be >= 3"
    assert "batch_size" in config["train_config"]


def test_dataset_reversal_integrity():
    """Verify generated dataset files exist and every row is a strict string reversal."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    assert os.path.exists(train_path), f"Train dataset missing at {train_path}"
    assert os.path.exists(val_path), f"Val dataset missing at {val_path}"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    assert len(train_df) >= 100, "Train dataset should have sufficient rows"
    assert len(val_df) >= 50, "Val dataset should have sufficient rows"

    assert "source" in train_df.columns and "target" in train_df.columns
    assert "source" in val_df.columns and "target" in val_df.columns

    # Assert exact reversal for all rows
    for _, row in train_df.iterrows():
        src = str(row["source"])
        tgt = str(row["target"])
        assert tgt == src[::-1], f"Mismatch in train set: {src} -> {tgt}"

    for _, row in val_df.iterrows():
        src = str(row["source"])
        tgt = str(row["target"])
        assert tgt == src[::-1], f"Mismatch in val set: {src} -> {tgt}"


def test_metrics_loss_convergence():
    """Verify metrics.csv shows downward loss trend, at least 3 epochs, and no NaNs."""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    metrics_path = os.path.join(output_dir, "metrics.csv")

    assert os.path.exists(metrics_path), f"metrics.csv missing at {metrics_path}"

    metrics_df = pd.read_csv(metrics_path)
    assert "epoch" in metrics_df.columns
    assert "train_loss" in metrics_df.columns
    assert "val_loss" in metrics_df.columns

    assert len(metrics_df) >= 3, f"Must have trained for at least 3 epochs, found {len(metrics_df)}"

    # Check for NaNs
    assert not metrics_df["train_loss"].isna().any(), "train_loss contains NaN values"
    assert not metrics_df["val_loss"].isna().any(), "val_loss contains NaN values"

    first_val_loss = float(metrics_df["val_loss"].iloc[0])
    final_val_loss = float(metrics_df["val_loss"].iloc[-1])

    assert final_val_loss < first_val_loss, (
        f"Validation loss did not decrease: initial={first_val_loss}, final={final_val_loss}"
    )


def test_checkpoint_validity():
    """Verify checkpoint.pt exists and is a valid loadable PyTorch file."""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    ckpt_path = os.path.join(output_dir, "checkpoint.pt")

    assert os.path.exists(ckpt_path), f"checkpoint.pt missing at {ckpt_path}"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    assert checkpoint is not None


def test_evaluation_em_score_threshold():
    """Verify eval_results.json schema and that trained model outperforms random baseline by > 0.1."""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    eval_path = os.path.join(output_dir, "eval_results.json")

    assert os.path.exists(eval_path), f"eval_results.json missing at {eval_path}"

    with open(eval_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    assert "trained_em_score" in results
    assert "random_em_score" in results
    assert "sample_prediction" in results

    trained_em = float(results["trained_em_score"])
    random_em = float(results["random_em_score"])

    assert 0.0 <= trained_em <= 1.0
    assert 0.0 <= random_em <= 1.0
    assert trained_em > random_em + 0.1, (
        f"Trained model EM score ({trained_em}) must beat random baseline ({random_em}) by > 0.1"
    )
