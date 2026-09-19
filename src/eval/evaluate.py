import os
import sys
import json
import argparse

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional

from src.data.tokenizer import CharacterTokenizer, PAD_ID, SOS_ID, EOS_ID
from src.model.transformer import TinySeq2SeqTransformer


def greedy_decode(
    model: nn.Module,
    src_tensor: torch.Tensor,
    max_len: int = 30,
    sos_id: int = SOS_ID,
    eos_id: int = EOS_ID,
    pad_id: int = PAD_ID,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Autoregressively generate output token IDs for a given source sequence using greedy search.

    Args:
        model (nn.Module): Transformer model.
        src_tensor (torch.Tensor): Source tensor of shape (1, src_seq_len) or (batch_size, src_seq_len).
        max_len (int): Maximum sequence length cutoff. Defaults to 30.
        sos_id (int): Start of sequence token ID. Defaults to SOS_ID (1).
        eos_id (int): End of sequence token ID. Defaults to EOS_ID (2).
        pad_id (int): Padding token ID. Defaults to PAD_ID (0).
        device (Optional[torch.device]): Compute device.

    Returns:
        torch.Tensor: Decoded token IDs tensor of shape (1, generated_seq_len).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    if src_tensor.dim() == 1:
        src_tensor = src_tensor.unsqueeze(0)

    src_tensor = src_tensor.to(device)
    src_padding_mask = (src_tensor == pad_id)

    with torch.no_grad():
        memory = model.encode(src_tensor, src_padding_mask=src_padding_mask)
        ys = torch.tensor([[sos_id]], dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_mask = TinySeq2SeqTransformer.generate_square_subsequent_mask(
                ys.size(1), device=device
            )
            out = model.decode(
                tgt=ys,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
            )

            # Get logits for the last token position
            prob = out[:, -1, :]
            next_token = torch.argmax(prob, dim=-1, keepdim=True)

            ys = torch.cat([ys, next_token], dim=1)

            if next_token.item() == eos_id:
                break

    return ys


def evaluate_exact_match(
    model: nn.Module,
    val_df: pd.DataFrame,
    tokenizer: CharacterTokenizer,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Tuple[float, List[Dict[str, str]]]:
    """Compute Exact Match (EM) accuracy on a validation dataset.

    Args:
        model (nn.Module): Transformer model to evaluate.
        val_df (pd.DataFrame): DataFrame containing 'source' and 'target' columns.
        tokenizer (CharacterTokenizer): Tokenizer instance.
        device (torch.device): Compute device.
        max_samples (Optional[int]): Max samples to evaluate (evaluates all if None).

    Returns:
        Tuple[float, List[Dict[str, str]]]: (Exact Match Accuracy [0.0..1.0], sample predictions list)
    """
    model.eval()
    samples = val_df if max_samples is None else val_df.iloc[:max_samples]

    correct = 0
    total = len(samples)
    prediction_logs: List[Dict[str, str]] = []

    for idx, row in samples.iterrows():
        source_str = str(row["source"])
        target_str = str(row["target"])

        src_ids = tokenizer.encode(source_str, add_special_tokens=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

        pred_ids = greedy_decode(
            model=model,
            src_tensor=src_tensor,
            max_len=len(source_str) + 5,
            device=device,
        )

        pred_str = tokenizer.decode(pred_ids[0], remove_special=True)
        is_match = (pred_str == target_str)
        if is_match:
            correct += 1

        if len(prediction_logs) < 10:
            prediction_logs.append({
                "source": source_str,
                "expected": target_str,
                "predicted": pred_str,
                "match": is_match,
            })

    em_score = correct / max(total, 1)
    return em_score, prediction_logs


def run_evaluation(
    checkpoint_path: str = "./output/checkpoint.pt",
    val_path: str = "./data/val.csv",
    output_dir: str = "./output",
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """Run full evaluation comparing trained model checkpoint against random baseline.

    Args:
        checkpoint_path (str): Path to trained model checkpoint.
        val_path (str): Path to validation dataset CSV.
        output_dir (str): Path to output directory to store eval_results.json.
        device_str (str): Compute device string ('cpu' or 'cuda').

    Returns:
        Dict[str, Any]: Evaluation summary dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"Evaluating on device: {device}")

    tokenizer = CharacterTokenizer()

    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation dataset not found at {val_path}. Run src/data/generate.py first.")

    val_df = pd.read_csv(val_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run src/train/train.py first.")

    print(f"Loading trained model checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model configuration
    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        m_cfg = checkpoint["model_config"]
        d_model = m_cfg.get("d_model", 64)
        nhead = m_cfg.get("nhead", 4)
        num_layers = m_cfg.get("num_layers", 2)
        state_dict = checkpoint["model_state_dict"]
    else:
        # Fallback if checkpoint contains only state_dict
        d_model = 64
        nhead = 4
        num_layers = 2
        state_dict = checkpoint

    # Instantiate trained model
    trained_model = TinySeq2SeqTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        pad_idx=PAD_ID,
    ).to(device)
    trained_model.load_state_dict(state_dict)

    # Instantiate uninitialized random-weight baseline model (same architecture)
    random_model = TinySeq2SeqTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        pad_idx=PAD_ID,
    ).to(device)

    print("Evaluating trained model on validation set...")
    trained_em, trained_samples = evaluate_exact_match(
        model=trained_model,
        val_df=val_df,
        tokenizer=tokenizer,
        device=device,
    )

    print("Evaluating random-weight baseline model on validation set...")
    random_em, random_samples = evaluate_exact_match(
        model=random_model,
        val_df=val_df,
        tokenizer=tokenizer,
        device=device,
    )

    sample_pred_entry = trained_samples[0] if trained_samples else {"source": "", "expected": "", "predicted": ""}
    sample_prediction_str = (
        f"Source: {sample_pred_entry['source']} | "
        f"Expected: {sample_pred_entry['expected']} | "
        f"Predicted: {sample_pred_entry['predicted']}"
    )

    results = {
        "trained_em_score": round(float(trained_em), 4),
        "random_em_score": round(float(random_em), 4),
        "sample_prediction": sample_prediction_str,
    }

    eval_json_path = os.path.join(output_dir, "eval_results.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluation results to {eval_json_path}")
    print(f"Results: {json.dumps(results, indent=2)}")

    return results


def main():
    """CLI entrypoint for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Tiny Seq2Seq Transformer model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.getenv("CHECKPOINT_PATH", "./output/checkpoint.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default=os.path.join(os.getenv("DATA_DIR", "./data"), "val.csv"),
        help="Path to validation CSV",
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
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        val_path=args.val_path,
        output_dir=args.output_dir,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
