"""Unit tests for TinySeq2SeqTransformer architecture and PositionalEncoding."""

import ast
import os
import torch
import pytest
from src.model.transformer import PositionalEncoding, TinySeq2SeqTransformer


def test_no_huggingface_transformers_imported():
    """Verify AST ensures high-level 'transformers' package is not imported in model source."""
    model_file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src", "model", "transformer.py")
    )
    with open(model_file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=model_file_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "transformers" not in alias.name, f"Forbidden package imported: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert "transformers" not in node.module, f"Forbidden package imported from: {node.module}"


def test_positional_encoding():
    """Verify positional encoding module produces correct tensor shapes and adds to embeddings."""
    d_model = 64
    seq_len = 10
    batch_size = 2
    pe_module = PositionalEncoding(d_model=d_model, max_len=100)

    dummy_embeddings = torch.zeros(batch_size, seq_len, d_model)
    output = pe_module(dummy_embeddings)

    assert output.shape == (batch_size, seq_len, d_model)
    # Ensure buffer 'pe' is registered and not all zeros
    assert hasattr(pe_module, "pe")
    assert not torch.all(pe_module.pe == 0)


def test_transformer_forward_dimensions():
    """Verify model forward pass outputs correct logits tensor dimensions (batch, seq_len, vocab_size)."""
    vocab_size = 29
    d_model = 64
    nhead = 4
    num_layers = 2
    batch_size = 4
    src_len = 8
    tgt_len = 7

    model = TinySeq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        pad_idx=0,
    )

    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

    src_padding_mask = (src == 0)
    tgt_padding_mask = (tgt == 0)
    tgt_mask = TinySeq2SeqTransformer.generate_square_subsequent_mask(tgt_len)

    logits = model(
        src=src,
        tgt=tgt,
        src_mask=None,
        tgt_mask=tgt_mask,
        src_padding_mask=src_padding_mask,
        tgt_padding_mask=tgt_padding_mask,
    )

    assert logits.shape == (batch_size, tgt_len, vocab_size)


def test_model_parameter_count_under_one_million():
    """Verify the model stays lightweight and easily below 1,000,000 parameters."""
    model = TinySeq2SeqTransformer(vocab_size=29, d_model=64, nhead=4, num_layers=2)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count < 1_000_000
    print(f"Model parameter count: {param_count}")


def test_transformer_encode_decode_methods():
    """Verify encode and decode methods used for autoregressive generation."""
    vocab_size = 29
    d_model = 64
    batch_size = 2
    src_len = 6

    model = TinySeq2SeqTransformer(vocab_size=vocab_size, d_model=d_model, nhead=4, num_layers=2)
    src = torch.randint(1, vocab_size, (batch_size, src_len))

    memory = model.encode(src)
    assert memory.shape == (batch_size, src_len, d_model)

    tgt = torch.tensor([[1], [1]], dtype=torch.long)  # Start with SOS
    tgt_mask = TinySeq2SeqTransformer.generate_square_subsequent_mask(tgt.size(1))
    logits = model.decode(tgt=tgt, memory=memory, tgt_mask=tgt_mask)

    assert logits.shape == (batch_size, 1, vocab_size)
