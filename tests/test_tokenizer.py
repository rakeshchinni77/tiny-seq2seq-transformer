"""Unit tests for CharacterTokenizer and Dataset collator."""

import string
import torch
import pytest
from src.data.tokenizer import CharacterTokenizer, PAD_ID, SOS_ID, EOS_ID
from src.data.dataset import collate_fn


def test_tokenizer_initialization():
    """Verify special tokens and vocabulary mapping upon initialization."""
    tokenizer = CharacterTokenizer()
    assert tokenizer.pad_id == PAD_ID
    assert tokenizer.sos_id == SOS_ID
    assert tokenizer.eos_id == EOS_ID
    assert tokenizer.vocab_size == 29  # 3 special + 26 alphabet
    assert len(tokenizer) == 29

    for char in string.ascii_uppercase:
        assert char in tokenizer.char2id
        tid = tokenizer.char2id[char]
        assert tokenizer.id2char[tid] == char


def test_tokenizer_encode_decode():
    """Verify encoding with special tokens and decoding back to original text."""
    tokenizer = CharacterTokenizer()
    text = "HELLO"
    encoded = tokenizer.encode(text, add_special_tokens=True)

    assert encoded[0] == SOS_ID
    assert encoded[-1] == EOS_ID
    assert len(encoded) == len(text) + 2

    # Verify decoding
    decoded = tokenizer.decode(encoded, remove_special=True)
    assert decoded == text


def test_tokenizer_decode_with_special():
    """Verify decoding without stripping special tokens."""
    tokenizer = CharacterTokenizer()
    text = "AB"
    encoded = tokenizer.encode(text, add_special_tokens=True)
    decoded = tokenizer.decode(encoded, remove_special=False)
    assert decoded == f"<SOS>AB<EOS>"


def test_tokenizer_unknown_char_error():
    """Verify ValueError is raised on unknown character."""
    tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError):
        tokenizer.encode("hello123")


def test_collate_fn_padding():
    """Verify batch collation pads sequences to maximum length with PAD_ID."""
    tokenizer = CharacterTokenizer()
    sample1 = (tokenizer.encode("A"), tokenizer.encode("A"))
    sample2 = (tokenizer.encode("ABCD"), tokenizer.encode("DCBA"))

    batch = [sample1, sample2]
    src_padded, tgt_padded = collate_fn(batch, pad_id=PAD_ID)

    # Max length of src: len("ABCD") + 2 (SOS, EOS) = 6
    assert src_padded.shape == (2, 6)
    assert tgt_padded.shape == (2, 6)

    # First sample should be padded at the end
    assert src_padded[0, -1].item() == PAD_ID
    assert src_padded[0, -2].item() == PAD_ID
    assert src_padded[0, -3].item() == PAD_ID
    # Second sample should not end in PAD_ID
    assert src_padded[1, -1].item() == EOS_ID
