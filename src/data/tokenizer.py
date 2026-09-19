"""Custom character-level tokenizer for sequence reversal task.

Handles vocabulary creation, special tokens (<PAD>, <SOS>, <EOS>),
encoding strings into token ID sequences with special tokens,
and decoding token ID sequences back to strings.
"""

import string
from typing import List, Optional, Union, Sequence, Any
import torch


PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2


class CharacterTokenizer:
    """CharacterTokenizer manages mapping between characters and discrete integer token IDs.

    Attributes:
        charset (List[str]): List of valid characters.
        pad_id (int): Token ID for <PAD> (0).
        sos_id (int): Token ID for <SOS> (1).
        eos_id (int): Token ID for <EOS> (2).
        char2id (dict): Mapping from character/token string to integer ID.
        id2char (dict): Mapping from integer ID to character/token string.
        vocab_size (int): Total vocabulary size including special tokens.
    """

    def __init__(self, charset: Optional[List[str]] = None) -> None:
        """Initialize vocabulary with special tokens and charset.

        Args:
            charset (Optional[List[str]]): List of valid characters.
                Defaults to uppercase ASCII letters ['A'..'Z'].
        """
        if charset is None:
            charset = list(string.ascii_uppercase)
        self.charset = list(charset)

        # Special tokens
        self.pad_token = PAD_TOKEN
        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN

        self.pad_id = PAD_ID
        self.sos_id = SOS_ID
        self.eos_id = EOS_ID

        # Build vocabulary mappings
        self.char2id = {
            self.pad_token: self.pad_id,
            self.sos_token: self.sos_id,
            self.eos_token: self.eos_id,
        }
        self.id2char = {
            self.pad_id: self.pad_token,
            self.sos_id: self.sos_token,
            self.eos_id: self.eos_token,
        }

        # Assign IDs to charset
        current_id = 3
        for char in self.charset:
            if char not in self.char2id:
                self.char2id[char] = current_id
                self.id2char[current_id] = char
                current_id += 1

        self.vocab_size = len(self.char2id)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode a string into a list of integer token IDs.

        Args:
            text (str): Input text sequence (e.g. "ABC").
            add_special_tokens (bool): If True, prepends <SOS> and appends <EOS>. Defaults to True.

        Returns:
            List[int]: List of token IDs (e.g. [1, 3, 4, 5, 2]).
        """
        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(self.sos_id)

        for char in text:
            if char in self.char2id:
                tokens.append(self.char2id[char])
            else:
                raise ValueError(f"Character '{char}' not found in tokenizer vocabulary.")

        if add_special_tokens:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, token_ids: Union[List[int], torch.Tensor], remove_special: bool = True) -> str:
        """Decode a sequence of integer token IDs back into a string.

        Args:
            token_ids (Union[List[int], torch.Tensor]): List or 1D Tensor of token IDs.
            remove_special (bool): If True, strips special tokens (<PAD>, <SOS>, <EOS>)
                                   and stops decoding after <EOS>. Defaults to True.

        Returns:
            str: Decoded text string.
        """
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        chars: List[str] = []
        for tid in token_ids:
            if tid == self.eos_id and remove_special:
                break
            if remove_special and tid in (self.pad_id, self.sos_id, self.eos_id):
                continue
            char = self.id2char.get(tid, "")
            chars.append(char)

        return "".join(chars)

    def __len__(self) -> int:
        """Return total vocabulary size."""
        return self.vocab_size
