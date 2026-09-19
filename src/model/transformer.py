"""PyTorch Seq2Seq Transformer Model from scratch with Sinusoidal Positional Encoding."""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding module (deterministic Sine/Cosine).

    Adds position information to token embeddings using sine and cosine functions
    of different frequencies as described in 'Attention is All You Need'.

    Attributes:
        pe (torch.Tensor): Registered buffer holding positional encodings of shape (1, max_len, d_model).
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """Initialize sinusoidal positional encoding table.

        Args:
            d_model (int): Hidden embedding dimension size.
            max_len (int): Maximum sequence length to precompute. Defaults to 5000.
        """
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: (1, max_len, d_model) for batch_first tensors
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to token embeddings.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, seq_len, d_model) with added positional encoding.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TinySeq2SeqTransformer(nn.Module):
    """Custom Sequence-to-Sequence Transformer implemented with raw PyTorch.

    Combines token embeddings, sinusoidal positional encoding, standard Transformer
    Encoder-Decoder blocks, and a final linear projection to vocabulary size.

    Attributes:
        vocab_size (int): Size of vocabulary.
        d_model (int): Model hidden dimension.
        src_embedding (nn.Embedding): Embedding layer for source tokens.
        tgt_embedding (nn.Embedding): Embedding layer for target tokens.
        positional_encoding (PositionalEncoding): Fixed sinusoidal positional encodings.
        transformer (nn.Transformer): Core Transformer module (batch_first=True).
        fc_out (nn.Linear): Linear output projection from d_model to vocab_size.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        pad_idx: int = 0,
        num_encoder_layers: Optional[int] = None,
        num_decoder_layers: Optional[int] = None,
    ) -> None:
        """Initialize the Tiny Seq2Seq Transformer model.

        Args:
            vocab_size (int): Total vocabulary size.
            d_model (int): Hidden dimension size. Defaults to 64.
            nhead (int): Number of attention heads. Defaults to 4.
            num_layers (int): Number of encoder and decoder layers if not specified separately. Defaults to 2.
            dim_feedforward (int): Dimension of feedforward network. Defaults to 128.
            dropout (float): Dropout probability. Defaults to 0.1.
            pad_idx (int): Padding token index. Defaults to 0.
            num_encoder_layers (Optional[int]): Explicit encoder layer count.
            num_decoder_layers (Optional[int]): Explicit decoder layer count.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx

        enc_layers = num_encoder_layers if num_encoder_layers is not None else num_layers
        dec_layers = num_decoder_layers if num_decoder_layers is not None else num_layers

        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Execute forward pass through encoder-decoder transformer.

        Args:
            src (torch.Tensor): Source token indices of shape (batch_size, src_seq_len).
            tgt (torch.Tensor): Target token indices of shape (batch_size, tgt_seq_len).
            src_mask (Optional[torch.Tensor]): Source attention mask.
            tgt_mask (Optional[torch.Tensor]): Causal target attention mask of shape (tgt_seq_len, tgt_seq_len).
            src_padding_mask (Optional[torch.Tensor]): Source key padding boolean mask of shape (batch_size, src_seq_len).
            tgt_padding_mask (Optional[torch.Tensor]): Target key padding boolean mask of shape (batch_size, tgt_seq_len).
            memory_key_padding_mask (Optional[torch.Tensor]): Memory key padding mask (usually same as src_padding_mask).

        Returns:
            torch.Tensor: Logits of shape (batch_size, tgt_seq_len, vocab_size).
        """
        if memory_key_padding_mask is None and src_padding_mask is not None:
            memory_key_padding_mask = src_padding_mask

        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return self.fc_out(out)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode source sequence into memory representations.

        Args:
            src (torch.Tensor): Source token indices of shape (batch_size, src_seq_len).
            src_mask (Optional[torch.Tensor]): Source attention mask.
            src_padding_mask (Optional[torch.Tensor]): Source padding mask (batch_size, src_seq_len).

        Returns:
            torch.Tensor: Encoded memory tensor of shape (batch_size, src_seq_len, d_model).
        """
        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(
            src=src_emb,
            mask=src_mask,
            src_key_padding_mask=src_padding_mask,
        )

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode memory representations into target vocabulary logits.

        Args:
            tgt (torch.Tensor): Target token indices of shape (batch_size, tgt_seq_len).
            memory (torch.Tensor): Memory from encoder of shape (batch_size, src_seq_len, d_model).
            tgt_mask (Optional[torch.Tensor]): Causal target mask of shape (tgt_seq_len, tgt_seq_len).
            tgt_padding_mask (Optional[torch.Tensor]): Target padding mask of shape (batch_size, tgt_seq_len).
            memory_key_padding_mask (Optional[torch.Tensor]): Memory key padding mask (batch_size, src_seq_len).

        Returns:
            torch.Tensor: Logits of shape (batch_size, tgt_seq_len, vocab_size).
        """
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc_out(out)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate upper triangular causal mask where True indicates positions to be masked out.

        Args:
            sz (int): Target sequence length.
            device (Optional[torch.device]): Target device for tensor.

        Returns:
            torch.Tensor: Causal boolean mask of shape (sz, sz) where True is masked.
        """
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)

    @staticmethod
    def generate_causal_mask(size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate upper triangular causal mask (boolean mask where True indicates masked positions).

        Args:
            size (int): Target sequence length.
            device (Optional[torch.device]): Target device for tensor.

        Returns:
            torch.Tensor: Causal boolean mask of shape (size, size).
        """
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
