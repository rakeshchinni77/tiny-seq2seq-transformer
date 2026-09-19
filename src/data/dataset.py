"""PyTorch Dataset and Collate function for sequence reversal."""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Union
from src.data.tokenizer import CharacterTokenizer, PAD_ID


class SequenceReversalDataset(Dataset):
    """PyTorch Dataset for loading source-target string pairs from a CSV file.

    Attributes:
        data (pd.DataFrame): DataFrame containing 'source' and 'target' columns.
        tokenizer (CharacterTokenizer): Tokenizer instance to encode strings.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: Optional[CharacterTokenizer] = None,
    ) -> None:
        """Initialize SequenceReversalDataset.

        Args:
            csv_path (str): Path to CSV file containing 'source' and 'target' columns.
            tokenizer (Optional[CharacterTokenizer]): Tokenizer instance. Defaults to default CharacterTokenizer.
        """
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer if tokenizer is not None else CharacterTokenizer()

    def __len__(self) -> int:
        """Return total number of samples in dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        """Fetch encoded source and target token ID lists for a given index.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[List[int], List[int]]: (source_token_ids, target_token_ids)
        """
        row = self.data.iloc[idx]
        src_text = str(row["source"])
        tgt_text = str(row["target"])

        src_ids = self.tokenizer.encode(src_text, add_special_tokens=True)
        tgt_ids = self.tokenizer.encode(tgt_text, add_special_tokens=True)

        return src_ids, tgt_ids


def collate_fn(
    batch: List[Tuple[List[int], List[int]]],
    pad_id: int = PAD_ID,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate and pad a batch of variable-length source and target token sequences.

    Args:
        batch (List[Tuple[List[int], List[int]]]): List of (src_ids, tgt_ids) tuples.
        pad_id (int): Integer token ID used for padding. Defaults to PAD_ID (0).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - src_tensor: LongTensor of shape (batch_size, max_src_len)
            - tgt_tensor: LongTensor of shape (batch_size, max_tgt_len)
    """
    src_seqs = [item[0] for item in batch]
    tgt_seqs = [item[1] for item in batch]

    max_src_len = max(len(seq) for seq in src_seqs)
    max_tgt_len = max(len(seq) for seq in tgt_seqs)

    batch_size = len(batch)

    src_padded = torch.full((batch_size, max_src_len), pad_id, dtype=torch.long)
    tgt_padded = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)

    for i, (s_seq, t_seq) in enumerate(zip(src_seqs, tgt_seqs)):
        src_padded[i, : len(s_seq)] = torch.tensor(s_seq, dtype=torch.long)
        tgt_padded[i, : len(t_seq)] = torch.tensor(t_seq, dtype=torch.long)

    return src_padded, tgt_padded
