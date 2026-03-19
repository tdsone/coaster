"""DataLoader utilities for DNA/RNA training data."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from coaster.tokenizer import RNATokenizer


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate a list of dataset items into a padded batch.

    Returns:
        dna_ids:      (B, L_dna)   int64
        rna_input:    (B, T)       int64  — BOS + rna tokens (teacher-forced input)
        rna_target:   (B, T)       int64  — rna tokens + EOS (cross-entropy target)
        rna_pad_mask: (B, T)       bool   — True where PAD
    """
    BOS, EOS, PAD = RNATokenizer.BOS, RNATokenizer.EOS, RNATokenizer.PAD

    dna_ids = torch.stack([item["dna_ids"] for item in batch])

    rna_inputs = [
        torch.cat([torch.tensor([BOS], dtype=torch.long), item["rna_ids"]]) for item in batch
    ]
    rna_targets = [
        torch.cat([item["rna_ids"], torch.tensor([EOS], dtype=torch.long)]) for item in batch
    ]

    rna_input = pad_sequence(rna_inputs, batch_first=True, padding_value=PAD)
    rna_target = pad_sequence(rna_targets, batch_first=True, padding_value=PAD)
    rna_pad_mask = rna_input == PAD

    return {
        "dna_ids": dna_ids,
        "rna_input": rna_input,
        "rna_target": rna_target,
        "rna_pad_mask": rna_pad_mask,
    }


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
        num_workers=4, persistent_workers=True, pin_memory=True,
    )
