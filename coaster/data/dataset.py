"""Dataset and DataLoader utilities for synthetic DNA/RNA training data."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from coaster.data.synthetic import DEFAULT_DNA_LEN, DEFAULT_RNA_LEN
from coaster.tokenizer import RNATokenizer


class SyntheticDataset(Dataset):
    """Lazily generates synthetic (DNA, RNA-read) pairs using seeded numpy RNGs.

    DNA sequences are random uniform A/T/G/C. Each RNA target is a 150-nt
    window sampled uniformly from the DNA with T→U substitution applied.

    Items are stored as pre-tokenized int8 arrays to avoid per-call string
    allocation. Token mapping:
        DNA: A=1, T=2, G=3, C=4  (PAD=0)
        RNA: A=3, U=4, G=5, C=6  (PAD=0, BOS=1, EOS=2)
    which is consistent with DNATokenizer / RNATokenizer.
    """

    def __init__(
        self,
        num_samples: int = 50000,
        dna_len: int = DEFAULT_DNA_LEN,
        rna_len: int = DEFAULT_RNA_LEN,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.dna_len = dna_len
        self.rna_len = rna_len
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng([self.seed, idx])
        # dna_bases in {0,1,2,3} → DNA token ids {1,2,3,4}
        dna_bases = rng.integers(0, 4, size=self.dna_len, dtype=np.int8)
        start = int(rng.integers(0, self.dna_len - self.rna_len + 1))
        window = dna_bases[start : start + self.rna_len]
        # DNA base idx: 0=A,1=T,2=G,3=C
        # RNA token:    A→3, U(T)→4, G→5, C→6  ⟹ base_idx + 3
        return {
            "dna_ids": torch.from_numpy((dna_bases + 1).astype(np.int64)),
            "rna_ids": torch.from_numpy((window + 3).astype(np.int64)),
        }


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
        torch.cat([torch.tensor([BOS], dtype=torch.long), item["rna_ids"]])
        for item in batch
    ]
    rna_targets = [
        torch.cat([item["rna_ids"], torch.tensor([EOS], dtype=torch.long)])
        for item in batch
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
    # num_workers=0: required for MPS (fork-based workers conflict with MPS init)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0)
