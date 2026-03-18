"""Real S. cerevisiae RNA-seq dataset from pre-processed BAM reads.

Expects two parquet files produced by scripts/extract_reads_modal.py:
  - samples_yeast.parquet: one row per genomic window (DNA sequence + metadata)
  - reads.parquet:         one row per extracted read (sample_idx, read_seq)

The interface matches SyntheticDataset: __getitem__ returns
  {"dna_ids": LongTensor(dna_len,), "rna_ids": LongTensor(read_len,)}
so the same collate_fn and training loop work unchanged.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from coaster.tokenizer import DNATokenizer, RNATokenizer

_DNA_TOK = DNATokenizer()
_RNA_TOK = RNATokenizer()


def _trim_or_pad_dna(seq: str, target_len: int) -> str:
    """Center-crop if longer than target; right-pad with N if shorter."""
    n = len(seq)
    if n >= target_len:
        start = (n - target_len) // 2
        return seq[start : start + target_len]
    return seq + "N" * (target_len - n)


class RealRNADataset(Dataset):
    """(DNA window, RNA read) pairs from real S. cerevisiae RNA-seq.

    Args:
        samples_path: path to samples_yeast.parquet
        reads_path:   path to reads.parquet
        fold:         'train', 'val', 'test', or None (load all folds)
        dna_len:      target DNA length passed to the encoder (trim/pad as needed)
        min_read_len: drop reads shorter than this (nt)
        max_read_len: drop reads longer than this (nt)
    """

    def __init__(
        self,
        samples_path: str | Path,
        reads_path: str | Path,
        fold: str | None = "train",
        dna_len: int = 5000,
        min_read_len: int = 50,
        max_read_len: int = 200,
        max_reads_per_window: int | None = None,
        seed: int = 42,
    ) -> None:
        samples = pd.read_parquet(samples_path)
        reads = pd.read_parquet(reads_path)

        if fold is not None:
            fold_idx = set(samples.index[samples["fold"] == fold])
            reads = reads[reads["sample_idx"].isin(fold_idx)]
            samples = samples[samples["fold"] == fold]

        # Drop reads outside the acceptable length range
        read_lens = reads["read_seq"].str.len()
        reads = reads[(read_lens >= min_read_len) & (read_lens <= max_read_len)]

        # Subsample reads per window if requested
        if max_reads_per_window is not None:
            reads = (
                reads.groupby("sample_idx", sort=False)
                .apply(lambda g: g.sample(min(len(g), max_reads_per_window), random_state=seed))
                .reset_index(drop=True)
            )

        # Build a fast dict lookup: sample_idx (int) → dna string
        self.dna_lookup: dict[int, str] = samples["input_sequence"].to_dict()
        # Flat list of records; shuffled order handled by DataLoader
        self.reads: list[dict] = reads[["sample_idx", "read_seq"]].to_dict("records")
        self.dna_len = dna_len

        n_windows = len(samples)
        n_reads = len(self.reads)
        print(
            f"RealRNADataset [{fold}]: {n_reads:,} reads across {n_windows} windows "
            f"(avg {n_reads / max(n_windows, 1):.0f} reads/window)"
        )

    def __len__(self) -> int:
        return len(self.reads)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.reads[idx]
        dna = _trim_or_pad_dna(self.dna_lookup[record["sample_idx"]], self.dna_len)
        rna = record["read_seq"]  # already T→U from extraction script

        return {
            "dna_ids": torch.tensor(_DNA_TOK.encode(dna), dtype=torch.long),
            # No BOS/EOS here — collate_fn adds them, matching SyntheticDataset
            "rna_ids": torch.tensor(_RNA_TOK.encode(rna, add_special=False), dtype=torch.long),
        }
