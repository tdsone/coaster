"""Dataloader for the pointer-factorized read model.

ReadsDataset yields per-read samples; the collate combines them with:

  - MLM masking per sample (independent Bernoulli with prob `p_mlm`).
  - "has_reads" flag per sample (independent Bernoulli with prob `p_reads`).
    When False, the read fields are still present but will be ignored by the
    training step (so we can keep tensor shapes uniform).
  - Reverse-complement augmentation per sample (Bernoulli `rc_aug_prob`).

The three §4 populations emerge from setting (p_mlm, p_reads).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from coaster.data.rc_aug import rc_dna, rc_positions, rc_strand
from coaster.tokenizer import DNATokenizer

_TOK = DNATokenizer()
PAD, MASK = _TOK.PAD, _TOK.MASK


# ---------------------------------------------------------------------------
def _trim_or_pad_dna(seq: str, target_len: int) -> tuple[str, int]:
    """Center-crop or right-pad with N. Returns (trimmed_seq, offset_into_original)
    where offset is the index in `seq` of the first character of the trimmed view.
    For padded outputs the offset is 0 (no shift)."""
    n = len(seq)
    if n >= target_len:
        offset = (n - target_len) // 2
        return seq[offset : offset + target_len], offset
    return seq + "N" * (target_len - n), 0


# ---------------------------------------------------------------------------
class ReadsDataset(Dataset):
    """One sample == one aligned read with its window.

    Args:
        samples_path: parquet of windows with columns
            ['input_sequence', 'start_seq', 'fold', ...].
        reads_path:   parquet of reads with columns
            ['sample_idx', 'read_seq', 'strand', 'ref_start', 'ref_end'].
        fold:         'train' / 'val' / 'test' / None.
        dna_len:      target DNA window length passed to the encoder.

    Position math:
        window_offset = start_seq + crop_offset       # ref coord of dna_ids[0]
        local_start   = ref_start - window_offset     # 0-based in the trimmed window
        local_end     = ref_end   - window_offset

    Reads whose span falls outside the trimmed window are filtered.
    """

    def __init__(
        self,
        samples_path: str | Path,
        reads_path: str | Path,
        fold: str | None = "train",
        dna_len: int = 4992,
    ) -> None:
        samples = pd.read_parquet(samples_path)
        reads = pd.read_parquet(reads_path)

        required_cols = {"strand", "ref_start", "ref_end"}
        missing = required_cols - set(reads.columns)
        if missing:
            raise ValueError(
                f"reads.parquet is missing columns {missing}. Re-run extract_reads_modal.py."
            )

        if fold is not None:
            fold_idx = set(samples.index[samples["fold"] == fold])
            samples = samples[samples["fold"] == fold]
            reads = reads[reads["sample_idx"].isin(fold_idx)]

        # Pre-trim DNA windows and remember the crop offset per window.
        self.dna_len = dna_len
        windows: dict[int, tuple[str, int]] = {}
        for idx, row in samples.iterrows():
            trimmed, crop_off = _trim_or_pad_dna(row["input_sequence"], dna_len)
            windows[int(idx)] = (trimmed, crop_off)
        self.windows = windows
        # Reference coordinate of dna_ids[0] per sample_idx.
        self.window_origin: dict[int, int] = {
            int(idx): int(row["start_seq"]) + windows[int(idx)][1]
            for idx, row in samples.iterrows()
        }

        # Compute local positions and filter reads outside the trimmed window.
        sample_idxs = reads["sample_idx"].to_numpy()
        ref_starts = reads["ref_start"].to_numpy()
        ref_ends = reads["ref_end"].to_numpy()
        strands = reads["strand"].to_numpy()

        # Map sample_idx -> window_origin via lookup array (vectorized).
        origin = np.array(
            [self.window_origin.get(int(si), -10**9) for si in sample_idxs], dtype=np.int64
        )
        local_starts = ref_starts.astype(np.int64) - origin
        local_ends = ref_ends.astype(np.int64) - origin

        keep = (local_starts >= 0) & (local_ends < dna_len) & (local_starts <= local_ends)
        self.sample_idxs = sample_idxs[keep]
        self.local_starts = local_starts[keep].astype(np.int64)
        self.local_ends = local_ends[keep].astype(np.int64)
        self.strands = strands[keep].astype(np.int8)

        print(
            f"ReadsDataset [{fold}]: {len(self.sample_idxs):,} reads kept across "
            f"{len(self.windows)} windows "
            f"(dropped {(~keep).sum():,} reads outside the trimmed window)"
        )

    def __len__(self) -> int:
        return int(len(self.sample_idxs))

    def __getitem__(self, idx: int) -> dict:
        sample_idx = int(self.sample_idxs[idx])
        dna_seq, _ = self.windows[sample_idx]
        return {
            "dna_ids": torch.tensor(_TOK.encode(dna_seq), dtype=torch.long),
            "start": int(self.local_starts[idx]),
            "end": int(self.local_ends[idx]),
            "strand": int(self.strands[idx]),
        }


# ---------------------------------------------------------------------------
def _mask_dna(dna_ids: torch.Tensor, prob: float, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace ~`prob` fraction of non-PAD positions with MASK.
    Returns (masked_ids, bool mask of replaced positions)."""
    L = dna_ids.size(0)
    rand = torch.rand(L, generator=rng)
    valid = dna_ids != PAD
    mask = (rand < prob) & valid
    masked = dna_ids.clone()
    masked[mask] = MASK
    return masked, mask


# ---------------------------------------------------------------------------
def make_collate_fn(
    mlm_mask_prob: float = 0.15,
    p_mlm: float = 0.6,
    p_reads: float = 0.6,
    rc_aug_prob: float = 0.5,
    seed: int | None = None,
):
    """Returns a collate_fn that builds a batched dict for training_step().

    Output keys (all torch tensors):
      dna_ids:       (B, L) int64, possibly with MASK at some positions.
      mlm_mask:      (B, L) bool — True where the MLM head should compute loss.
      target_nucl:   (B, L) int64 — original (pre-mask) nucleotide ids.
      has_reads:     (B,)   bool — True if this sample contributes read losses.
      strand_true:   (B,)   int64 (valid wherever has_reads is True).
      start_true:    (B,)   int64.
      end_true:      (B,)   int64.

    A seeded `torch.Generator` is used so the same dataset row produces deterministic
    masks across workers when `seed` is set (useful for tests).
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        B = len(batch)
        L = batch[0]["dna_ids"].size(0)
        dna = torch.stack([item["dna_ids"] for item in batch])           # (B, L)
        target_nucl = dna.clone()

        # Per-sample population assignment.
        apply_mlm = torch.rand(B, generator=rng) < p_mlm
        apply_reads = torch.rand(B, generator=rng) < p_reads
        apply_rc = torch.rand(B, generator=rng) < rc_aug_prob

        strand = torch.tensor([item["strand"] for item in batch], dtype=torch.long)
        start = torch.tensor([item["start"] for item in batch], dtype=torch.long)
        end = torch.tensor([item["end"] for item in batch], dtype=torch.long)

        # RC augmentation per sample (touches DNA, positions, and strand).
        for b in range(B):
            if apply_rc[b]:
                dna[b] = rc_dna(dna[b])
                target_nucl[b] = rc_dna(target_nucl[b])
                s_new, e_new = rc_positions(int(start[b]), int(end[b]), L)
                start[b] = s_new
                end[b] = e_new
                strand[b] = rc_strand(int(strand[b]))

        # MLM masking per sample.
        mlm_mask = torch.zeros(B, L, dtype=torch.bool)
        for b in range(B):
            if apply_mlm[b]:
                dna[b], mlm_mask[b] = _mask_dna(dna[b], mlm_mask_prob, rng)

        return {
            "dna_ids": dna,
            "mlm_mask": mlm_mask,
            "target_nucl": target_nucl,
            "has_reads": apply_reads,
            "strand_true": strand,
            "start_true": start,
            "end_true": end,
        }

    return _collate


# ---------------------------------------------------------------------------
def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    collate_fn,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=shuffle,
    )
