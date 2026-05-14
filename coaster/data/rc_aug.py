"""Reverse-complement augmentation for the read-pointer model.

Given a window of length L and a triple (start, end, strand) with `start <= end`,
the RC-flipped sample is:

    dna'   = complement(dna)[::-1]
    start' = L - 1 - end
    end'   = L - 1 - start
    strand'= 1 - strand
"""
from __future__ import annotations

import torch

from coaster.tokenizer import DNATokenizer

_COMPLEMENT = torch.tensor(DNATokenizer.COMPLEMENT, dtype=torch.long)


def rc_dna(dna_ids: torch.Tensor) -> torch.Tensor:
    """Reverse-complement a (L,) or (B, L) tensor of DNA token ids."""
    table = _COMPLEMENT.to(dna_ids.device)
    return table[dna_ids].flip(-1)


def rc_positions(start: int, end: int, L: int) -> tuple[int, int]:
    return L - 1 - end, L - 1 - start


def rc_strand(strand: int) -> int:
    return 1 - strand
