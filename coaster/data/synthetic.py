"""Synthetic DNA/RNA pair generation for Phase 1 training."""
from __future__ import annotations

import random

DEFAULT_DNA_LEN = 4992
DEFAULT_RNA_LEN = 150
_DNA_CHARS = ["A", "T", "G", "C"]


def random_dna(length: int = DEFAULT_DNA_LEN, rng: random.Random | None = None) -> str:
    """Generate a random DNA string of uniform A/T/G/C composition."""
    r = rng or random
    return "".join(r.choice(_DNA_CHARS) for _ in range(length))


def dna_to_rna_window(
    dna: str,
    window_start: int | None = None,
    window_len: int = DEFAULT_RNA_LEN,
    rng: random.Random | None = None,
) -> tuple[str, int]:
    """Extract a window from *dna* and apply T→U substitution.

    Returns (rna_sequence, window_start_position).
    If *window_start* is None, a position is sampled uniformly.

    Note: synthetic reads are always on the + strand; no reverse complement
    is applied. The central-3kb constraint applies only to real yeast data.
    """
    r = rng or random
    max_start = len(dna) - window_len
    if max_start < 0:
        raise ValueError(f"DNA too short ({len(dna)}) for window_len={window_len}")
    start = r.randint(0, max_start) if window_start is None else window_start
    rna = dna[start : start + window_len].replace("T", "U")
    return rna, start


def generate_synthetic_pair(
    dna_len: int = DEFAULT_DNA_LEN,
    rna_len: int = DEFAULT_RNA_LEN,
    rng: random.Random | None = None,
) -> tuple[str, str]:
    """Return a (dna_string, rna_string) synthetic training pair."""
    dna = random_dna(dna_len, rng)
    rna, _ = dna_to_rna_window(dna, window_len=rna_len, rng=rng)
    return dna, rna
