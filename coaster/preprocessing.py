"""Pure-Python helpers for read preprocessing (no Modal dependency).

Used by scripts/extract_reads_modal.py and tests/test_preprocessing.py.
"""
from __future__ import annotations

_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")

MIN_OVERLAP = 10  # minimum nt overlap to call a merged pair


def revcomp(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(_RC_TABLE)[::-1]


def to_gene_sense(query_sequence: str, gene_strand: str) -> str:
    """Convert a pysam query_sequence to the gene-sense (5'→3') direction.

    pysam always returns query_sequence in the + genomic strand direction
    (reverse-strand reads are stored as their RC in the BAM). For - strand
    genes, gene-sense is the - strand, so we reverse-complement.
    """
    if gene_strand == "-":
        return revcomp(query_sequence)
    return query_sequence


def merge_pair(r2_sense: str, r1_sense: str, min_overlap: int = MIN_OVERLAP) -> str:
    """Merge R2 (5' end) and R1 (3' end), both in gene-sense direction.

    Finds the longest suffix of r2_sense that matches a prefix of r1_sense.
    Returns the merged insert if overlap >= min_overlap, else r2_sense alone.
    """
    max_overlap = min(len(r2_sense), len(r1_sense))
    for overlap in range(max_overlap, min_overlap - 1, -1):
        if r2_sense[-overlap:] == r1_sense[:overlap]:
            return r2_sense + r1_sense[overlap:]
    return r2_sense
