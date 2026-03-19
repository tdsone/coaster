"""Tests for the data preprocessing pipeline.

Covers:
  1. revcomp() and to_gene_sense() — unit tests
  2. merge_pair() — unit tests
  3. Strand orientation validation against real data
  4. compute_coverage() — unit tests
"""
from pathlib import Path

import numpy as np
import pytest

from coaster.preprocessing import MIN_OVERLAP, merge_pair, revcomp, to_gene_sense
from evals.compute_coverage import compute_coverage

DATA_DIR = Path(__file__).parent.parent / "data"
try:
    import pyarrow  # noqa: F401
    _pyarrow_available = True
except ImportError:
    _pyarrow_available = False

_data_available = (
    _pyarrow_available
    and (DATA_DIR / "samples_yeast.parquet").exists()
    and (DATA_DIR / "reads.parquet").exists()
)


# ── 1. revcomp and to_gene_sense ──────────────────────────────────────────────

def test_revcomp_known_sequence():
    assert revcomp("ATCG") == "CGAT"


def test_revcomp_with_n():
    assert revcomp("ATCGN") == "NCGAT"


def test_revcomp_involution():
    assert revcomp(revcomp("AATTCCGGNN")) == "AATTCCGGNN"


def test_revcomp_empty():
    assert revcomp("") == ""


def test_to_gene_sense_plus_unchanged():
    seq = "AATCGGCAT"
    assert to_gene_sense(seq, "+") == seq


def test_to_gene_sense_minus_reverse_complements():
    seq = "AATCGGCAT"
    assert to_gene_sense(seq, "-") == revcomp(seq)


def test_to_gene_sense_minus_is_involution():
    seq = "AATTCCGGNN"
    assert to_gene_sense(to_gene_sense(seq, "-"), "-") == seq


# ── 2. merge_pair ─────────────────────────────────────────────────────────────

def test_merge_pair_known_overlap():
    # R2 = AAAAAABBBBB, R1 = BBBBBCCCCCC; overlap = BBBBB (5 bases)
    merged = merge_pair("AAAAAABBBBB", "BBBBBCCCCCC", min_overlap=5)
    assert merged == "AAAAAABBBBBCCCCCC"


def test_merge_pair_no_overlap_returns_r2():
    r2 = "AAAAAAAAAA"
    assert merge_pair(r2, "TTTTTTTTTT", min_overlap=5) == r2


def test_merge_pair_overlap_below_min_returns_r2():
    # 4-base overlap but min_overlap=5 → no merge
    r2 = "AAAAAACCCC"
    assert merge_pair(r2, "CCCCTTTTTT", min_overlap=5) == r2


def test_merge_pair_merged_starts_with_full_r2():
    r2 = "AACCGGTTAA"
    r1 = "TTAACCGGTT"
    merged = merge_pair(r2, r1, min_overlap=4)
    assert merged.startswith(r2)


def test_merge_pair_merged_ends_with_full_r1():
    r2 = "AACCGGTTAA"
    r1 = "TTAACCGGTT"
    merged = merge_pair(r2, r1, min_overlap=4)
    assert merged.endswith(r1)


def test_merge_pair_r1_fully_contained_in_r2():
    # Overlap = len(R1) → merged == R2
    r2 = "AAAAABBBBBCCCCC"
    r1 = "BBBBBCCCCC"
    assert merge_pair(r2, r1, min_overlap=5) == r2


def test_merge_pair_realistic_fragment():
    # 200 bp fragment, 151 bp reads overlapping by 102 bp
    fragment = "A" * 49 + "G" * 102 + "T" * 49
    r2 = fragment[:151]
    r1 = fragment[-151:]
    assert merge_pair(r2, r1, min_overlap=10) == fragment


# ── 3. Strand orientation using real data ─────────────────────────────────────

@pytest.mark.skipif(not _data_available, reason="data parquets not in data/")
@pytest.mark.parametrize("strand", ["+", "-"])
def test_gene_sense_reads_match_input_sequence_forward(strand):
    """After to_gene_sense(), reads must match input_sequence forward.

    reads.parquet stores pysam query_sequence (always + genomic strand) with
    T→U applied. Reversing T→U and calling to_gene_sense() should orient every
    read in the gene-sense direction, i.e. they should be a forward substring
    of input_sequence (within sequencing-error limits).

    Fails when to_gene_sense() is wrong — e.g. if it doesn't RC for '-' strand
    genes, those reads remain in the + genomic / gene-antisense direction and
    match RC of input_sequence instead of forward, collapsing the forward match
    rate to ~0 % and tripping the assertion.
    """
    import pandas as pd

    samples = pd.read_parquet(DATA_DIR / "samples_yeast.parquet")
    reads_df = pd.read_parquet(DATA_DIR / "reads.parquet")

    strand_samples = samples[samples["strand"] == strand]
    read_set = set(reads_df["sample_idx"])
    test_idxs = [i for i in strand_samples.index if i in read_set][:5]
    assert len(test_idxs) >= 3, f"Not enough {strand!r}-strand samples with reads"

    total = matched = 0
    for idx in test_idxs:
        ref = strand_samples.loc[idx, "input_sequence"]
        for r in reads_df[reads_df["sample_idx"] == idx]["read_seq"].tolist()[:100]:
            dna = r.replace("U", "T")
            gene_sense = to_gene_sense(dna, strand)
            total += 1
            if gene_sense in ref:
                matched += 1

    match_rate = matched / total
    assert match_rate > 0.5, (
        f"{strand!r}-strand forward match rate is {match_rate:.1%} ({matched}/{total}). "
        f"to_gene_sense() may be handling {strand!r} strand incorrectly."
    )


# ── 4. compute_coverage ───────────────────────────────────────────────────────

def test_coverage_forward_match():
    ref = "AAATTTGGGCCC"
    cov = compute_coverage(["TTTGGG"], ref)
    assert cov.shape == (len(ref),)
    assert cov[3:9].sum() == 6   # positions 3–8 covered once each
    assert cov[:3].sum() == 0
    assert cov[9:].sum() == 0


def test_coverage_rc_match():
    ref = "AAATTTGGGCCC"
    # RC of "TTTGGG" is "CCCAAA", which is at position 6 in ref
    cov = compute_coverage(["CCCAAA"], ref)
    assert cov.sum() > 0


def test_coverage_rna_u_to_t_conversion():
    ref = "AAATTTGGGCCC"
    cov = compute_coverage(["UUUGGG"], ref)  # RNA form of "TTTGGG"
    assert cov[3:9].sum() == 6


def test_coverage_no_match_gives_zero():
    ref = "AAATTTGGGCCC"
    cov = compute_coverage(["GGGGGGGGGGGG"], ref)
    assert cov.sum() == 0


def test_coverage_multiple_reads():
    ref = "AAATTTGGGCCC"
    cov = compute_coverage(["AAATTT", "GGGCCC"], ref)
    assert cov[:6].sum() == 6
    assert cov[6:].sum() == 6


def test_coverage_empty_reads_gives_zero():
    ref = "ACGT" * 10
    cov = compute_coverage([], ref)
    assert cov.shape == (len(ref),)
    assert cov.sum() == 0
