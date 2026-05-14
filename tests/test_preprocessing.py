"""Tests for read-preprocessing helpers (revcomp, to_gene_sense, merge_pair)."""
from coaster.preprocessing import merge_pair, revcomp, to_gene_sense


# ── revcomp / to_gene_sense ──────────────────────────────────────────────────

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


# ── merge_pair ───────────────────────────────────────────────────────────────

def test_merge_pair_known_overlap():
    merged = merge_pair("AAAAAABBBBB", "BBBBBCCCCCC", min_overlap=5)
    assert merged == "AAAAAABBBBBCCCCCC"


def test_merge_pair_no_overlap_returns_r2():
    r2 = "AAAAAAAAAA"
    assert merge_pair(r2, "TTTTTTTTTT", min_overlap=5) == r2


def test_merge_pair_overlap_below_min_returns_r2():
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
    r2 = "AAAAABBBBBCCCCC"
    r1 = "BBBBBCCCCC"
    assert merge_pair(r2, r1, min_overlap=5) == r2


def test_merge_pair_realistic_fragment():
    fragment = "A" * 49 + "G" * 102 + "T" * 49
    r2 = fragment[:151]
    r1 = fragment[-151:]
    assert merge_pair(r2, r1, min_overlap=10) == fragment


# Integration tests against real BAM-extracted parquets live in test_data.py
# (ReadsDataset position mapping) once a fresh extract_reads_modal.py run has
# produced the new (strand, ref_start, ref_end) columns.
