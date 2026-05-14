"""Tests for ReadsDataset, RC augmentation, and the collate's task-mixing logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from coaster.data.dataset import ReadsDataset, _trim_or_pad_dna, make_collate_fn
from coaster.data.rc_aug import rc_dna, rc_positions, rc_strand
from coaster.tokenizer import DNATokenizer

TOK = DNATokenizer()


# ── RC augmentation ─────────────────────────────────────────────────────────

def test_rc_dna_simple():
    dna = torch.tensor(TOK.encode("ACGT"))    # [A, C, G, T]
    rc = rc_dna(dna)
    assert rc.tolist() == TOK.encode("ACGT")  # ACGT -> ACGT (palindrome)


def test_rc_dna_non_palindrome():
    dna = torch.tensor(TOK.encode("AAAT"))
    rc = rc_dna(dna)
    # complement(AAAT) = TTTA; reversed = ATTT
    assert rc.tolist() == TOK.encode("ATTT")


def test_rc_dna_involution():
    dna = torch.tensor(TOK.encode("ACGTNACG"))
    assert torch.equal(rc_dna(rc_dna(dna)), dna)


def test_rc_dna_preserves_mask_and_n():
    dna = torch.tensor([TOK.MASK, TOK.N, TOK.A, TOK.T])
    rc = rc_dna(dna)
    # complement: MASK→MASK, N→N, A→T, T→A; reversed: A T N MASK
    assert rc.tolist() == [TOK.A, TOK.T, TOK.N, TOK.MASK]


def test_rc_positions_basic():
    s, e = rc_positions(start=3, end=7, L=10)
    assert (s, e) == (2, 6)


def test_rc_positions_full_window():
    s, e = rc_positions(start=0, end=9, L=10)
    assert (s, e) == (0, 9)


def test_rc_positions_invariant_under_rc_dna():
    """If we RC the DNA *and* RC the positions, the covered substring is identical."""
    L = 20
    seq = "ACGTACGTACGTACGTACGT"
    assert len(seq) == L
    dna = torch.tensor(TOK.encode(seq))
    start, end = 4, 11
    original = dna[start : end + 1]

    rc = rc_dna(dna)
    s_new, e_new = rc_positions(start, end, L)
    new = rc[s_new : e_new + 1]
    # The new slice should equal the RC of the original slice.
    assert torch.equal(new, rc_dna(original))


def test_rc_strand():
    assert rc_strand(0) == 1
    assert rc_strand(1) == 0


# ── DNA cropping helper ─────────────────────────────────────────────────────

def test_trim_pad_no_change():
    seq = "ACGT" * 4
    out, off = _trim_or_pad_dna(seq, 16)
    assert out == seq and off == 0


def test_trim_centred_crop():
    seq = "X" * 100
    out, off = _trim_or_pad_dna(seq, 80)
    assert len(out) == 80
    assert off == 10            # (100 - 80) / 2 centred


def test_trim_pads_with_N():
    seq = "ACGT"
    out, off = _trim_or_pad_dna(seq, 10)
    assert out == "ACGTNNNNNN" and off == 0


# ── ReadsDataset position mapping (synthetic parquets) ──────────────────────

def _write_synthetic_parquets(tmp_path):
    samples = pd.DataFrame(
        [
            {"input_sequence": "A" * 100, "start_seq": 1000, "fold": "train"},
            {"input_sequence": "C" * 100, "start_seq": 2000, "fold": "train"},
        ]
    )
    # Reads. Window 0 origin = 1000; window 1 origin = 2000.
    reads = pd.DataFrame(
        [
            {"sample_idx": 0, "read_seq": "AAA", "strand": 0, "ref_start": 1010, "ref_end": 1019},
            {"sample_idx": 0, "read_seq": "AAA", "strand": 1, "ref_start": 1050, "ref_end": 1060},
            # Out of window — should be filtered:
            {"sample_idx": 0, "read_seq": "AAA", "strand": 0, "ref_start": 900,  "ref_end": 910},
            {"sample_idx": 1, "read_seq": "CCC", "strand": 0, "ref_start": 2005, "ref_end": 2009},
        ]
    )
    s_path = tmp_path / "samples.parquet"
    r_path = tmp_path / "reads.parquet"
    samples.to_parquet(s_path)
    reads.to_parquet(r_path)
    return s_path, r_path


def test_reads_dataset_local_position_mapping(tmp_path):
    s_path, r_path = _write_synthetic_parquets(tmp_path)
    ds = ReadsDataset(s_path, r_path, fold="train", dna_len=100)
    # The out-of-window read should be dropped.
    assert len(ds) == 3
    # First read: ref_start=1010, origin=1000 -> local 10..19.
    item = ds[0]
    assert item["dna_ids"].shape == (100,)
    assert item["start"] == 10 and item["end"] == 19
    assert item["strand"] == 0


def test_reads_dataset_requires_position_columns(tmp_path):
    samples = pd.DataFrame([{"input_sequence": "A" * 50, "start_seq": 0, "fold": "train"}])
    reads = pd.DataFrame([{"sample_idx": 0, "read_seq": "AAA"}])  # missing required cols
    s_path = tmp_path / "samples.parquet"
    r_path = tmp_path / "reads.parquet"
    samples.to_parquet(s_path)
    reads.to_parquet(r_path)
    with pytest.raises(ValueError, match="missing columns"):
        ReadsDataset(s_path, r_path, fold="train", dna_len=50)


# ── Collate / task mixing ───────────────────────────────────────────────────

def _make_items(B: int, L: int = 32) -> list[dict]:
    items = []
    for i in range(B):
        items.append({
            "dna_ids": torch.tensor(TOK.encode(("ACGT" * (L // 4))[:L])),
            "start": i % L,
            "end": (i % L) + 3,
            "strand": i % 2,
        })
    return items


def test_collate_shapes_and_types():
    collate = make_collate_fn(p_mlm=0.5, p_reads=0.5, rc_aug_prob=0.0, seed=0)
    batch = collate(_make_items(8, L=32))
    for key in ("dna_ids", "mlm_mask", "target_nucl"):
        assert batch[key].shape == (8, 32)
    for key in ("has_reads", "strand_true", "start_true", "end_true"):
        assert batch[key].shape == (8,)
    assert batch["mlm_mask"].dtype == torch.bool
    assert batch["has_reads"].dtype == torch.bool


def test_collate_mlm_only_population():
    """p_mlm=1, p_reads=0 → every sample is MLM-only."""
    collate = make_collate_fn(p_mlm=1.0, p_reads=0.0, rc_aug_prob=0.0, seed=0)
    batch = collate(_make_items(16))
    assert batch["mlm_mask"].any()
    assert not batch["has_reads"].any()


def test_collate_read_only_population():
    """p_mlm=0, p_reads=1 → no masking, every sample contributes reads."""
    collate = make_collate_fn(p_mlm=0.0, p_reads=1.0, rc_aug_prob=0.0, seed=0)
    batch = collate(_make_items(16))
    assert not batch["mlm_mask"].any()
    assert batch["has_reads"].all()
    # DNA should be unmodified (no MASK tokens).
    assert (batch["dna_ids"] != TOK.MASK).all()


def test_collate_mlm_masks_with_mask_token():
    """When MLM applies, the masked positions in dna_ids should be MASK and
    target_nucl should hold the original value."""
    collate = make_collate_fn(p_mlm=1.0, p_reads=0.0, mlm_mask_prob=0.5, rc_aug_prob=0.0, seed=0)
    batch = collate(_make_items(4))
    mask = batch["mlm_mask"]
    # At every masked position, dna_ids == MASK and target_nucl != MASK.
    assert (batch["dna_ids"][mask] == TOK.MASK).all()
    assert (batch["target_nucl"][mask] != TOK.MASK).all()
    # Unmasked positions are untouched.
    assert (batch["dna_ids"][~mask] == batch["target_nucl"][~mask]).all()


def test_collate_rc_aug_flips_consistently():
    """With rc_aug_prob=1 every sample is RC'd: strand flips, start/end remap, DNA is RC of original."""
    items = _make_items(6, L=32)
    # Capture original values.
    orig_dna = torch.stack([it["dna_ids"] for it in items])
    orig_starts = torch.tensor([it["start"] for it in items])
    orig_ends = torch.tensor([it["end"] for it in items])
    orig_strand = torch.tensor([it["strand"] for it in items])

    collate = make_collate_fn(p_mlm=0.0, p_reads=1.0, rc_aug_prob=1.0, seed=42)
    batch = collate(items)
    L = 32

    expected_starts = L - 1 - orig_ends
    expected_ends = L - 1 - orig_starts
    expected_strand = 1 - orig_strand

    assert torch.equal(batch["start_true"], expected_starts)
    assert torch.equal(batch["end_true"], expected_ends)
    assert torch.equal(batch["strand_true"], expected_strand)
    # DNA flipped via the same complement table.
    assert torch.equal(batch["dna_ids"], rc_dna(orig_dna))


def test_collate_population_ratios_are_close_to_target():
    """Big batch with default 0.6/0.6 → roughly the right populations."""
    collate = make_collate_fn(p_mlm=0.6, p_reads=0.6, rc_aug_prob=0.0, seed=123)
    batch = collate(_make_items(2048, L=16))
    p_mlm = batch["mlm_mask"].any(dim=-1).float().mean().item()
    p_reads = batch["has_reads"].float().mean().item()
    assert 0.55 < p_mlm < 0.65
    assert 0.55 < p_reads < 0.65
