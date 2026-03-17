import random

import torch
import pytest

from coaster.data.synthetic import (
    random_dna,
    dna_to_rna_window,
    generate_synthetic_pair,
    DEFAULT_DNA_LEN,
    DEFAULT_RNA_LEN,
)
from coaster.data.dataset import SyntheticDataset, collate_fn
from coaster.tokenizer import RNATokenizer


def test_random_dna_length():
    assert len(random_dna(100)) == 100


def test_random_dna_chars():
    dna = random_dna(500)
    assert all(c in "ATGC" for c in dna)


def test_random_dna_seeded():
    r = random.Random(42)
    assert random_dna(50, r) == random_dna(50, random.Random(42))


def test_dna_to_rna_window_length():
    dna = random_dna(DEFAULT_DNA_LEN)
    rna, _ = dna_to_rna_window(dna)
    assert len(rna) == DEFAULT_RNA_LEN


def test_dna_to_rna_no_t():
    dna = random_dna(DEFAULT_DNA_LEN)
    rna, _ = dna_to_rna_window(dna)
    assert "T" not in rna
    assert all(c in "AUGC" for c in rna)


def test_dna_to_rna_fixed_start():
    dna = "A" * 300
    rna, start = dna_to_rna_window(dna, window_start=10)
    assert start == 10
    assert rna == "A" * DEFAULT_RNA_LEN


def test_dna_to_rna_t_converted():
    dna = "T" * 300
    rna, _ = dna_to_rna_window(dna)
    assert rna == "U" * DEFAULT_RNA_LEN


def test_dna_too_short_raises():
    with pytest.raises(ValueError):
        dna_to_rna_window("ATGC", window_len=10)


def test_generate_pair_lengths():
    dna, rna = generate_synthetic_pair()
    assert len(dna) == DEFAULT_DNA_LEN
    assert len(rna) == DEFAULT_RNA_LEN


def test_generate_pair_rna_is_rna():
    _, rna = generate_synthetic_pair()
    assert all(c in "AUGC" for c in rna)


def test_dataset_len():
    ds = SyntheticDataset(num_samples=20)
    assert len(ds) == 20


def test_dataset_getitem_keys():
    ds = SyntheticDataset(num_samples=5)
    item = ds[0]
    assert "dna_ids" in item and "rna_ids" in item


def test_dataset_getitem_shapes():
    ds = SyntheticDataset(num_samples=5)
    item = ds[0]
    assert item["dna_ids"].shape == (DEFAULT_DNA_LEN,)
    assert item["rna_ids"].shape == (DEFAULT_RNA_LEN,)


def test_dataset_dtypes():
    ds = SyntheticDataset(num_samples=3)
    item = ds[0]
    assert item["dna_ids"].dtype == torch.long
    assert item["rna_ids"].dtype == torch.long


def test_dataset_dna_token_range():
    ds = SyntheticDataset(num_samples=5)
    item = ds[0]
    assert item["dna_ids"].min() >= 1
    assert item["dna_ids"].max() <= 4


def test_dataset_rna_token_range():
    ds = SyntheticDataset(num_samples=5)
    item = ds[0]
    # RNA: A=3, U=4, G=5, C=6
    assert item["rna_ids"].min() >= 3
    assert item["rna_ids"].max() <= 6


def test_dataset_reproducible():
    ds1 = SyntheticDataset(num_samples=5, seed=0)
    ds2 = SyntheticDataset(num_samples=5, seed=0)
    assert torch.equal(ds1[0]["dna_ids"], ds2[0]["dna_ids"])


def test_dataset_different_seeds():
    ds1 = SyntheticDataset(num_samples=5, seed=0)
    ds2 = SyntheticDataset(num_samples=5, seed=99)
    assert not torch.equal(ds1[0]["dna_ids"], ds2[0]["dna_ids"])


def test_collate_shapes():
    ds = SyntheticDataset(num_samples=4)
    batch = [ds[i] for i in range(4)]
    out = collate_fn(batch)
    T = DEFAULT_RNA_LEN + 1  # BOS/EOS on opposite ends, same length
    assert out["dna_ids"].shape == (4, DEFAULT_DNA_LEN)
    assert out["rna_input"].shape == (4, T)
    assert out["rna_target"].shape == (4, T)
    assert out["rna_pad_mask"].shape == (4, T)


def test_collate_bos():
    ds = SyntheticDataset(num_samples=2)
    batch = [ds[i] for i in range(2)]
    out = collate_fn(batch)
    assert (out["rna_input"][:, 0] == RNATokenizer.BOS).all()


def test_collate_eos():
    ds = SyntheticDataset(num_samples=2)
    batch = [ds[i] for i in range(2)]
    out = collate_fn(batch)
    assert (out["rna_target"][:, -1] == RNATokenizer.EOS).all()


def test_collate_dtypes():
    ds = SyntheticDataset(num_samples=2)
    batch = [ds[i] for i in range(2)]
    out = collate_fn(batch)
    assert out["dna_ids"].dtype == torch.long
    assert out["rna_input"].dtype == torch.long
    assert out["rna_target"].dtype == torch.long
    assert out["rna_pad_mask"].dtype == torch.bool
