"""Tests for ReadModel: forward shapes, head masking, head backward, coverage agg."""
import torch
import torch.nn.functional as F

from coaster.model.config import ModelConfig
from coaster.model.read_model import ReadModel, coverage_from_reads


def _tiny_config(**kw) -> ModelConfig:
    base = dict(
        d_model=32, n_heads=4, n_layers=2, d_ffn=64,
        vocab_size=7, dna_len=64, n_assays=0, n_cell_types=0, n_species=0,
    )
    base.update(kw)
    return ModelConfig(**base)


def _random_dna(B: int, L: int) -> torch.Tensor:
    """Tokens in {A, C, G, T, N} = {1..5}."""
    return torch.randint(1, 6, (B, L))


def test_ffn_dim_llama_rule():
    cfg = ModelConfig(d_model=384)
    assert cfg.ffn_dim == round(8 / 3 * 384 / 64) * 64    # 1024
    assert cfg.ffn_dim % 64 == 0


def test_encode_shapes():
    cfg = _tiny_config()
    model = ReadModel(cfg)
    dna = _random_dna(3, cfg.dna_len)
    h_cls, h_seq = model.encode(dna)
    assert h_cls.shape == (3, cfg.d_model)
    assert h_seq.shape == (3, cfg.dna_len, cfg.d_model)


def test_strand_logits_shape():
    cfg = _tiny_config()
    model = ReadModel(cfg)
    dna = _random_dna(3, cfg.dna_len)
    h_cls, _ = model.encode(dna)
    logits = model.strand_logits(h_cls)
    assert logits.shape == (3, 2)


def test_start_logits_shape():
    cfg = _tiny_config()
    model = ReadModel(cfg)
    dna = _random_dna(3, cfg.dna_len)
    _, h_seq = model.encode(dna)
    strand = torch.zeros(3, dtype=torch.long)
    logits = model.start_logits(h_seq, strand)
    assert logits.shape == (3, cfg.dna_len)
    assert torch.isfinite(logits).all()


def test_end_logits_mask_below_start():
    """end_logits at j < start must be -inf."""
    cfg = _tiny_config()
    model = ReadModel(cfg)
    dna = _random_dna(4, cfg.dna_len)
    _, h_seq = model.encode(dna)
    strand = torch.tensor([0, 1, 0, 1])
    start = torch.tensor([10, 0, 30, 5])
    logits = model.end_logits(h_seq, start, strand)
    assert logits.shape == (4, cfg.dna_len)
    for b, s in enumerate(start.tolist()):
        # positions < start[b] must be -inf
        if s > 0:
            assert torch.isinf(logits[b, :s]).all() and (logits[b, :s] < 0).all()
        # positions >= start[b] must be finite
        assert torch.isfinite(logits[b, s:]).all()


def test_end_logits_max_read_len_mask():
    """If max_read_len is provided, j > start + max_read_len is masked."""
    cfg = _tiny_config()
    model = ReadModel(cfg)
    dna = _random_dna(2, cfg.dna_len)
    _, h_seq = model.encode(dna)
    strand = torch.zeros(2, dtype=torch.long)
    start = torch.tensor([10, 20])
    L_max = 5
    logits = model.end_logits(h_seq, start, strand, max_read_len=L_max)
    for b, s in enumerate(start.tolist()):
        # Finite window: s .. s+L_max inclusive
        finite_slice = logits[b, s : s + L_max + 1]
        assert torch.isfinite(finite_slice).all()
        # Beyond s + L_max must be -inf
        beyond = logits[b, s + L_max + 1 :]
        if beyond.numel() > 0:
            assert (beyond == float("-inf")).all()


def test_strand_logits_distinguish_strands():
    """Strand head should produce different logits given a different CLS — verified by
    flipping the strand_embed input to start head (sanity that strand_embed is used)."""
    cfg = _tiny_config()
    model = ReadModel(cfg)
    dna = _random_dna(2, cfg.dna_len)
    _, h_seq = model.encode(dna)
    l_plus = model.start_logits(h_seq, torch.zeros(2, dtype=torch.long))
    l_minus = model.start_logits(h_seq, torch.ones(2, dtype=torch.long))
    # Different strands → different start logits (otherwise σ has no effect).
    assert not torch.allclose(l_plus, l_minus)


def test_mlm_logits_shape():
    cfg = _tiny_config()
    model = ReadModel(cfg)
    dna = _random_dna(2, cfg.dna_len)
    _, h_seq = model.encode(dna)
    logits = model.mlm_logits(h_seq)
    assert logits.shape == (2, cfg.dna_len, cfg.vocab_size)


def test_conditioning_tokens_stripped_from_h_seq():
    """h_seq should have length dna_len even when conditioning tokens are prepended."""
    cfg = _tiny_config(n_assays=4, n_cell_types=3)
    model = ReadModel(cfg)
    dna = _random_dna(2, cfg.dna_len)
    assay = torch.tensor([0, 2])
    cell = torch.tensor([1, 0])
    h_cls, h_seq = model.encode(dna, assay=assay, cell_type=cell)
    assert h_seq.shape == (2, cfg.dna_len, cfg.d_model)
    assert h_cls.shape == (2, cfg.d_model)


def test_conditioning_tokens_affect_output():
    """Swapping the assay token should change predictions (Karollus species-swap analogue)."""
    cfg = _tiny_config(n_assays=4)
    model = ReadModel(cfg)
    dna = _random_dna(2, cfg.dna_len)
    h0_cls, _ = model.encode(dna, assay=torch.tensor([0, 0]))
    h1_cls, _ = model.encode(dna, assay=torch.tensor([1, 1]))
    assert not torch.allclose(h0_cls, h1_cls)


def test_backward_through_all_heads():
    """All three read losses + MLM should accumulate gradients onto the encoder."""
    cfg = _tiny_config()
    model = ReadModel(cfg)
    dna = _random_dna(4, cfg.dna_len)
    h_cls, h_seq = model.encode(dna)

    strand_true = torch.randint(0, 2, (4,))
    start_true = torch.randint(0, cfg.dna_len // 2, (4,))
    end_offset = torch.randint(0, cfg.dna_len // 2, (4,))
    end_true = start_true + end_offset

    loss_sigma = F.cross_entropy(model.strand_logits(h_cls), strand_true)
    loss_start = F.cross_entropy(model.start_logits(h_seq, strand_true), start_true)
    loss_end = F.cross_entropy(model.end_logits(h_seq, start_true, strand_true), end_true)
    target = torch.randint(1, cfg.vocab_size, (4, cfg.dna_len))
    loss_mlm = F.cross_entropy(
        model.mlm_logits(h_seq).reshape(-1, cfg.vocab_size), target.reshape(-1)
    )
    (loss_sigma + loss_start + loss_end + loss_mlm).backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    # Encoder embedding must receive gradient (used by every loss path).
    assert model.embed.weight.grad is not None
    # Both strand-bilinear matrices should get gradient when both strands appear in batch.
    if (strand_true == 0).any() and (strand_true == 1).any():
        assert (model.end_bilinear.grad != 0).any()


def test_sample_reads_shapes_and_constraints():
    cfg = _tiny_config()
    model = ReadModel(cfg)
    dna = _random_dna(2, cfg.dna_len)
    strands, starts, ends = model.sample_reads(dna, n_reads=5)
    assert strands.shape == (2, 5)
    assert starts.shape == (2, 5)
    assert ends.shape == (2, 5)
    # Constraint from end-head mask: end >= start.
    assert (ends >= starts).all()
    # Strands in {0,1}; starts/ends within window.
    assert ((strands == 0) | (strands == 1)).all()
    assert (starts >= 0).all() and (starts < cfg.dna_len).all()
    assert (ends >= 0).all() and (ends < cfg.dna_len).all()


def test_coverage_from_reads_simple():
    starts = torch.tensor([[0, 5]])
    ends = torch.tensor([[2, 6]])
    cov = coverage_from_reads(starts, ends, L=8)
    # Read 1 covers 0..2, read 2 covers 5..6.
    assert cov.tolist() == [[1, 1, 1, 0, 0, 1, 1, 0]]


def test_coverage_from_reads_strand_filter():
    starts = torch.tensor([[0, 5]])
    ends = torch.tensor([[2, 6]])
    strands = torch.tensor([[0, 1]])
    cov_plus = coverage_from_reads(starts, ends, L=8, strands=strands, target_strand=0)
    cov_minus = coverage_from_reads(starts, ends, L=8, strands=strands, target_strand=1)
    assert cov_plus.tolist() == [[1, 1, 1, 0, 0, 0, 0, 0]]
    assert cov_minus.tolist() == [[0, 0, 0, 0, 0, 1, 1, 0]]
