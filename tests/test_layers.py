"""Tests for layer primitives: RMSNorm, RoPE, SwiGLU, Block."""
import torch

from coaster.model.layers import Block, RMSNorm, RoPECache, SwiGLU, apply_rope


def test_rmsnorm_shape_and_finite():
    norm = RMSNorm(32)
    x = torch.randn(2, 10, 32)
    y = norm(x)
    assert y.shape == (2, 10, 32)
    assert torch.isfinite(y).all()


def test_rmsnorm_unit_rms_with_identity_weight():
    norm = RMSNorm(64)
    x = torch.randn(4, 7, 64) * 5.0
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_cache_shape():
    cache = RoPECache(head_dim=16, max_len=64)
    cos, sin = cache(50)
    assert cos.shape == (50, 16)
    assert sin.shape == (50, 16)


def test_rope_identity_at_position_zero():
    """At pos=0 RoPE is identity: cos=1, sin=0 everywhere."""
    cache = RoPECache(head_dim=16, max_len=4)
    cos, sin = cache(1)
    x = torch.randn(1, 2, 1, 16)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(y, x, atol=1e-6)


def test_rope_preserves_norm():
    """RoPE is a rotation, so |x| per (head, pos) should be preserved."""
    cache = RoPECache(head_dim=32, max_len=10)
    cos, sin = cache(10)
    x = torch.randn(2, 4, 10, 32)
    y = apply_rope(x, cos, sin)
    nx = x.pow(2).sum(-1)
    ny = y.pow(2).sum(-1)
    assert torch.allclose(nx, ny, atol=1e-5)


def test_rope_position_dependent():
    """Different positions should produce different rotations."""
    cache = RoPECache(head_dim=8, max_len=16)
    cos, sin = cache(16)
    x = torch.randn(1, 1, 16, 8)
    y = apply_rope(x, cos, sin)
    # First and last positions should disagree
    assert not torch.allclose(y[:, :, 0], y[:, :, -1])
    # But y[:, :, 0] == x[:, :, 0] (identity at pos 0)
    assert torch.allclose(y[:, :, 0], x[:, :, 0], atol=1e-6)


def test_swiglu_shape_and_no_bias():
    f = SwiGLU(32, 64)
    x = torch.randn(2, 10, 32)
    assert f(x).shape == (2, 10, 32)
    for layer in (f.w_gate, f.w_up, f.w_down):
        assert layer.bias is None


def test_block_forward_shape():
    blk = Block(d_model=32, n_heads=4, d_ffn=64)
    cache = RoPECache(head_dim=8, max_len=20)
    cos, sin = cache(20)
    x = torch.randn(3, 20, 32)
    y = blk(x, cos, sin)
    assert y.shape == (3, 20, 32)
    assert torch.isfinite(y).all()


def test_block_backward():
    blk = Block(d_model=32, n_heads=4, d_ffn=64)
    cache = RoPECache(head_dim=8, max_len=20)
    cos, sin = cache(20)
    x = torch.randn(2, 20, 32, requires_grad=True)
    y = blk(x, cos, sin)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
