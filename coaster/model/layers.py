"""Building blocks for the pointer-factorized read model: RMSNorm, RoPE, SwiGLU."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich 2019). No bias."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability under autocast.
        dtype = x.dtype
        xf = x.float()
        rms = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (xf * rms).to(dtype) * self.weight


class RoPECache(nn.Module):
    """Pre-computed cos/sin tables for rotary positional embeddings.

    The convention here matches the spec's `apply_rope`: each head-dimension
    pair `(x_{2i}, x_{2i+1})` is rotated by angle `θ_i(pos)`.
    `cos`/`sin` are expanded to per-element tables of shape (max_len, head_dim)
    via `repeat_interleave(2)`, so they line up element-wise with the
    interleaved layout `[θ_0, θ_0, θ_1, θ_1, …]`.
    """

    def __init__(self, head_dim: int, max_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.einsum("l,d->ld", t, inv_freq)              # (L, head_dim/2)
        cos = freqs.cos().repeat_interleave(2, dim=-1)           # (L, head_dim)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.max_len = max_len

    def forward(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert length <= self.max_len, f"RoPE table too short: {length} > {self.max_len}"
        return self.cos[:length], self.sin[:length]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate even/odd pairs of `x` by position-dependent angles.

    x:   (B, H, L, D)
    cos: (L, D), sin: (L, D) — element-wise tables (see RoPECache).
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    # Broadcast cos/sin over batch and heads via shape (1, 1, L, D/2).
    c = cos[..., ::2].unsqueeze(0).unsqueeze(0)
    s = sin[..., ::2].unsqueeze(0).unsqueeze(0)
    rx1 = x1 * c - x2 * s
    rx2 = x1 * s + x2 * c
    out = torch.empty_like(x)
    out[..., ::2] = rx1
    out[..., 1::2] = rx2
    return out


class SwiGLU(nn.Module):
    """LLaMA-style SwiGLU FFN with no biases: down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model: int, d_ffn: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    """Pre-norm transformer block: RMSNorm → MHA(+RoPE) → residual → RMSNorm → SwiGLU → residual."""

    def __init__(self, d_model: int, n_heads: int, d_ffn: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.norm_attn = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm_ffn = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ffn)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, d = x.shape
        h = self.norm_attn(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)   # (B, H, L, D)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(B, L, d)
        x = x + self.proj(attn)
        x = x + self.ffn(self.norm_ffn(x))
        return x
