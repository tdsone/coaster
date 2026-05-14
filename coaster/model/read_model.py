"""Pointer-factorized read model: p(σ | D) · p(s | σ, D) · p(e | s, σ, D) + MLM head.

Architecture (see specs/pointer-factorized.md):
  - Pre-norm transformer encoder over the DNA window (no conv downsampling).
  - RMSNorm + RoPE + SwiGLU primitives, no biases on linears.
  - Prepended [CLS] token + optional assay/cell-type/species conditioning tokens.
  - Strand head: linear from h_cls.
  - Start head: per-position linear + strand-conditioned global bias.
  - End head: bilinear in (h_j, h_s) with a per-strand matrix + position/strand biases.
  - MLM head: per-position linear over vocab.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coaster.model.config import ModelConfig
from coaster.model.layers import Block, RMSNorm, RoPECache


class ReadModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d, h = config.d_model, config.n_heads
        d_ffn = config.ffn_dim

        # Token + conditioning embeddings.
        self.embed = nn.Embedding(config.vocab_size, d, padding_idx=0)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls, std=0.02)
        self.assay_embed = nn.Embedding(config.n_assays, d) if config.n_assays > 0 else None
        self.cell_embed = nn.Embedding(config.n_cell_types, d) if config.n_cell_types > 0 else None
        self.species_embed = nn.Embedding(config.n_species, d) if config.n_species > 0 else None

        # Max RoPE length = window + max prefix size (CLS + up to 3 conditioning tokens).
        max_len = config.dna_len + 4
        self.rope = RoPECache(config.head_dim, max_len, base=config.rope_base)

        self.blocks = nn.ModuleList([Block(d, h, d_ffn) for _ in range(config.n_layers)])
        self.norm_out = RMSNorm(d)

        # Strand embedding used by start/end heads.
        self.strand_embed = nn.Embedding(2, d)              # 0: +, 1: -

        # Heads.
        self.strand_head = nn.Linear(d, 2, bias=False)
        self.start_head = nn.Linear(d, 1, bias=False)
        self.start_strand_bias = nn.Linear(d, 1, bias=False)    # uses σ embedding → scalar bias
        self.end_bilinear = nn.Parameter(torch.zeros(2, d, d))  # one matrix per σ
        self.end_pos_bias = nn.Linear(d, 1, bias=False)
        self.end_strand_bias = nn.Embedding(2, d)               # σ → d_model
        self.mlm_head = nn.Linear(d, config.vocab_size, bias=False)

        self._init_weights()

    # --------------------------------------------------------------------
    def _init_weights(self) -> None:
        # Standard small-std init for embeddings and projections; bilinear
        # matrix initialised orthogonal so each strand starts with a non-zero,
        # full-rank transform.
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.strand_embed.weight, std=0.02)
        nn.init.normal_(self.end_strand_bias.weight, std=0.02)
        for emb in (self.assay_embed, self.cell_embed, self.species_embed):
            if emb is not None:
                nn.init.normal_(emb.weight, std=0.02)
        for sigma in (0, 1):
            nn.init.orthogonal_(self.end_bilinear[sigma])

    # --------------------------------------------------------------------
    def _prefix(
        self,
        batch_size: int,
        assay: torch.Tensor | None,
        cell_type: torch.Tensor | None,
        species: torch.Tensor | None,
    ) -> torch.Tensor:
        """Stack CLS + conditioning tokens into a (B, n_prefix, d) tensor."""
        toks: list[torch.Tensor] = [self.cls.expand(batch_size, 1, -1)]
        if self.assay_embed is not None and assay is not None:
            toks.append(self.assay_embed(assay).unsqueeze(1))
        if self.cell_embed is not None and cell_type is not None:
            toks.append(self.cell_embed(cell_type).unsqueeze(1))
        if self.species_embed is not None and species is not None:
            toks.append(self.species_embed(species).unsqueeze(1))
        return torch.cat(toks, dim=1)

    def encode(
        self,
        dna: torch.Tensor,                     # (B, L) int64
        assay: torch.Tensor | None = None,     # (B,) or None
        cell_type: torch.Tensor | None = None,
        species: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the encoder. Returns (h_cls (B, d), h_seq (B, L, d))."""
        B, L = dna.shape
        prefix = self._prefix(B, assay, cell_type, species)
        h = torch.cat([prefix, self.embed(dna)], dim=1)        # (B, n_prefix+L, d)
        cos, sin = self.rope(h.size(1))
        for blk in self.blocks:
            h = blk(h, cos, sin)
        h = self.norm_out(h)
        n_prefix = prefix.size(1)
        return h[:, 0], h[:, n_prefix:]

    # ---- heads ---------------------------------------------------------
    def strand_logits(self, h_cls: torch.Tensor) -> torch.Tensor:
        return self.strand_head(h_cls)                         # (B, 2)

    def start_logits(self, h_seq: torch.Tensor, strand: torch.Tensor) -> torch.Tensor:
        sigma_e = self.strand_embed(strand)                    # (B, d)
        per_pos = self.start_head(h_seq).squeeze(-1)           # (B, L)
        strand_bias = self.start_strand_bias(sigma_e).squeeze(-1).unsqueeze(1)  # (B, 1)
        return per_pos + strand_bias

    def end_logits(
        self,
        h_seq: torch.Tensor,
        start: torch.Tensor,
        strand: torch.Tensor,
        max_read_len: int | None = None,
    ) -> torch.Tensor:
        B, L, d = h_seq.shape
        # Gather h at the conditioning start position.
        h_s = h_seq.gather(1, start.view(B, 1, 1).expand(B, 1, d)).squeeze(1)   # (B, d)
        W = self.end_bilinear[strand]                                            # (B, d, d)
        # bilinear[b, j] = h_seq[b, j] @ W[b] @ h_s[b]
        bil = torch.einsum("bld,bde,be->bl", h_seq, W, h_s)                      # (B, L)
        pos_bias = self.end_pos_bias(h_seq).squeeze(-1)                          # (B, L)
        strand_bias = (h_seq * self.end_strand_bias(strand).unsqueeze(1)).sum(-1)  # (B, L)
        logits = bil + pos_bias + strand_bias

        positions = torch.arange(L, device=h_seq.device).unsqueeze(0)            # (1, L)
        mask = positions < start.unsqueeze(1)
        max_read_len = max_read_len if max_read_len is not None else self.config.max_read_len
        if max_read_len is not None:
            mask = mask | (positions > (start.unsqueeze(1) + max_read_len))
        return logits.masked_fill(mask, float("-inf"))

    def mlm_logits(self, h_seq: torch.Tensor) -> torch.Tensor:
        return self.mlm_head(h_seq)                            # (B, L, V)

    # ---- inference -----------------------------------------------------
    @torch.no_grad()
    def sample_reads(
        self,
        dna: torch.Tensor,
        n_reads: int,
        assay: torch.Tensor | None = None,
        cell_type: torch.Tensor | None = None,
        species: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ancestral sampling. Returns (strands, starts, ends), each (B, n_reads)."""
        self.eval()
        h_cls, h_seq = self.encode(dna, assay=assay, cell_type=cell_type, species=species)
        sigma_dist = torch.distributions.Categorical(logits=self.strand_logits(h_cls))

        strands, starts, ends = [], [], []
        for _ in range(n_reads):
            sigma = sigma_dist.sample()                                              # (B,)
            s = torch.distributions.Categorical(
                logits=self.start_logits(h_seq, sigma)
            ).sample()                                                                # (B,)
            e = torch.distributions.Categorical(
                logits=self.end_logits(h_seq, s, sigma)
            ).sample()
            strands.append(sigma); starts.append(s); ends.append(e)
        return (
            torch.stack(strands, dim=1),
            torch.stack(starts, dim=1),
            torch.stack(ends, dim=1),
        )


def coverage_from_reads(
    starts: torch.Tensor,
    ends: torch.Tensor,
    L: int,
    strands: torch.Tensor | None = None,
    target_strand: int | None = None,
) -> torch.Tensor:
    """Aggregate sampled (start, end) pairs into a per-position coverage profile.

    Inputs: starts/ends of shape (B, N). Returns (B, L) int64.
    If `target_strand` is provided, only reads with matching σ contribute.
    """
    if target_strand is not None:
        assert strands is not None, "Need strands to filter by target_strand"
        keep = strands == target_strand
        starts = torch.where(keep, starts, torch.full_like(starts, -1))
        ends = torch.where(keep, ends, torch.full_like(ends, -2))
    positions = torch.arange(L, device=starts.device)                        # (L,)
    covered = (starts.unsqueeze(-1) <= positions) & (ends.unsqueeze(-1) >= positions)
    return covered.sum(dim=1)
