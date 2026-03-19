"""Autoregressive RNA decoder with cross-attention to DNA encoder output."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import DecoderConfig
from .layers import RMSNorm


class RNADecoder(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        d = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, d, padding_idx=0)
        self.pos_emb = nn.Embedding(config.max_rna_len, d)  # learned
        self.dropout = nn.Dropout(config.dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            norm_first=True,
            batch_first=True,
        )
        self.layers = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layers)
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, config.vocab_size, bias=True)
        self._init_head_bias()

    def _init_head_bias(self) -> None:
        """Initialise head bias so the 4 nucleotides start with roughly equal
        probability and special tokens are suppressed from the start."""
        with torch.no_grad():
            nn.init.zeros_(self.head.bias)
            # Special tokens that should almost never be generated
            self.head.bias[0] = -10.0  # PAD
            self.head.bias[1] = -10.0  # BOS
            # Rare but valid tokens
            self.head.bias[2] = -2.0   # EOS  (~1/read_len frequency)
            self.head.bias[7] = -3.0   # N    (ambiguous base, uncommon)
            # A/U/G/C (indices 3-6) left at 0: roughly equal.
            # The random weight matrix breaks exact symmetry, giving
            # "roughly but not identically" equal initial probabilities.

    def forward(
        self,
        tgt_ids: torch.Tensor,  # (B, T)
        memory: torch.Tensor,  # (B, S, d)
        tgt_key_padding_mask: torch.Tensor | None = None,  # (B, T) bool
        memory_key_padding_mask: torch.Tensor | None = None,  # (B, S) bool
    ) -> torch.Tensor:
        T = tgt_ids.size(1)
        positions = torch.arange(T, device=tgt_ids.device)
        x = self.dropout(self.embedding(tgt_ids) + self.pos_emb(positions))

        # Bool causal mask: True = masked (future tokens); created on CPU for MPS safety
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1).to(tgt_ids.device)

        x = self.layers(
            x,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.head(self.norm(x))  # (B, T, vocab_size)
