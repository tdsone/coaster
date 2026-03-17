"""DNA transformer encoder with strided conv downsampling."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import EncoderConfig
from .layers import SinusoidalPosEmb


class DNAEncoder(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        d = config.d_model
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, d, padding_idx=0)
        # Non-overlapping conv to compress 4992 → 624 positions
        self.conv = nn.Conv1d(d, d, kernel_size=config.conv_kernel, stride=config.conv_stride)
        self.pos_emb = SinusoidalPosEmb(config.enc_seq_len, d)
        self.dropout = nn.Dropout(config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            norm_first=True,
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers, enable_nested_tensor=False
        )

    def forward(
        self,
        dna_ids: torch.Tensor,  # (B, L)
        src_key_padding_mask: torch.Tensor | None = None,  # (B, L) bool, True=pad
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embedding(dna_ids)  # (B, L, d)
        # Conv expects (B, d, L)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, L//stride, d)
        x = self.dropout(self.pos_emb(x))

        # Downsample padding mask: mark compressed position as padded if any
        # source position in its stride window was padded
        mem_mask: torch.Tensor | None = None
        if src_key_padding_mask is not None:
            B, S = dna_ids.size(0), x.size(1)
            mem_mask = src_key_padding_mask[:, : S * self.config.conv_stride].view(
                B, S, self.config.conv_stride
            ).any(dim=-1)

        x = self.layers(x, src_key_padding_mask=mem_mask)
        return x, mem_mask  # (B, S, d), (B, S) or None
