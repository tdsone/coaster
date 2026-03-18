"""Top-level CoasterModel: DNA encoder + RNA decoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EncoderConfig, DecoderConfig
from .encoder import DNAEncoder
from .decoder import RNADecoder


class CoasterModel(nn.Module):
    def __init__(self, enc_config: EncoderConfig, dec_config: DecoderConfig) -> None:
        super().__init__()
        self.encoder = DNAEncoder(enc_config)
        self.decoder = RNADecoder(dec_config)
        # Project encoder output to decoder d_model if they differ
        if enc_config.d_model != dec_config.d_model:
            self.enc_proj: nn.Module = nn.Linear(enc_config.d_model, dec_config.d_model, bias=False)
        else:
            self.enc_proj = nn.Identity()

    def forward(
        self,
        dna_ids: torch.Tensor,  # (B, L_dna)
        rna_ids: torch.Tensor,  # (B, T) teacher-forced input
        src_padding_mask: torch.Tensor | None = None,  # (B, L_dna)
        tgt_padding_mask: torch.Tensor | None = None,  # (B, T)
    ) -> torch.Tensor:  # (B, T, rna_vocab_size)
        memory, mem_mask = self.encoder(dna_ids, src_padding_mask)
        memory = self.enc_proj(memory)
        return self.decoder(rna_ids, memory, tgt_padding_mask, mem_mask)

    def encode(self, dna_ids: torch.Tensor) -> torch.Tensor:
        """Run the encoder and return memory tensor (B, S, d). Used for batched generation."""
        memory, _ = self.encoder(dna_ids)
        return self.enc_proj(memory)

    @torch.no_grad()
    def generate(
        self,
        dna_ids: torch.Tensor | None,  # (B, L_dna), or None if memory is provided
        rna_tokenizer,
        max_len: int = 200,
        temperature: float = 1.0,
        greedy: bool = False,
        memory: torch.Tensor | None = None,  # (B, S, d) precomputed encoder output
    ) -> list[str]:
        """Autoregressively generate RNA reads using KV-cached decoding.

        Accepts either dna_ids (runs encoder) or precomputed memory (skips encoder).
        Returns one decoded RNA string per batch item (stops at EOS or max_len).
        """
        self.eval()

        if memory is None:
            assert dna_ids is not None
            memory = self.encode(dna_ids)

        device = memory.device
        B = memory.size(0)
        dec = self.decoder
        layers = dec.layers.layers  # nn.ModuleList
        n_layers = len(layers)
        d_model = dec.embedding.embedding_dim

        tokens = torch.full((B, 1), rna_tokenizer.BOS, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        # KV cache per layer: list of (k, v) each shape (B, t, d_model)
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.empty(B, 0, d_model, device=device),
             torch.empty(B, 0, d_model, device=device))
            for _ in range(n_layers)
        ]

        for step in range(max_len - 1):
            pos = torch.tensor([step], device=device)
            x = dec.embedding(tokens[:, -1:]) + dec.pos_emb(pos)  # (B, 1, d)

            new_kv_cache = []
            for i, layer in enumerate(layers):
                k_prev, v_prev = kv_cache[i]
                mha = layer.self_attn
                n_heads = mha.num_heads
                head_dim = d_model // n_heads

                # Split QKV weights
                w_q, w_k, w_v = mha.in_proj_weight.chunk(3, dim=0)
                if mha.in_proj_bias is not None:
                    b_q, b_k, b_v = mha.in_proj_bias.chunk(3)
                else:
                    b_q = b_k = b_v = None

                # Pre-norm + project new token only
                x_norm = layer.norm1(x)  # (B, 1, d)
                q       = F.linear(x_norm, w_q, b_q)   # (B, 1, d)
                k_new   = F.linear(x_norm, w_k, b_k)   # (B, 1, d)
                v_new   = F.linear(x_norm, w_v, b_v)   # (B, 1, d)

                # Grow KV cache
                k_full = torch.cat([k_prev, k_new], dim=1)  # (B, t+1, d)
                v_full = torch.cat([v_prev, v_new], dim=1)
                new_kv_cache.append((k_full, v_full))

                # Multi-head SDPA: reshape to (B, heads, T, head_dim)
                def to_mh(t: torch.Tensor) -> torch.Tensor:
                    return t.reshape(B, -1, n_heads, head_dim).transpose(1, 2)

                attn_out = F.scaled_dot_product_attention(
                    to_mh(q), to_mh(k_full), to_mh(v_full)
                )  # (B, n_heads, 1, head_dim)
                attn_out = (attn_out.transpose(1, 2)
                                    .reshape(B, 1, d_model))
                attn_out = F.linear(attn_out, mha.out_proj.weight, mha.out_proj.bias)
                x = x + layer.dropout1(attn_out)

                # Cross-attention to (fixed) memory — no cache needed
                cross_out, _ = layer.multihead_attn(
                    layer.norm2(x), memory, memory, need_weights=False
                )
                x = x + layer.dropout2(cross_out)

                # FFN
                x_ff = layer.norm3(x)
                x_ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_ff))))
                x = x + layer.dropout3(x_ff)

            kv_cache = new_kv_cache

            # Final norm + head — only need last position
            logits = dec.head(dec.norm(x[:, 0]))  # (B, vocab)

            if greedy:
                next_id = logits.argmax(-1)
            else:
                probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)
                next_id = torch.multinomial(probs, 1).squeeze(-1)

            tokens = torch.cat([tokens, next_id.unsqueeze(1)], dim=1)
            done |= next_id == rna_tokenizer.EOS
            if done.all():
                break

        return [rna_tokenizer.decode(row.tolist()) for row in tokens]
