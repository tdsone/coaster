"""Top-level CoasterModel: DNA encoder + RNA decoder."""
from __future__ import annotations

import torch
import torch.nn as nn

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

    @torch.no_grad()
    def generate(
        self,
        dna_ids: torch.Tensor,  # (B, L_dna)
        rna_tokenizer,
        max_len: int = 200,
        temperature: float = 1.0,
        greedy: bool = True,
    ) -> list[str]:
        """Autoregressively generate RNA reads from DNA context.

        Returns one decoded RNA string per batch item (stops at EOS).
        """
        self.eval()
        device = dna_ids.device
        B = dna_ids.size(0)
        bos_id = rna_tokenizer.BOS
        eos_id = rna_tokenizer.EOS

        memory, mem_mask = self.encoder(dna_ids)
        memory = self.enc_proj(memory)

        tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.decoder(tokens, memory, memory_key_padding_mask=mem_mask)
            next_logits = logits[:, -1]  # (B, vocab)
            if greedy:
                next_id = next_logits.argmax(-1)
            else:
                probs = torch.softmax(next_logits / max(temperature, 1e-8), dim=-1)
                next_id = torch.multinomial(probs, 1).squeeze(-1)
            tokens = torch.cat([tokens, next_id.unsqueeze(1)], dim=1)
            done |= next_id == eos_id
            if done.all():
                break

        return [rna_tokenizer.decode(row.tolist()) for row in tokens]
