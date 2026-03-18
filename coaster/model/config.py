"""Model and training configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EncoderConfig:
    vocab_size: int = 6
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    dna_len: int = 4992
    conv_kernel: int = 8
    conv_stride: int = 8

    @property
    def enc_seq_len(self) -> int:
        return (self.dna_len - self.conv_kernel) // self.conv_stride + 1


@dataclass(frozen=True)
class DecoderConfig:
    vocab_size: int = 8
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    max_rna_len: int = 200


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    max_steps: int = 500_000
    eval_interval: int = 10_000
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    num_samples: int = 50000
    seed: int = 42
    device: str = "mps"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 100
    wandb_project: str | None = None


def load_config(path: str | Path) -> tuple[EncoderConfig, DecoderConfig, TrainingConfig]:
    with open(path) as f:
        d = yaml.safe_load(f)
    return (
        EncoderConfig(**d.get("encoder", {})),
        DecoderConfig(**d.get("decoder", {})),
        TrainingConfig(**d.get("training", {})),
    )
