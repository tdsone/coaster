"""Configuration dataclasses for the pointer-factorized read model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    # Architecture
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 8
    d_ffn: int | None = None        # None → round 8/3 * d_model to multiple of 64
    vocab_size: int = 7              # DNATokenizer.VOCAB_SIZE
    dna_len: int = 4992
    rope_base: float = 10_000.0
    # Conditioning. 0 disables the embedding for that axis.
    n_assays: int = 0
    n_cell_types: int = 0
    n_species: int = 0
    # End-head sees a known max-length budget; None disables the extra mask.
    max_read_len: int | None = None

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_dim(self) -> int:
        if self.d_ffn is not None:
            return self.d_ffn
        # LLaMA recipe: 8/3 * d, rounded to a multiple of 64.
        return int(round(8 / 3 * self.d_model / 64) * 64)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    max_steps: int = 500_000
    eval_interval: int = 10_000
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2_000
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 100
    wandb_project: str | None = None
    # Loss weights
    alpha_mlm: float = 1.0
    beta_read: float = 1.0
    # In-batch population probabilities (independent Bernoullis per sample).
    # (p_mlm, p_reads) controls the three §4 populations: MLM-only=(1,0),
    # read-only=(0,1), joint=(1,1). Spec default ~40/40/20 ⇒ (0.6, 0.6).
    p_mlm: float = 0.6
    p_reads: float = 0.6
    mlm_mask_prob: float = 0.15
    rc_aug_prob: float = 0.5


def load_config(path: str | Path) -> tuple[ModelConfig, TrainingConfig]:
    with open(path) as f:
        d = yaml.safe_load(f)
    return (
        ModelConfig(**d.get("model", {})),
        TrainingConfig(**d.get("training", {})),
    )
