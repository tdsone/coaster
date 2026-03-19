#!/usr/bin/env python3
"""Train CoasterModel on Modal with an A100 40GB GPU.

Usage:
    uv run modal run scripts/train_modal.py
"""

import dataclasses
from pathlib import Path

import modal

app = modal.App("coaster-train")

data_vol = modal.Volume.from_name("coaster-data")
checkpoints_vol = modal.Volume.from_name("coaster-checkpoints", create_if_missing=True)

DATA_DIR = Path("/data/coaster")
CHECKPOINT_DIR = Path("/checkpoints")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy>=1.26.0",
        "pandas>=2.0",
        "pyarrow>=14.0",
        "pyyaml>=6.0",
        "wandb>=0.19.0",
        "tqdm>=4.66.0",
    )
    .add_local_python_source("coaster")
    .add_local_file("configs/default.yaml", "/configs/default.yaml")
)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={
        str(DATA_DIR): data_vol,
        str(CHECKPOINT_DIR): checkpoints_vol,
    },
    secrets=[modal.Secret.from_name("wandb")],
    timeout=86400,  # 24h
)
def train() -> None:
    import torch
    from coaster.model import CoasterModel, load_config
    from coaster.data.dataset import make_dataloader
    from coaster.data.real_data import RealRNADataset
    from coaster.training import Trainer

    enc_cfg, dec_cfg, train_cfg = load_config("/configs/default.yaml")
    train_cfg = dataclasses.replace(train_cfg, checkpoint_dir=str(CHECKPOINT_DIR))

    device = torch.device("cuda")
    print(f"Device: {device}")
    torch.manual_seed(train_cfg.seed)

    train_ds = RealRNADataset(
        DATA_DIR / "samples_yeast.parquet",
        DATA_DIR / "reads.parquet",
        fold="train",
        dna_len=enc_cfg.dna_len,
    )
    val_ds = RealRNADataset(
        DATA_DIR / "samples_yeast.parquet",
        DATA_DIR / "reads.parquet",
        fold="val",
        dna_len=enc_cfg.dna_len,
        max_reads_per_window=train_cfg.val_reads_per_window,
    )

    train_loader = make_dataloader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)

    model = CoasterModel(enc_cfg, dec_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    trainer = Trainer(model, train_loader, val_loader, train_cfg, device)

    # Wrap save_checkpoint to commit to Modal volume after each save
    _orig_save = trainer.save_checkpoint
    def _save_and_commit(path: str) -> None:
        _orig_save(path)
        checkpoints_vol.commit()
    trainer.save_checkpoint = _save_and_commit

    trainer.train()


@app.local_entrypoint()
def main() -> None:
    train.remote()
