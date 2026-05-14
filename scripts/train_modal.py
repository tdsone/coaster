#!/usr/bin/env python3
"""Train the pointer-factorized read model on Modal with an A100 40GB GPU.

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
    timeout=86400,
)
def train() -> None:
    import torch
    from coaster.data import ReadsDataset, make_collate_fn, make_dataloader
    from coaster.model import ReadModel, load_config
    from coaster.training import Trainer

    model_cfg, train_cfg = load_config("/configs/default.yaml")
    train_cfg = dataclasses.replace(train_cfg, checkpoint_dir=str(CHECKPOINT_DIR))

    device = torch.device("cuda")
    print(f"Device: {device}")
    torch.manual_seed(train_cfg.seed)

    train_ds = ReadsDataset(
        DATA_DIR / "samples_yeast.parquet",
        DATA_DIR / "reads.parquet",
        fold="train",
        dna_len=model_cfg.dna_len,
    )
    val_ds = ReadsDataset(
        DATA_DIR / "samples_yeast.parquet",
        DATA_DIR / "reads.parquet",
        fold="val",
        dna_len=model_cfg.dna_len,
    )

    collate_train = make_collate_fn(
        mlm_mask_prob=train_cfg.mlm_mask_prob,
        p_mlm=train_cfg.p_mlm,
        p_reads=train_cfg.p_reads,
        rc_aug_prob=train_cfg.rc_aug_prob,
    )
    collate_val = make_collate_fn(
        mlm_mask_prob=train_cfg.mlm_mask_prob,
        p_mlm=1.0, p_reads=1.0, rc_aug_prob=0.0, seed=train_cfg.seed,
    )

    train_loader = make_dataloader(
        train_ds, batch_size=train_cfg.batch_size, collate_fn=collate_train, shuffle=True,
    )
    val_loader = make_dataloader(
        val_ds, batch_size=train_cfg.batch_size, collate_fn=collate_val, shuffle=False,
    )

    model = ReadModel(model_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    trainer = Trainer(model, train_loader, val_loader, train_cfg, device)

    _orig_save = trainer.save_checkpoint

    def _save_and_commit(path: str) -> None:
        _orig_save(path)
        checkpoints_vol.commit()

    trainer.save_checkpoint = _save_and_commit

    resume_path = CHECKPOINT_DIR / "best.pt"
    if resume_path.exists():
        print(f"Resuming from {resume_path}")
        trainer.load_checkpoint(str(resume_path))
        steps_to_advance = trainer.step - trainer.scheduler.last_epoch
        for _ in range(steps_to_advance):
            trainer.scheduler.step()
        print(f"Resumed at step {trainer.step}, lr {trainer.scheduler.get_last_lr()[0]:.2e}")
    else:
        print("No checkpoint found, training from scratch")

    trainer.train()


@app.local_entrypoint()
def main() -> None:
    train.spawn()
