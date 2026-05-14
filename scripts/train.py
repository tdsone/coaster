#!/usr/bin/env python3
"""Train the pointer-factorized read model."""
from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
from datetime import datetime

import torch
import yaml

from coaster.data import ReadsDataset, make_collate_fn, make_dataloader
from coaster.model import ReadModel, load_config
from coaster.training import Trainer


def _create_run_dir(base_dir: str, config_path: str, model_cfg, train_cfg) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy2(config_path, os.path.join(run_dir, "config.yaml"))
    resolved = {
        "model": dataclasses.asdict(model_cfg),
        "training": dataclasses.asdict(train_cfg),
    }
    with open(os.path.join(run_dir, "config_resolved.yaml"), "w") as f:
        yaml.dump(resolved, f, default_flow_style=False, sort_keys=False)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None, help="Override config device (e.g. cuda:0)")
    parser.add_argument("--samples", default="data/samples_yeast.parquet")
    parser.add_argument("--reads", default="data/reads.parquet")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overfit", action="store_true",
                        help="Repeat a single batch to verify the loss can be driven down")
    parser.add_argument("--resume", default=None, metavar="CHECKPOINT",
                        help="Resume from a checkpoint (e.g. checkpoints/best.pt)")
    args = parser.parse_args()

    model_cfg, train_cfg = load_config(args.config)

    run_dir = _create_run_dir(train_cfg.checkpoint_dir, args.config, model_cfg, train_cfg)
    train_cfg = dataclasses.replace(train_cfg, checkpoint_dir=run_dir)
    print(f"Run directory: {run_dir}")

    device_str = args.device or train_cfg.device
    if device_str == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    torch.manual_seed(train_cfg.seed)

    train_ds = ReadsDataset(args.samples, args.reads, fold="train", dna_len=model_cfg.dna_len)
    val_ds = ReadsDataset(args.samples, args.reads, fold="val", dna_len=model_cfg.dna_len)

    collate_train = make_collate_fn(
        mlm_mask_prob=train_cfg.mlm_mask_prob,
        p_mlm=train_cfg.p_mlm,
        p_reads=train_cfg.p_reads,
        rc_aug_prob=train_cfg.rc_aug_prob,
    )
    # Validation: deterministic, no RC, both heads always active so val_loss is meaningful.
    collate_val = make_collate_fn(
        mlm_mask_prob=train_cfg.mlm_mask_prob,
        p_mlm=1.0,
        p_reads=1.0,
        rc_aug_prob=0.0,
        seed=train_cfg.seed,
    )

    if args.overfit:
        # Materialize one batch and feed it on repeat.
        batch = collate_train([train_ds[i] for i in range(train_cfg.batch_size)])
        train_loader = [batch] * train_cfg.max_steps
        val_loader = None
        train_cfg = dataclasses.replace(train_cfg, warmup_steps=0, log_interval=10)
    else:
        train_loader = make_dataloader(
            train_ds, batch_size=train_cfg.batch_size, collate_fn=collate_train,
            shuffle=True, num_workers=args.num_workers,
        )
        val_loader = make_dataloader(
            val_ds, batch_size=train_cfg.batch_size, collate_fn=collate_val,
            shuffle=False, num_workers=args.num_workers,
        )

    model = ReadModel(model_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    trainer = Trainer(model, train_loader, val_loader, train_cfg, device)

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
        steps_to_advance = trainer.step - trainer.scheduler.last_epoch
        for _ in range(steps_to_advance):
            trainer.scheduler.step()
        print(f"Resumed at step {trainer.step}, lr {trainer.scheduler.get_last_lr()[0]:.2e}")

    trainer.train()


if __name__ == "__main__":
    main()
