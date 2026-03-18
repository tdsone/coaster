#!/usr/bin/env python3
"""Train the Coaster DNA→RNA encoder-decoder model."""
import argparse
import dataclasses

import torch

from coaster.model import CoasterModel, load_config
from coaster.data.dataset import collate_fn, make_dataloader
from coaster.data.real_data import RealRNADataset
from coaster.training import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None, help="Override config device (e.g. cuda:0)")
    parser.add_argument("--samples", default="data/samples_yeast.parquet")
    parser.add_argument("--reads", default="data/reads.parquet")
    parser.add_argument("--overfit", action="store_true", help="Overfit a single batch (sanity check)")
    args = parser.parse_args()

    enc_cfg, dec_cfg, train_cfg = load_config(args.config)

    device_str = args.device or train_cfg.device
    if device_str == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    torch.manual_seed(train_cfg.seed)

    train_ds = RealRNADataset(args.samples, args.reads, fold="train", dna_len=enc_cfg.dna_len)
    val_ds = RealRNADataset(args.samples, args.reads, fold="val", dna_len=enc_cfg.dna_len,
                            max_reads_per_window=train_cfg.val_reads_per_window)

    if args.overfit:
        # Single-batch overfit: grab one batch, train on it to ~0 loss
        batch = collate_fn([train_ds[i] for i in range(train_cfg.batch_size)])
        train_loader = [batch] * train_cfg.max_steps
        val_loader = None
        train_cfg = dataclasses.replace(train_cfg, warmup_steps=0, log_interval=10)
    else:
        train_loader = make_dataloader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
        val_loader = make_dataloader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)

    model = CoasterModel(enc_cfg, dec_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    trainer = Trainer(model, train_loader, val_loader, train_cfg, device)
    trainer.train()


if __name__ == "__main__":
    main()
