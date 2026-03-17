#!/usr/bin/env python3
"""Train the Coaster DNA→RNA encoder-decoder model."""
import argparse
from torch.utils.data import random_split

import torch

from coaster.model import CoasterModel, load_config
from coaster.data.dataset import SyntheticDataset, make_dataloader
from coaster.training import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None, help="Override config device (cpu/mps/cuda)")
    args = parser.parse_args()

    enc_cfg, dec_cfg, train_cfg = load_config(args.config)

    device_str = args.device or train_cfg.device
    if device_str == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    torch.manual_seed(train_cfg.seed)

    dataset = SyntheticDataset(num_samples=train_cfg.num_samples, seed=train_cfg.seed)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(train_cfg.seed),
    )

    train_loader = make_dataloader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)

    model = CoasterModel(enc_cfg, dec_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    trainer = Trainer(model, train_loader, val_loader, train_cfg, device)
    trainer.train()


if __name__ == "__main__":
    main()
