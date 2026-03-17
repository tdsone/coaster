import os
import tempfile

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from coaster.model.config import EncoderConfig, DecoderConfig, TrainingConfig
from coaster.model.transformer import CoasterModel
from coaster.data.dataset import SyntheticDataset, collate_fn, make_dataloader
from coaster.training import Trainer

# Small configs for fast CPU tests
ENC = EncoderConfig(d_model=32, n_heads=2, n_layers=1, ffn_dim=64, dna_len=64, conv_kernel=8, conv_stride=8)
DEC = DecoderConfig(d_model=32, n_heads=2, n_layers=1, ffn_dim=64, max_rna_len=30)


@pytest.fixture
def tiny_loader():
    ds = SyntheticDataset(num_samples=10, dna_len=ENC.dna_len, rna_len=20, seed=0)
    return make_dataloader(ds, batch_size=2, shuffle=False)


@pytest.fixture
def small_train_cfg(tmp_path):
    return TrainingConfig(
        batch_size=2,
        num_epochs=2,
        lr=1e-3,
        num_samples=10,
        log_interval=1000,
        checkpoint_dir=str(tmp_path / "ckpts"),
    )


def test_one_step(tiny_loader, small_train_cfg):
    model = CoasterModel(ENC, DEC)
    device = torch.device("cpu")
    trainer = Trainer(model, tiny_loader, None, small_train_cfg, device)
    batch = next(iter(tiny_loader))
    loss = trainer._forward_loss(batch)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_loss_decreases(tiny_loader):
    model = CoasterModel(ENC, DEC)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    batch = next(iter(tiny_loader))
    dna_ids = batch["dna_ids"]
    rna_input = batch["rna_input"]
    rna_target = batch["rna_target"]

    losses = []
    for _ in range(60):
        logits = model(dna_ids, rna_input)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            rna_target.reshape(-1),
            ignore_index=0,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


def test_checkpoint_save_load(tiny_loader, small_train_cfg, tmp_path):
    model = CoasterModel(ENC, DEC)
    device = torch.device("cpu")
    trainer = Trainer(model, tiny_loader, None, small_train_cfg, device)

    path = str(tmp_path / "ckpt.pt")
    trainer.save_checkpoint(path)
    assert os.path.exists(path)

    model2 = CoasterModel(ENC, DEC)
    trainer2 = Trainer(model2, tiny_loader, None, small_train_cfg, device)
    trainer2.load_checkpoint(path)

    for (k, v1), (_, v2) in zip(
        model.state_dict().items(), model2.state_dict().items()
    ):
        assert torch.allclose(v1, v2), f"State mismatch at {k}"


def test_lr_lambda_warmup():
    fn = Trainer._lr_lambda(warmup_steps=10, total_steps=100)
    assert fn(0) == pytest.approx(0.0)
    assert fn(5) == pytest.approx(0.5)
    assert fn(10) == pytest.approx(1.0)


def test_lr_lambda_decay():
    fn = Trainer._lr_lambda(warmup_steps=0, total_steps=100)
    # At step 0 (progress=0) cosine gives 1.0; at step 100 (progress=1) gives 0.0
    assert fn(0) == pytest.approx(1.0)
    assert fn(100) == pytest.approx(0.0, abs=1e-6)


def test_trainer_runs_one_epoch(tiny_loader, small_train_cfg):
    cfg = TrainingConfig(
        batch_size=2,
        num_epochs=1,
        lr=1e-3,
        num_samples=10,
        log_interval=1000,
        checkpoint_dir=small_train_cfg.checkpoint_dir,
    )
    model = CoasterModel(ENC, DEC)
    device = torch.device("cpu")
    trainer = Trainer(model, tiny_loader, None, cfg, device)
    trainer.train()  # should complete without error
