"""Tests for the trainer's loss decomposition and end-to-end step."""
from __future__ import annotations

import torch

from coaster.data.dataset import make_collate_fn
from coaster.model.config import ModelConfig, TrainingConfig
from coaster.model.read_model import ReadModel
from coaster.tokenizer import DNATokenizer
from coaster.training.trainer import Trainer, training_step


TOK = DNATokenizer()


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        d_model=32, n_heads=4, n_layers=2, d_ffn=64,
        vocab_size=TOK.VOCAB_SIZE, dna_len=32,
    )


def _make_batch(B: int = 4, L: int = 32, *, p_mlm: float, p_reads: float, seed: int = 0) -> dict:
    collate = make_collate_fn(p_mlm=p_mlm, p_reads=p_reads, rc_aug_prob=0.0, seed=seed)
    items = []
    for i in range(B):
        items.append({
            "dna_ids": torch.tensor(TOK.encode(("ACGT" * (L // 4))[:L])),
            "start": i % L,
            "end": (i % L + 3) % L if (i % L + 3) < L else L - 1,
            "strand": i % 2,
        })
        items[-1]["end"] = max(items[-1]["start"], items[-1]["end"])
    return collate(items)


def test_training_step_full_population_returns_finite():
    model = ReadModel(_tiny_model_config())
    batch = _make_batch(p_mlm=1.0, p_reads=1.0)
    out = training_step(model, batch)
    assert torch.isfinite(out["loss"])
    assert out["mlm"] > 0
    assert out["read"] > 0
    assert out["sigma"] > 0
    assert out["start"] > 0
    assert out["end"] > 0


def test_training_step_read_only_zero_mlm():
    model = ReadModel(_tiny_model_config())
    batch = _make_batch(p_mlm=0.0, p_reads=1.0)
    out = training_step(model, batch)
    assert out["mlm"].item() == 0.0
    assert out["read"].item() > 0
    assert torch.isfinite(out["loss"])


def test_training_step_mlm_only_zero_read():
    model = ReadModel(_tiny_model_config())
    batch = _make_batch(p_mlm=1.0, p_reads=0.0)
    out = training_step(model, batch)
    assert out["read"].item() == 0.0
    assert out["mlm"].item() > 0
    assert torch.isfinite(out["loss"])


def test_training_step_no_populations_returns_zero():
    """Degenerate batch with neither population active -> total loss is exactly 0."""
    model = ReadModel(_tiny_model_config())
    batch = _make_batch(p_mlm=0.0, p_reads=0.0)
    out = training_step(model, batch)
    assert out["loss"].item() == 0.0


def test_training_step_alpha_beta_weighting():
    """Doubling beta should at least double the read contribution to total."""
    model = ReadModel(_tiny_model_config())
    batch = _make_batch(p_mlm=0.0, p_reads=1.0)
    out_1 = training_step(model, batch, alpha=1.0, beta=1.0)
    out_2 = training_step(model, batch, alpha=1.0, beta=2.0)
    assert torch.isclose(out_2["loss"], 2 * out_1["loss"], rtol=1e-5)


def test_training_step_backward_flows_to_encoder():
    model = ReadModel(_tiny_model_config())
    batch = _make_batch(p_mlm=1.0, p_reads=1.0)
    loss = training_step(model, batch)["loss"]
    loss.backward()
    assert model.embed.weight.grad is not None
    assert torch.isfinite(model.embed.weight.grad).all()


def test_loss_decreases_on_overfit_batch():
    """Sanity: optimising a single batch should reduce the loss."""
    torch.manual_seed(0)
    model = ReadModel(_tiny_model_config())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batch = _make_batch(p_mlm=1.0, p_reads=1.0, seed=0)

    losses = []
    for _ in range(40):
        out = training_step(model, batch)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        losses.append(out["loss"].item())
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


def test_trainer_runs_one_iteration(tmp_path):
    """Smoke test the Trainer.train() loop on CPU."""
    train_cfg = TrainingConfig(
        batch_size=2, max_steps=2, eval_interval=1000, lr=1e-3,
        log_interval=1000, warmup_steps=0, device="cpu",
        checkpoint_dir=str(tmp_path / "ckpts"),
    )
    model = ReadModel(_tiny_model_config())
    batches = [_make_batch(p_mlm=1.0, p_reads=1.0, seed=k) for k in range(4)]
    trainer = Trainer(model, batches, None, train_cfg, torch.device("cpu"))
    trainer.train()
    assert trainer.step == train_cfg.max_steps


def test_checkpoint_round_trip(tmp_path):
    cfg = TrainingConfig(
        batch_size=2, max_steps=1, eval_interval=1000, lr=1e-3,
        log_interval=1000, warmup_steps=0, device="cpu",
        checkpoint_dir=str(tmp_path / "ckpts"),
    )
    mcfg = _tiny_model_config()
    model = ReadModel(mcfg)
    batches = [_make_batch(p_mlm=1.0, p_reads=1.0, seed=k) for k in range(2)]
    trainer = Trainer(model, batches, None, cfg, torch.device("cpu"))
    path = str(tmp_path / "ckpts" / "x.pt")
    trainer.save_checkpoint(path)

    model2 = ReadModel(mcfg)
    trainer2 = Trainer(model2, batches, None, cfg, torch.device("cpu"))
    trainer2.load_checkpoint(path)
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
        assert k1 == k2
        assert torch.allclose(v1, v2), f"State mismatch at {k1}"


def test_lr_lambda_warmup_and_decay():
    fn = Trainer._lr_lambda(warmup_steps=10, total_steps=100)
    assert fn(0) > 0           # never exactly 0 (avoid no-op first step)
    assert abs(fn(10) - 1.0) < 1e-6
    fn0 = Trainer._lr_lambda(warmup_steps=0, total_steps=100)
    assert abs(fn0(0) - 1.0) < 1e-6
    assert abs(fn0(100) - 0.0) < 1e-6
