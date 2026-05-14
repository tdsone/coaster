"""Training loop for the pointer-factorized read model.

Implements the in-batch task-mixed loss from spec §4:

    L = α · L_MLM (over samples with any masked positions)
      + β · L_read (over samples with `has_reads=True`),

where L_read = CE(σ) + CE(s | σ) + CE(e | s, σ).
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from coaster.model.config import TrainingConfig
from coaster.model.read_model import ReadModel

try:
    import wandb
except ImportError:
    wandb = None


# ---------------------------------------------------------------------------
def training_step(
    model: ReadModel,
    batch: dict[str, torch.Tensor],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute the joint MLM + read loss for one batch.

    Returns a dict with `loss` (scalar to backward) and per-component scalars
    for logging. Components that don't apply this batch are returned as 0.
    """
    dna = batch["dna_ids"]
    mlm_mask = batch["mlm_mask"]
    target_nucl = batch["target_nucl"]
    has_reads = batch["has_reads"]
    strand_true = batch["strand_true"]
    start_true = batch["start_true"]
    end_true = batch["end_true"]
    assay = batch.get("assay")
    cell_type = batch.get("cell_type")
    species = batch.get("species")

    h_cls, h_seq = model.encode(dna, assay=assay, cell_type=cell_type, species=species)

    zero = h_seq.new_zeros(())
    # ---- MLM loss --------------------------------------------------------
    if mlm_mask.any():
        logits_m = model.mlm_logits(h_seq)                                # (B, L, V)
        mlm_loss = F.cross_entropy(logits_m[mlm_mask], target_nucl[mlm_mask])
    else:
        mlm_loss = zero

    # ---- Read losses -----------------------------------------------------
    if has_reads.any():
        hc = h_cls[has_reads]
        hs = h_seq[has_reads]
        s_true = strand_true[has_reads]
        s = start_true[has_reads]
        e = end_true[has_reads]

        loss_sigma = F.cross_entropy(model.strand_logits(hc), s_true)
        loss_start = F.cross_entropy(model.start_logits(hs, s_true), s)
        loss_end = F.cross_entropy(model.end_logits(hs, s, s_true), e)
        read_loss = loss_sigma + loss_start + loss_end
    else:
        loss_sigma = loss_start = loss_end = read_loss = zero

    total = alpha * mlm_loss + beta * read_loss
    return {
        "loss": total,
        "mlm": mlm_loss.detach(),
        "read": read_loss.detach(),
        "sigma": loss_sigma.detach(),
        "start": loss_start.detach(),
        "end": loss_end.detach(),
    }


# ---------------------------------------------------------------------------
class Trainer:
    def __init__(
        self,
        model: ReadModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.step = 0
        self.epoch = 0

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self._lr_lambda(config.warmup_steps, config.max_steps)
        )

        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        if wandb is not None and config.wandb_project:
            wandb.init(project=config.wandb_project, config=vars(config), save_code=False)
            wandb.watch(model, log=None)

    # --------------------------------------------------------------------
    @staticmethod
    def _lr_lambda(warmup_steps: int, total_steps: int):
        def fn(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return fn

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _forward_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        batch = self._to_device(batch)
        if self.use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = training_step(
                    self.model, batch, alpha=self.config.alpha_mlm, beta=self.config.beta_read
                )
        else:
            out = training_step(
                self.model, batch, alpha=self.config.alpha_mlm, beta=self.config.beta_read
            )
        # Always cast the scalar loss back to float32 for grad scaling stability.
        out["loss"] = out["loss"].float()
        return out

    # --------------------------------------------------------------------
    def train(self) -> None:
        self.best_val_loss = float("inf")
        try:
            self.model.train()
            while self.step < self.config.max_steps:
                for batch in self.train_loader:
                    if self.step >= self.config.max_steps:
                        break

                    out = self._forward_loss(batch)
                    loss = out["loss"]

                    if self.use_amp:
                        assert self.scaler is not None
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if self.step % self.config.log_interval == 0:
                        self._log_train(out, grad_norm)

                    if self.step > 0 and self.step % self.config.eval_interval == 0:
                        self._maybe_eval_and_checkpoint()

                    self.step += 1
                self.epoch += 1

        except KeyboardInterrupt:
            print("\nInterrupted — saving last.pt …")
            self.save_checkpoint(os.path.join(self.config.checkpoint_dir, "last.pt"))
            print("Saved. Re-raising KeyboardInterrupt.")
            raise

    def _log_train(self, out: dict[str, torch.Tensor], grad_norm) -> None:
        lr = self.scheduler.get_last_lr()[0]
        gn = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        print(
            f"step {self.step:7d} | loss {out['loss'].item():.4f} "
            f"(mlm {out['mlm'].item():.4f} read {out['read'].item():.4f}: "
            f"σ {out['sigma'].item():.4f} s {out['start'].item():.4f} e {out['end'].item():.4f}) "
            f"| gnorm {gn:.2f} | lr {lr:.2e}"
        )
        if wandb is not None and self.config.wandb_project:
            wandb.log(
                {
                    "train/loss": out["loss"].item(),
                    "train/mlm": out["mlm"].item(),
                    "train/read": out["read"].item(),
                    "train/sigma": out["sigma"].item(),
                    "train/start": out["start"].item(),
                    "train/end": out["end"].item(),
                    "train/lr": lr,
                    "train/grad_norm": gn,
                },
                step=self.step,
            )

    def _maybe_eval_and_checkpoint(self) -> None:
        if self.val_loader is not None:
            val_loss = self._eval()
            print(f"step {self.step:7d} | val_loss {val_loss:.4f}")
            if wandb is not None and self.config.wandb_project:
                wandb.log({"val/loss": val_loss}, step=self.step)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(os.path.join(self.config.checkpoint_dir, "best.pt"))
                print(f"step {self.step:7d} | saved best.pt (val_loss {val_loss:.4f})")
        self.save_checkpoint(os.path.join(self.config.checkpoint_dir, f"step_{self.step:07d}.pt"))
        self.model.train()

    @torch.no_grad()
    def _eval(self) -> float:
        assert self.val_loader is not None
        self.model.eval()
        total, count = 0.0, 0
        for batch in self.val_loader:
            out = self._forward_loss(batch)
            total += float(out["loss"].item())
            count += 1
        self.model.train()
        return total / max(1, count)

    # --------------------------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "step": self.step,
                "epoch": self.epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        result = self.model.load_state_dict(ckpt["model"], strict=False)
        if result.missing_keys:
            print(f"Checkpoint missing keys (initialised values used): {result.missing_keys}")
        if result.unexpected_keys:
            print(f"Checkpoint has unexpected keys (ignored): {result.unexpected_keys}")
        try:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError as e:
            print(f"Optimizer state incompatible ({e}); starting fresh.")
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step = ckpt.get("step", 0)
        self.epoch = ckpt.get("epoch", 0)
