"""Training loop for CoasterModel."""
from __future__ import annotations

import itertools
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from coaster.model.transformer import CoasterModel
from coaster.model.config import TrainingConfig
from coaster.tokenizer import RNATokenizer

try:
    import wandb
except ImportError:
    wandb = None


class Trainer:
    def __init__(
        self,
        model: CoasterModel,
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

        # AMP only on CUDA; MPS uses float32
        self.use_amp = device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        if wandb is not None and config.wandb_project:
            wandb.init(project=config.wandb_project, config=vars(config), save_code=False)
            wandb.watch(model, log=None)

    @staticmethod
    def _lr_lambda(warmup_steps: int, total_steps: int):
        def fn(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return fn

    def _forward_loss(self, batch: dict) -> torch.Tensor:
        dna_ids = batch["dna_ids"].to(self.device)
        rna_input = batch["rna_input"].to(self.device)
        rna_target = batch["rna_target"].to(self.device)
        rna_pad_mask = batch["rna_pad_mask"].to(self.device)

        if self.use_amp:
            with torch.autocast("cuda"):
                logits = self.model(dna_ids, rna_input, tgt_padding_mask=rna_pad_mask)
        else:
            logits = self.model(dna_ids, rna_input, tgt_padding_mask=rna_pad_mask)

        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            rna_target.reshape(-1),
            ignore_index=RNATokenizer.PAD,
        )

    def train(self) -> None:
        self.best_val_loss = float("inf")
        try:
            self.model.train()
            for batch in itertools.islice(itertools.cycle(self.train_loader), self.config.max_steps):
                loss = self._forward_loss(batch)
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if self.step % self.config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"step {self.step:7d} | loss {loss.item():.4f} | lr {lr:.2e}")
                    if wandb is not None and self.config.wandb_project:
                        wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=self.step)

                if self.step > 0 and self.step % self.config.eval_interval == 0:
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

                self.step += 1

        except KeyboardInterrupt:
            print("\nInterrupted — saving last.pt …")
            self.save_checkpoint(os.path.join(self.config.checkpoint_dir, "last.pt"))
            print("Saved. Re-raising KeyboardInterrupt.")
            raise

    def _eval(self) -> float:
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                total += self._forward_loss(batch).item()
                count += 1
        self.model.train()
        return total / max(1, count)

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "step": self.step,
                "epoch": self.epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt.get("step", 0)
        self.epoch = ckpt.get("epoch", 0)
