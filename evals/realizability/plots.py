"""Per-window diagnostic plots for the realizability eval.

`plot_deconv_examples` draws a 3 × K grid (one column per window): the
predicted coverage, the recovered read-start density ρ̂ from inverse
filtering with negative regions shaded, and the real coverage. Used to
visually corroborate the per-window neg_mass_fraction score.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sps

from .kernel import survival_kernel


def _row(pred: np.ndarray, lengths: np.ndarray, real: np.ndarray):
    K = survival_kernel(lengths)
    rho = sps.lfilter(np.array([1.0]), K.astype(np.float64), pred.astype(np.float64))
    return pred, rho, real


def plot_deconv_examples(
    sample_idxs: Iterable[int],
    pred_by_idx: dict[int, np.ndarray],
    lengths_by_idx: dict[int, np.ndarray],
    real_by_idx: dict[int, np.ndarray],
    metrics_by_idx: dict[int, dict[str, float]],
    out_path: Path,
    title: str = "",
) -> None:
    sample_idxs = list(sample_idxs)
    K = len(sample_idxs)
    if K == 0:
        return

    fig, axes = plt.subplots(3, K, figsize=(3.2 * K, 7.5), sharex="col")
    if K == 1:
        axes = axes[:, None]

    for col, sidx in enumerate(sample_idxs):
        pred, rho, real = _row(
            pred_by_idx[sidx], lengths_by_idx[sidx], real_by_idx[sidx]
        )
        x = np.arange(pred.size)

        m = metrics_by_idx[sidx]
        head = (
            f"win {sidx}  n={int(m.get('n_reads', 0))}\n"
            f"neg={m.get('neg_mass_fraction', float('nan')):.2f}  "
            f"r={m.get('pearson_r', float('nan')):.2f}"
        )

        ax0 = axes[0, col]
        ax0.plot(x, pred, lw=0.6, color="steelblue")
        ax0.set_title(head, fontsize=9)
        ax0.fill_between(x, pred, 0, color="steelblue", alpha=0.15)

        ax1 = axes[1, col]
        ax1.plot(x, rho, lw=0.5, color="dimgray")
        ax1.axhline(0, lw=0.4, color="black")
        ax1.fill_between(x, rho, 0, where=(rho < 0), color="crimson", alpha=0.55)
        ax1.fill_between(x, rho, 0, where=(rho >= 0), color="dimgray", alpha=0.15)

        ax2 = axes[2, col]
        ax2.plot(x, real, lw=0.6, color="darkorange")
        ax2.fill_between(x, real, 0, color="darkorange", alpha=0.15)
        ax2.set_xlabel("position (bp)")

    axes[0, 0].set_ylabel("Yorzoi pred\ncoverage")
    axes[1, 0].set_ylabel("recovered ρ̂\n(red = negative)")
    axes[2, 0].set_ylabel("real coverage")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
