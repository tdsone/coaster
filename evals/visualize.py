#!/usr/bin/env python3
"""Visualize real vs synthetic coverage side by side.

Usage:
    # Plot 12 random val windows
    uv run python evals/visualize.py \\
        --real-coverage evals/output/real_coverage \\
        --synth-coverage evals/output/synth_coverage \\
        --samples data/samples_yeast.parquet \\
        --fold val

    # Plot specific windows by sample_idx
    uv run python evals/visualize.py ... --idx 42 137 801
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr


def load_coverage(cov_dir: Path, sample_idx: int) -> np.ndarray | None:
    path = cov_dir / f"{sample_idx}.npy"
    return np.load(str(path)) if path.exists() else None


def smooth(x: np.ndarray, window: int = 10) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def normalize(x: np.ndarray) -> np.ndarray:
    s = x.sum()
    return x / s if s > 0 else x


def plot_windows(
    sample_idxs: list[int],
    samples: pd.DataFrame,
    real_cov_dir: Path,
    synth_cov_dir: Path,
    smooth_window: int = 1,
    normalize_counts: bool = True,
    ncols: int = 3,
    output: str | None = None,
) -> None:
    n = len(sample_idxs)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(ncols * 5, nrows * 3))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.35)

    for i, idx in enumerate(sample_idxs):
        real = load_coverage(real_cov_dir, idx)
        synth = load_coverage(synth_cov_dir, idx)
        row_data = samples.loc[idx] if idx in samples.index else None

        ax = fig.add_subplot(gs[i // ncols, i % ncols])

        if real is None or synth is None:
            ax.set_title(f"#{idx} — missing data")
            ax.axis("off")
            continue

        # Align lengths (synth covers full input_sequence, real covers coverage window)
        min_len = min(len(real), len(synth))
        real = real[:min_len]
        synth = synth[:min_len]

        if normalize_counts:
            real = normalize(real)
            synth = normalize(synth)

        x = np.arange(min_len)
        r_smooth = smooth(real, smooth_window)
        s_smooth = smooth(synth, smooth_window)

        ax.fill_between(x, r_smooth, alpha=0.5, color="#2196F3", label="real")
        ax.fill_between(x, s_smooth, alpha=0.5, color="#FF5722", label="synth")

        # Pearson on raw counts
        if real.sum() > 0 and synth.sum() > 0:
            r, _ = pearsonr(real, synth)
            r_str = f"r={r:.3f}"
        else:
            r_str = "r=n/a"

        title = f"#{idx}"
        if row_data is not None:
            title += f"  {row_data.get('chr', '')} {row_data.get('strand', '')}"
        ax.set_title(f"{title}\n{r_str}", fontsize=8)
        ax.set_xlabel("position (bp)", fontsize=7)
        ax.set_ylabel("coverage", fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide unused subplots
    for j in range(n, nrows * ncols):
        fig.add_subplot(gs[j // ncols, j % ncols]).axis("off")

    fig.suptitle("Real (blue) vs Synthetic (orange) coverage", fontsize=11, y=1.01)

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


def plot_correlation_scatter(
    sample_idxs: list[int],
    real_cov_dir: Path,
    synth_cov_dir: Path,
    output: str | None = None,
) -> None:
    """Scatter plot of per-window Pearson r values vs real total coverage."""
    results = []
    for idx in sample_idxs:
        real = load_coverage(real_cov_dir, idx)
        synth = load_coverage(synth_cov_dir, idx)
        if real is None or synth is None:
            continue
        min_len = min(len(real), len(synth))
        real, synth = real[:min_len], synth[:min_len]
        if real.sum() == 0 or synth.sum() == 0:
            continue
        r, _ = pearsonr(real, synth)
        results.append({"sample_idx": idx, "pearson": r, "real_total": real.sum()})

    if not results:
        print("No valid windows to plot.")
        return

    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["real_total"], df["pearson"], alpha=0.4, s=10, color="#2196F3")
    ax.axhline(df["pearson"].median(), color="gray", linestyle="--", linewidth=1,
               label=f"median r={df['pearson'].median():.3f}")
    ax.set_xlabel("real total coverage (reads)")
    ax.set_ylabel("Pearson r")
    ax.set_title("Per-window correlation vs real coverage depth")
    ax.legend(fontsize=8)

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-coverage", default="evals/output/real_coverage")
    parser.add_argument("--synth-coverage", default="evals/output/synth_coverage")
    parser.add_argument("--samples", default="data/samples_yeast.parquet")
    parser.add_argument("--fold", default="val")
    parser.add_argument("--idx", nargs="+", type=int, default=None,
                        help="Specific sample_idx values to plot (default: random 12)")
    parser.add_argument("--n", type=int, default=12,
                        help="Number of random windows to plot")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Smoothing window size in bp (1 = no smoothing)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip normalizing coverage to sum to 1")
    parser.add_argument("--scatter", action="store_true",
                        help="Plot correlation scatter instead of coverage tracks")
    parser.add_argument("--output", default=None, help="Save figure to file instead of showing")
    args = parser.parse_args()

    real_cov_dir = Path(args.real_coverage)
    synth_cov_dir = Path(args.synth_coverage)

    samples = pd.read_parquet(args.samples)
    if args.fold:
        samples = samples[samples["fold"] == args.fold]

    if args.idx:
        idxs = args.idx
    else:
        available = [
            idx for idx in samples.index
            if (real_cov_dir / f"{idx}.npy").exists()
            and (synth_cov_dir / f"{idx}.npy").exists()
        ]
        rng = np.random.default_rng(0)
        idxs = rng.choice(available, size=min(args.n, len(available)), replace=False).tolist()

    if args.scatter:
        plot_correlation_scatter(idxs, real_cov_dir, synth_cov_dir, output=args.output)
    else:
        plot_windows(idxs, samples, real_cov_dir, synth_cov_dir,
                     smooth_window=args.smooth, normalize_counts=not args.no_normalize,
                     output=args.output)


if __name__ == "__main__":
    main()
