#!/usr/bin/env python3
"""Compare synthetic bigwig coverage to real per-window coverage vectors.

Loads:
  - Real coverage:      evals/output/real_coverage/{sample_idx}.npy
  - Synthetic bigwig:   evals/output/bigwigs/synthetic_forward.bw  (or reverse)

Outputs per-window and aggregate Pearson/Spearman correlations.

Usage:
    uv run python evals/compare.py \\
        --real-coverage evals/output/real_coverage \\
        --bigwig evals/output/bigwigs/synthetic_forward.bw \\
        --samples data/samples_yeast.parquet \\
        --fold val
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def load_bigwig_coverage(bw_path: str, chrom: str, start: int, end: int) -> np.ndarray:
    """Extract per-base coverage from a bigwig for a genomic interval."""
    import pyBigWig
    bw = pyBigWig.open(bw_path)
    vals = bw.values(chrom, start, end, numpy=True)
    bw.close()
    vals = np.nan_to_num(vals, nan=0.0).astype(np.float32)
    return vals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-coverage", default="evals/output/real_coverage")
    parser.add_argument("--bigwig", required=True, help="Synthetic bigwig (.bw)")
    parser.add_argument("--samples", default="data/samples_yeast.parquet")
    parser.add_argument("--fold", default="val")
    parser.add_argument("--output", default=None, help="Save per-window results to CSV")
    args = parser.parse_args()

    samples = pd.read_parquet(args.samples)
    if args.fold:
        samples = samples[samples["fold"] == args.fold]
    print(f"Comparing {len(samples)} windows (fold={args.fold or 'all'})")

    real_cov_dir = Path(args.real_coverage)
    results = []

    for sample_idx, row in samples.iterrows():
        real_path = real_cov_dir / f"{sample_idx}.npy"
        if not real_path.exists():
            continue

        real = np.load(str(real_path))

        # Synthetic coverage comes from the bigwig, where each window is a
        # "chromosome" named by sample_idx
        try:
            synth = load_bigwig_coverage(
                args.bigwig,
                chrom=str(sample_idx),
                start=0,
                end=len(real),
            )
        except Exception:
            continue

        if real.sum() == 0 or synth.sum() == 0:
            pearson, spearman = float("nan"), float("nan")
        else:
            pearson = pearsonr(real, synth).statistic
            spearman = spearmanr(real, synth).statistic

        results.append({
            "sample_idx": sample_idx,
            "chr": row["chr"],
            "strand": row["strand"],
            "fold": row["fold"],
            "real_total": float(real.sum()),
            "synth_total": float(synth.sum()),
            "pearson": pearson,
            "spearman": spearman,
        })

    df = pd.DataFrame(results)
    n = len(df)
    n_valid = df["pearson"].notna().sum()
    print(f"\nWindows evaluated: {n}  ({n_valid} with non-zero coverage in both)")
    print(f"Pearson  — mean: {df['pearson'].mean():.4f}  median: {df['pearson'].median():.4f}")
    print(f"Spearman — mean: {df['spearman'].mean():.4f}  median: {df['spearman'].median():.4f}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Per-window results saved to {args.output}")


if __name__ == "__main__":
    main()
