"""Second-pass analysis of the per-window realizability scores.

The first-pass driver (`run_realizability.py`) writes `per_window.parquet`.
This script consumes it to investigate two follow-up questions:

  1. Magnitude error: how often does Yorzoi grossly miss the total signal in
     a window, and how does that interact with the realizability scorers?
  2. Independence: is `neg_mass_excess` just a proxy for magnitude error, or
     does it carry information about predictor pathology that pred-vs-real
     total-counts mismatch alone would not see?

Outputs go to `evals/output/realizability/plots/`:
  - magnitude_error_distribution.png
  - neg_mass_vs_magnitude.png
  - stratified_correlation.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = ROOT / "evals" / "output" / "realizability" / "per_window.parquet"
DEFAULT_PLOTS = ROOT / "evals" / "output" / "realizability" / "plots"


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Spearman partial correlation of x and y controlling for z.

    Computed by rank-transforming all three variables and taking the Pearson
    correlation of the OLS residuals of x and y regressed on ranked z.
    Returns (rho, p) where p is the Pearson p-value of the residuals.
    """
    from scipy.stats import pearsonr

    rx = rankdata(x); ry = rankdata(y); rz = rankdata(z)
    z1 = np.column_stack([np.ones_like(rz), rz])
    bx, *_ = np.linalg.lstsq(z1, rx, rcond=None)
    by, *_ = np.linalg.lstsq(z1, ry, rcond=None)
    ex = rx - z1 @ bx
    ey = ry - z1 @ by
    rho, p = pearsonr(ex, ey)
    return float(rho), float(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    ap.add_argument("--plots-dir", default=str(DEFAULT_PLOTS))
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    plots = Path(args.plots_dir)
    plots.mkdir(parents=True, exist_ok=True)

    # log10(pred_total / real_total) is undefined when either side is 0;
    # add a tiny floor so the small minority of windows stay on the axis.
    eps = 1e-3
    df["log_ratio"] = np.log10((df["pred_total"] + eps) / (df["real_total"] + eps))
    df["abs_log_ratio"] = df["log_ratio"].abs()
    df["err"] = 1.0 - df["pearson_r"]

    # ----- Magnitude error distribution -----
    lr = df["log_ratio"].dropna().to_numpy()
    print(f"log10(pred/real) summary  (n = {len(lr)}):")
    print(f"  median       = {np.median(lr):+.3f}")
    print(f"  mean         = {np.mean(lr):+.3f}")
    print(f"  std          = {np.std(lr):.3f}")
    print(f"  p10 .. p90   = {np.percentile(lr, 10):+.3f} .. {np.percentile(lr, 90):+.3f}")
    print(f"  windows with |log_ratio| > 1 (10x off): {(np.abs(lr) > 1).sum()}  "
          f"({100*(np.abs(lr) > 1).mean():.1f}%)")
    print(f"  windows with log_ratio  < -1 (underpred by 10x): {(lr < -1).sum()}  "
          f"({100*(lr < -1).mean():.1f}%)")
    print(f"  windows with log_ratio  > +1 (overpred  by 10x): {(lr > 1).sum()}  "
          f"({100*(lr > 1).mean():.1f}%)")
    print()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lr, bins=60, color="steelblue", edgecolor="white")
    ax.axvline(0.0, color="black", lw=0.6, linestyle="--", label="pred = real")
    ax.axvline(-1.0, color="crimson", lw=0.4, linestyle=":", label="10× under/over")
    ax.axvline(1.0, color="crimson", lw=0.4, linestyle=":")
    ax.set_xlabel("log₁₀(pred_total / real_total)")
    ax.set_ylabel("# windows")
    ax.set_title(f"Yorzoi vs real total-signal magnitude — n = {len(lr)}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plots / "magnitude_error_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {plots / 'magnitude_error_distribution.png'}")

    # ----- neg_mass vs magnitude (colored by error) -----
    sub = df.dropna(subset=["log_ratio", "neg_mass_fraction_pred", "err"])
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        sub["log_ratio"], sub["neg_mass_fraction_pred"],
        c=sub["err"].clip(0, 2), cmap="magma_r", s=14, alpha=0.7, edgecolors="none",
    )
    real_null_med = float(df["neg_mass_fraction_real"].median())
    ax.axhline(real_null_med, color="darkorange", lw=0.8, linestyle="--",
               label=f"median real-coverage null = {real_null_med:.2f}")
    ax.axvline(0.0, color="black", lw=0.5, linestyle="--")
    ax.set_xlabel("log₁₀(pred_total / real_total)")
    ax.set_ylabel("neg_mass_fraction (Yorzoi)")
    rho, p = spearmanr(sub["log_ratio"], sub["neg_mass_fraction_pred"])
    ax.set_title(f"Yorzoi neg_mass vs magnitude error\nSpearman ρ = {rho:+.3f}  p = {p:.1e}")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("1 − Pearson r")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "neg_mass_vs_magnitude.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {plots / 'neg_mass_vs_magnitude.png'}")
    print()

    # ----- Headline correlations and their decomposition -----
    sub = df.dropna(subset=["err", "neg_mass_fraction_excess", "log_ratio"])
    rho_mag, p_mag = spearmanr(sub["abs_log_ratio"], sub["err"])
    rho_nm,  p_nm  = spearmanr(sub["neg_mass_fraction_excess"], sub["err"])
    rho_par, p_par = partial_spearman(
        sub["neg_mass_fraction_excess"].to_numpy(),
        sub["err"].to_numpy(),
        sub["abs_log_ratio"].to_numpy(),
    )
    rho_tv,  p_tv  = spearmanr(sub["tv_ratio_excess"], sub["err"])
    rho_tv_par, p_tv_par = partial_spearman(
        sub["tv_ratio_excess"].to_numpy(),
        sub["err"].to_numpy(),
        sub["abs_log_ratio"].to_numpy(),
    )
    print("Spearman correlations with prediction error (1 - Pearson r):")
    print(f"  |log_ratio|                             ρ = {rho_mag:+.3f}  p = {p_mag:.1e}")
    print(f"  neg_mass_fraction_excess                ρ = {rho_nm:+.3f}  p = {p_nm:.1e}")
    print(f"  neg_mass_fraction_excess | |log_ratio|  ρ = {rho_par:+.3f}  p = {p_par:.1e}")
    print(f"  tv_ratio_excess                         ρ = {rho_tv:+.3f}  p = {p_tv:.1e}")
    print(f"  tv_ratio_excess | |log_ratio|           ρ = {rho_tv_par:+.3f}  p = {p_tv_par:.1e}")
    print()

    # ----- Stratified correlations by magnitude-error bin -----
    bins = [-np.inf, -1.0, -0.3, 0.3, 1.0, np.inf]
    labels = ["<-1 (under 10x)", "[-1,-0.3)", "[-0.3,0.3]", "(0.3,1]", ">1 (over 10x)"]
    sub["mag_bin"] = pd.cut(sub["log_ratio"], bins=bins, labels=labels, include_lowest=True)
    print("Stratified correlations of neg_mass_excess and tv_ratio_excess with 1-r, per |log_ratio| bin:")
    rows = []
    for lbl in labels:
        g = sub[sub["mag_bin"] == lbl]
        if len(g) < 10:
            rows.append((lbl, len(g), float("nan"), float("nan"), float("nan"), float("nan")))
            continue
        rho_n, p_n = spearmanr(g["neg_mass_fraction_excess"], g["err"])
        rho_t, p_t = spearmanr(g["tv_ratio_excess"], g["err"])
        rows.append((lbl, len(g), rho_n, p_n, rho_t, p_t))
        print(f"  {lbl:<20s}  n={len(g):>4d}  "
              f"ρ(neg_mass)={rho_n:+.3f} p={p_n:.1e}  "
              f"ρ(tv_ratio)={rho_t:+.3f} p={p_t:.1e}")

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    xs = np.arange(len(rows))
    width = 0.35
    nm_rhos = [r[2] for r in rows]
    tv_rhos = [r[4] for r in rows]
    ax.bar(xs - width/2, nm_rhos, width, label="neg_mass_excess", color="steelblue")
    ax.bar(xs + width/2, tv_rhos, width, label="tv_ratio_excess", color="darkorange")
    ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("Spearman ρ with 1 − Pearson r")
    ax.axhline(0, color="black", lw=0.5)
    for x, (_, n, *_rest) in zip(xs, rows):
        ax.text(x, ax.get_ylim()[1] * 0.95, f"n={n}", ha="center", va="top", fontsize=8)
    ax.set_title("Realizability vs error, stratified by magnitude-error bin")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "stratified_correlation.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {plots / 'stratified_correlation.png'}")


if __name__ == "__main__":
    main()
