"""Score Yorzoi predicted coverage profiles for read-realizability,
calibrated against the matching real coverage as the null.

Loads:
  - data/yorzoi_srr21628668_fwd_test.npz       Yorzoi predictions (N, 3000)
  - data/test_read_lengths.parquet             per-window insert lengths
  - evals/output/real_coverage/{i}.npy         real coverage (length 2999)
  - data/samples_yeast.parquet                 fold + start_seq/start_coverage offsets

For each test window:
  - Score the Yorzoi prediction with score_window → *_pred columns
  - Score the real coverage with score_window     → *_real columns
  - Pearson r between pred and real
  - Calibrated "excess" metrics = pred − real for each scorer

The calibrated headline metric is `neg_mass_excess`: the share of recovered
read-start density that goes negative *above* the finite-sample noise floor
established by real coverage scored against its own empirical kernel.

Yorzoi's center-3000 output corresponds to genomic positions
[start_seq + 999, start_seq + 3999) for the typical-case window
(offset_start = 1000). The real coverage .npy files cover
[start_seq + 1000, start_seq + 3999) = 2999 bp — one position shorter than
Yorzoi at the 5' end. So we drop pred[0] when joining the two.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from evals.realizability import score_window
from evals.realizability.plots import plot_deconv_examples

ROOT = Path(__file__).resolve().parents[1]

SCORERS = ["neg_mass_fraction", "deconv_residual", "tv_ratio",
           "tv_excess", "hf_energy_fraction"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "data" / "yorzoi_srr21628668_fwd_test.npz"))
    ap.add_argument("--samples", default=str(ROOT / "data" / "samples_yeast.parquet"))
    ap.add_argument("--lengths", default=str(ROOT / "data" / "test_read_lengths.parquet"))
    ap.add_argument("--real-cov-dir", default=str(ROOT / "evals" / "output" / "real_coverage"))
    ap.add_argument("--out-dir", default=str(ROOT / "evals" / "output" / "realizability"))
    ap.add_argument("--limit", type=int, default=None, help="Score only first N windows (debug)")
    ap.add_argument("--method", default="lfilter", choices=["lfilter", "nnls"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load inputs -----
    z = np.load(args.pred)
    pred_idx = z["sample_idx"].astype(np.int64)
    pred_arr = z["pred"].astype(np.float32)
    print(f"Yorzoi preds: {pred_arr.shape}, sample_idx range {pred_idx.min()}..{pred_idx.max()}")

    samples = pd.read_parquet(args.samples)
    print(f"Samples: {len(samples)} total; typical offset (1000): "
          f"{((samples['start_coverage'] - samples['start_seq']) == 1000).sum()}")

    lengths_df = pd.read_parquet(args.lengths)
    lengths_by_idx: dict[int, np.ndarray] = {
        int(k): grp["length"].to_numpy(dtype=np.int64)
        for k, grp in lengths_df.groupby("sample_idx")
    }
    print(f"Read-length dist available for {len(lengths_by_idx)} windows")

    real_cov_dir = Path(args.real_cov_dir)

    # ----- Score each window: pred AND real -----
    rows: list[dict] = []
    pred_by_idx: dict[int, np.ndarray] = {}
    real_by_idx: dict[int, np.ndarray] = {}
    lengths_used: dict[int, np.ndarray] = {}
    skipped = {"atypical_offset": 0, "missing_lengths": 0,
               "missing_real": 0, "shape_mismatch": 0}
    iter_idx = pred_idx if args.limit is None else pred_idx[: args.limit]
    t0 = time.time()

    for sidx in tqdm(iter_idx, desc="windows"):
        row = samples.loc[int(sidx)]
        if int(row["start_coverage"]) - int(row["start_seq"]) != 1000:
            skipped["atypical_offset"] += 1
            continue
        lengths = lengths_by_idx.get(int(sidx))
        if lengths is None or lengths.size == 0:
            skipped["missing_lengths"] += 1
            continue
        real_path = real_cov_dir / f"{int(sidx)}.npy"
        if not real_path.exists():
            skipped["missing_real"] += 1
            continue
        real = np.load(real_path).astype(np.float32)
        pred = pred_arr[np.where(pred_idx == sidx)[0][0]][1:]
        if pred.shape != real.shape:
            skipped["shape_mismatch"] += 1
            continue

        pred_scores = score_window(pred, lengths, method=args.method)
        real_scores = score_window(real, lengths, method=args.method)

        merged: dict[str, float] = {}
        for k in SCORERS:
            merged[f"{k}_pred"] = pred_scores[k]
            merged[f"{k}_real"] = real_scores[k]
            merged[f"{k}_excess"] = pred_scores[k] - real_scores[k]
        merged["pearson_r"] = (
            float(np.corrcoef(pred.astype(np.float64), real.astype(np.float64))[0, 1])
            if pred.std() > 0 and real.std() > 0 else float("nan")
        )
        merged["sample_idx"] = int(sidx)
        merged["n_reads"] = int(lengths.size)
        merged["pred_total"] = float(pred.sum())
        merged["real_total"] = float(real.sum())
        rows.append(merged)

        pred_by_idx[int(sidx)] = pred
        real_by_idx[int(sidx)] = real
        lengths_used[int(sidx)] = lengths

    dt = time.time() - t0
    print(f"\nScored {len(rows)} windows in {dt:.1f}s ({dt / max(len(rows), 1):.2f}s/window)")
    print(f"Skipped: {skipped}")

    df = pd.DataFrame(rows)
    out_pq = out_dir / "per_window.parquet"
    df.to_parquet(out_pq, index=False)
    print(f"Wrote {out_pq} ({len(df)} rows, {len(df.columns)} columns)")

    # ----- Headline scatter: calibrated implausibility vs error -----
    if {"neg_mass_fraction_excess", "pearson_r"}.issubset(df.columns):
        plot_df = df.dropna(subset=["neg_mass_fraction_excess", "pearson_r"])
        if len(plot_df) > 0:
            x = plot_df["neg_mass_fraction_excess"].to_numpy()
            y = (1.0 - plot_df["pearson_r"]).to_numpy()
            c = np.log10(plot_df["n_reads"].to_numpy().clip(min=1))
            from scipy.stats import spearmanr
            rho, p = spearmanr(x, y)
            fig, ax = plt.subplots(figsize=(7, 5))
            sc = ax.scatter(x, y, c=c, s=14, alpha=0.55, cmap="viridis", edgecolors="none")
            ax.axvline(0.0, lw=0.5, color="black", linestyle="--")
            ax.set_xlabel("neg_mass_fraction excess  (Yorzoi − real-coverage null)")
            ax.set_ylabel("1 − Pearson r  (Yorzoi vs real coverage)")
            ax.set_title(
                f"Calibrated realizability vs prediction error — n={len(plot_df)}\n"
                f"Spearman ρ = {rho:.3f}  (p = {p:.1e})"
            )
            cb = fig.colorbar(sc, ax=ax); cb.set_label("log₁₀(n_reads)")
            fig.tight_layout()
            out_png = plots_dir / "headline_scatter.png"
            fig.savefig(out_png, dpi=150); plt.close(fig)
            print(f"Wrote {out_png}  (Spearman ρ = {rho:.3f}, p = {p:.2e})")

    # ----- Null vs Yorzoi distribution overlays -----
    fig, axes = plt.subplots(1, len(SCORERS), figsize=(3.4 * len(SCORERS), 3.6))
    for ax, s in zip(axes, SCORERS):
        a = df[f"{s}_real"].dropna()
        b = df[f"{s}_pred"].dropna()
        if len(a) == 0 or len(b) == 0:
            continue
        lo = float(min(a.min(), b.min()))
        hi = float(max(a.quantile(0.99), b.quantile(0.99)))
        bins = np.linspace(lo, hi + 1e-9, 40)
        ax.hist(a, bins=bins, alpha=0.55, color="darkorange",
                label=f"real (median {a.median():.2f})", density=True)
        ax.hist(b, bins=bins, alpha=0.55, color="steelblue",
                label=f"Yorzoi (median {b.median():.2f})", density=True)
        ax.set_title(s, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Null calibration: realizability scores, real coverage vs Yorzoi", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = plots_dir / "null_distribution.png"
    fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"Wrote {out_png}")

    # ----- Worst- and best-offender deconv examples (by calibrated excess) -----
    if "neg_mass_fraction_excess" in df.columns:
        df_sorted = df.dropna(subset=["neg_mass_fraction_excess"]).sort_values(
            "neg_mass_fraction_excess"
        )
        worst = df_sorted.tail(4)[::-1]
        best = df_sorted.head(4)
        metrics_by_idx = {int(r["sample_idx"]): r.to_dict() for _, r in df.iterrows()}
        # Provide Pearson/neg_mass under the old keys plot_deconv_examples expects.
        for idx, m in metrics_by_idx.items():
            m["neg_mass_fraction"] = m.get("neg_mass_fraction_pred", float("nan"))

        plot_deconv_examples(
            [int(s) for s in worst["sample_idx"]],
            pred_by_idx, lengths_used, real_by_idx, metrics_by_idx,
            out_path=plots_dir / "deconv_worst.png",
            title="Highest neg_mass excess over null (Yorzoi − real)",
        )
        print(f"Wrote {plots_dir / 'deconv_worst.png'}")

        plot_deconv_examples(
            [int(s) for s in best["sample_idx"]],
            pred_by_idx, lengths_used, real_by_idx, metrics_by_idx,
            out_path=plots_dir / "deconv_best.png",
            title="Lowest neg_mass excess over null (most-realizable Yorzoi predictions)",
        )
        print(f"Wrote {plots_dir / 'deconv_best.png'}")

    # ----- tv_ratio scatter (pred vs real, unchanged) -----
    if {"tv_ratio_pred", "tv_ratio_real"}.issubset(df.columns):
        plot_df = df.dropna(subset=["tv_ratio_pred", "tv_ratio_real"])
        if len(plot_df) > 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(plot_df["tv_ratio_real"], plot_df["tv_ratio_pred"],
                       s=10, alpha=0.45, color="steelblue", edgecolors="none")
            lim = float(max(plot_df["tv_ratio_pred"].max(), plot_df["tv_ratio_real"].max()) * 1.05)
            ax.plot([0, lim], [0, lim], lw=0.6, color="black", linestyle="--", label="y = x")
            ax.axhline(1.0, lw=0.5, color="red", linestyle=":",
                       label="tv_ratio = 1 (impossibility bound)")
            ax.axvline(1.0, lw=0.5, color="red", linestyle=":")
            ax.set_xlabel("tv_ratio(real coverage)")
            ax.set_ylabel("tv_ratio(Yorzoi prediction)")
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            med_p = float(plot_df["tv_ratio_pred"].median())
            med_r = float(plot_df["tv_ratio_real"].median())
            ax.set_title(
                f"TV ratio: Yorzoi vs real — n={len(plot_df)}\n"
                f"median pred = {med_p:.2f}    median real = {med_r:.2f}"
            )
            ax.legend(loc="lower right", fontsize=9)
            fig.tight_layout()
            out_png = plots_dir / "tv_ratio_compare.png"
            fig.savefig(out_png, dpi=150); plt.close(fig)
            print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
