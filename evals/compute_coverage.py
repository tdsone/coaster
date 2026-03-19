#!/usr/bin/env python3
"""Compute per-base coverage .npy files from real and synthetic reads.

For each sample window:
  - Real reads: loaded from reads.parquet, aligned to input_sequence via exact
    substring search (forward + reverse complement). Coverage is extracted over
    the coverage window [start_coverage - start_seq, end_coverage - start_seq).
    Output: evals/output/real_coverage/{sample_idx}.npy  (length = end_coverage - start_coverage)

  - Synthetic reads: loaded from evals/output/reads/{sample_idx}.fastq, aligned
    to the full input_sequence.
    Output: evals/output/synth_coverage/{sample_idx}.npy  (length = len(input_sequence))

Usage:
    uv run python evals/compute_coverage.py \\
        --samples data/samples_yeast.parquet \\
        --real-reads data/reads.parquet \\
        --synth-reads-dir evals/output/reads \\
        --real-output evals/output/real_coverage \\
        --synth-output evals/output/synth_coverage

    # Only one type:
    uv run python evals/compute_coverage.py --skip-synth
    uv run python evals/compute_coverage.py --skip-real

    # Specific fold:
    uv run python evals/compute_coverage.py --fold val
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


def compute_coverage(reads: list[str], ref: str) -> np.ndarray:
    """Align reads to ref via exact substring search (fwd + RC).

    Reads are assumed to be RNA; U is converted to T before matching.
    Reads that don't match exactly (in either orientation) are skipped (~32%).
    Returns float32 coverage array of length len(ref).
    """
    cov = np.zeros(len(ref), dtype=np.int32)
    for read in reads:
        dna = read.replace("U", "T")
        pos = ref.find(dna)
        if pos == -1:
            pos = ref.find(revcomp(dna))
        if pos != -1:
            end = min(pos + len(dna), len(ref))
            cov[pos:end] += 1
    return cov.astype(np.float32)


def read_fastq_sequences(path: Path) -> list[str]:
    seqs: list[str] = []
    with open(path) as f:
        lines = f.readlines()
    for i in range(1, len(lines), 4):
        seqs.append(lines[i].strip())
    return seqs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="data/samples_yeast.parquet")
    parser.add_argument("--real-reads", default="data/reads.parquet")
    parser.add_argument("--synth-reads-dir", default="evals/output/reads")
    parser.add_argument("--real-output", default="evals/output/real_coverage")
    parser.add_argument("--synth-output", default="evals/output/synth_coverage")
    parser.add_argument("--fold", default=None, help="Filter samples by fold (e.g. val, train)")
    parser.add_argument("--skip-real", action="store_true", help="Skip real read coverage")
    parser.add_argument("--skip-synth", action="store_true", help="Skip synthetic read coverage")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if output exists")
    args = parser.parse_args()

    samples = pd.read_parquet(args.samples)
    if args.fold:
        samples = samples[samples["fold"] == args.fold]
    print(f"Windows: {len(samples):,}  (fold={args.fold or 'all'})")

    real_out = Path(args.real_output)
    synth_out = Path(args.synth_output)
    real_out.mkdir(parents=True, exist_ok=True)
    synth_out.mkdir(parents=True, exist_ok=True)

    # Determine whether samples have explicit start_seq / start_coverage columns
    # for computing the coverage-window offset within input_sequence.
    has_offset_cols = {"start_seq", "start_coverage", "end_coverage"}.issubset(samples.columns)
    if not has_offset_cols:
        print("Note: start_seq/start_coverage/end_coverage columns not found; "
              "real coverage will span the full input_sequence.")

    # ------------------------------------------------------------------
    # Load real reads (grouped by sample_idx)
    # ------------------------------------------------------------------
    real_reads_by_sample: dict[int, list[str]] = {}
    if not args.skip_real:
        print("Loading real reads…")
        reads_df = pd.read_parquet(args.real_reads, columns=["sample_idx", "read_seq"])
        sample_set = set(samples.index)
        reads_df = reads_df[reads_df["sample_idx"].isin(sample_set)]
        real_reads_by_sample = {
            idx: grp.tolist()
            for idx, grp in reads_df.groupby("sample_idx")["read_seq"]
        }
        print(f"  {len(reads_df):,} reads across {len(real_reads_by_sample):,} samples")

    synth_reads_dir = Path(args.synth_reads_dir)

    # ------------------------------------------------------------------
    # Process each window
    # ------------------------------------------------------------------
    real_match_total = real_total = 0
    synth_match_total = synth_total = 0

    for sample_idx, row in tqdm(samples.iterrows(), total=len(samples), desc="windows"):
        ref = row["input_sequence"]

        # Coverage window within ref
        if has_offset_cols:
            cov_offset = int(row["start_coverage"]) - int(row["start_seq"])
            cov_len = int(row["end_coverage"]) - int(row["start_coverage"])
            # Clamp to valid range
            cov_offset = max(0, min(cov_offset, len(ref)))
            cov_len = max(0, min(cov_len, len(ref) - cov_offset))
        else:
            cov_offset = 0
            cov_len = len(ref)

        # Real coverage
        if not args.skip_real:
            out_path = real_out / f"{sample_idx}.npy"
            if args.overwrite or not out_path.exists():
                reads = real_reads_by_sample.get(sample_idx, [])
                full_cov = compute_coverage(reads, ref)
                window_cov = full_cov[cov_offset : cov_offset + cov_len]
                np.save(str(out_path), window_cov)
                real_match_total += int(window_cov.sum())
                real_total += len(reads)

        # Synthetic coverage
        if not args.skip_synth:
            out_path = synth_out / f"{sample_idx}.npy"
            if args.overwrite or not out_path.exists():
                fastq_path = synth_reads_dir / f"{sample_idx}.fastq"
                if fastq_path.exists():
                    reads = read_fastq_sequences(fastq_path)
                    cov = compute_coverage(reads, ref)
                    np.save(str(out_path), cov)
                    synth_match_total += int(cov.sum())
                    synth_total += len(reads)

    if not args.skip_real and real_total > 0:
        # Each matched read contributes read_len counts; divide by ~avg_read_len to get
        # approximate number of matched reads (rough estimate).
        print(f"\nReal:  {real_total:,} reads processed")
        print(f"  total coverage counts written: {real_match_total:,}")

    if not args.skip_synth and synth_total > 0:
        print(f"\nSynth: {synth_total:,} reads processed")
        print(f"  total coverage counts written: {synth_match_total:,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
