#!/usr/bin/env python3
"""Extract RNA-seq reads from the aligned BAM for each yeast window in samples.pkl.

Runs on Modal. Reads from the coaster-aligned volume, writes to coaster-data volume.

Strand filtering: the SRR21628668 library is reverse-stranded (dUTP/RF).
  - R2 reads are in the sense direction of the transcript (same as RNA).
  - R1 reads are antisense → discarded.
  - No reverse-complement needed: pysam query_sequence for R2 is already 5'→3' RNA.

Chromosome mapping: samples.pkl uses NCBI RefSeq names (NC_001133.9, etc.);
  the BAM uses Ensembl R64-1-1 names (I, II, ..., XVI).

Output (coaster-data volume):
  /data/coaster/samples_yeast.parquet  — yeast windows with fold labels
  /data/coaster/reads.parquet          — (sample_idx, read_seq) pairs, T→U applied

Usage:
    modal run scripts/extract_reads_modal.py
    modal run scripts/extract_reads_modal.py --max-reads 500
"""

import pickle
import random
from pathlib import Path

import modal

app = modal.App("coaster-extract-reads")

aligned_vol = modal.Volume.from_name("coaster-aligned")
data_vol = modal.Volume.from_name("coaster-data")

ALIGNED_DIR = Path("/data/aligned")
DATA_DIR = Path("/data/coaster")   # coaster-data volume mount point
OUT_DIR = DATA_DIR

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=2.0", "pysam==0.22.1", "pandas", "pyarrow")
)

# NC accession → Ensembl R64-1-1 chromosome name (as used in the BAM)
NC_TO_ENSEMBL: dict[str, str] = {
    "NC_001133.9": "I",    "NC_001134.8": "II",   "NC_001135.5": "III",
    "NC_001136.10": "IV",  "NC_001137.3": "V",    "NC_001138.5": "VI",
    "NC_001139.9": "VII",  "NC_001140.6": "VIII", "NC_001141.2": "IX",
    "NC_001142.9": "X",    "NC_001143.9": "XI",   "NC_001144.5": "XII",
    "NC_001145.3": "XIII", "NC_001146.8": "XIV",  "NC_001147.6": "XV",
    "NC_001148.4": "XVI",
}


@app.function(
    image=image,
    volumes={
        str(ALIGNED_DIR): aligned_vol,
        str(DATA_DIR): data_vol,
    },
    cpu=4,
    memory=8192,
    timeout=7200,
)
def extract_reads(max_reads_per_window: int = 1000) -> None:
    import pysam
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    # samples.pkl was uploaded to the coaster-data volume via:
    #   modal volume put coaster-data data/samples.pkl /samples.pkl
    with open(DATA_DIR / "samples.pkl", "rb") as f:
        df = pickle.load(f)

    yeast = (
        df[df["chr"].str.contains("NC_001", na=False)]
        .reset_index(drop=True)
        .copy()
    )
    fold_counts = yeast["fold"].value_counts().to_dict()
    print(f"Yeast windows: {len(yeast)}  folds: {fold_counts}")

    bam_path = str(ALIGNED_DIR / "SRR21628668_Aligned.sortedByCoord.out.bam")
    rng = random.Random(42)
    records: list[dict] = []
    total_seen = 0

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for sample_idx, row in yeast.iterrows():
            ensembl_chr = NC_TO_ENSEMBL.get(row["chr"])
            if ensembl_chr is None:
                continue

            # samples.pkl is 1-based inclusive; pysam uses 0-based half-open
            start_0 = int(row["start_coverage"]) - 1
            end_0 = int(row["end_coverage"])

            # Reservoir sampling: uniform random sample of up to k reads
            # without loading the entire region into memory first.
            k = max_reads_per_window
            reservoir: list[str] = []
            n_seen = 0

            for read in bam.fetch(ensembl_chr, start_0, end_0):
                # Sense-strand filter for reverse-stranded (dUTP) library:
                # R2 query_sequence is always in the 5'→3' RNA direction.
                if (
                    read.is_unmapped
                    or read.is_secondary
                    or read.is_supplementary
                    or not read.is_read2
                    or read.mapping_quality < 10
                    or read.query_sequence is None
                ):
                    continue

                n_seen += 1
                if n_seen <= k:
                    reservoir.append(read.query_sequence)
                else:
                    j = rng.randint(0, n_seen - 1)
                    if j < k:
                        reservoir[j] = read.query_sequence

            total_seen += n_seen
            for seq in reservoir:
                records.append({
                    "sample_idx": int(sample_idx),
                    "read_seq": seq.replace("T", "U"),
                })

            if sample_idx % 500 == 0:
                print(
                    f"  {sample_idx}/{len(yeast)} windows | "
                    f"{n_seen} reads in window | "
                    f"{len(records):,} records so far"
                )

    print(f"\nTotal R2 reads seen: {total_seen:,}")
    print(f"Total records saved: {len(records):,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save reads
    pq.write_table(pa.Table.from_pylist(records), str(OUT_DIR / "reads.parquet"))

    # Save yeast sample metadata (DNA windows + fold labels) for the dataset
    keep_cols = [
        "chr", "strand", "start_seq", "end_seq",
        "start_coverage", "end_coverage", "input_sequence", "fold",
    ]
    yeast[keep_cols].to_parquet(str(OUT_DIR / "samples_yeast.parquet"), index=True)

    data_vol.commit()
    print("Saved reads.parquet and samples_yeast.parquet → volume coaster-data")


@app.local_entrypoint()
def main(max_reads: int = 1000) -> None:
    extract_reads.remote(max_reads_per_window=max_reads)
