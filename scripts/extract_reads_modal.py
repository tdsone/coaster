#!/usr/bin/env python3
"""Extract and merge paired RNA-seq reads for each yeast window.

Runs on Modal. Reads from the coaster-aligned volume, writes to coaster-data volume.

Library: SRR21628668 is reverse-stranded (dUTP/RF).
  - R2 = sense read (same direction as the RNA transcript).
  - R1 = antisense read; its gene-sense sequence covers the 3' end of the fragment.

Strand orientation:
  input_sequence in samples.pkl is always in the 5'→3' gene-sense direction:
    + strand genes: input_sequence is the + genomic strand.
    - strand genes: input_sequence is the reverse complement (already flipped).

  pysam stores reverse-strand-mapped reads as their reverse complement, so
  query_sequence is always in the + genomic strand direction regardless of the read.
  To convert to gene-sense:
    + strand genes: use query_sequence directly (+ genomic = gene-sense).
    - strand genes: reverse-complement query_sequence (+ genomic = gene-antisense).

  This applies to both R1 and R2.

Paired-end merging:
  After converting to gene-sense, R2 covers the 5' portion of the fragment and
  R1 covers the 3' portion (reading inward from the other end). When the insert
  is short enough for R2 and R1 to overlap, we merge them into the full insert:

    gene 5'→3':  [=======R2======>]
                          [<=======R1=======]
    merged:      [===========full insert===========]

  Overlap is detected by finding the longest suffix of R2 that matches a prefix
  of R1 (minimum overlap = MIN_OVERLAP bases to avoid false positives).
  If no overlap is found, R2 alone is kept.

Reservoir sampling is applied at the pair level (one slot per R2, whether merged
or not), so the sample size semantics are unchanged from v1.

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
from collections import defaultdict
from pathlib import Path

import modal
from coaster.preprocessing import MIN_OVERLAP, merge_pair, revcomp, to_gene_sense

app = modal.App("coaster-extract-reads")

aligned_vol = modal.Volume.from_name("coaster-aligned")
data_vol = modal.Volume.from_name("coaster-data")

ALIGNED_DIR = Path("/data/aligned")
DATA_DIR = Path("/data/coaster")
OUT_DIR = DATA_DIR

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=2.0", "pysam==0.22.1", "pandas", "pyarrow")
    .add_local_python_source("coaster")
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
    timeout=43200,
)
def extract_reads(max_reads_per_window: int = 1000) -> None:
    import pysam
    import pyarrow as pa
    import pyarrow.parquet as pq

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

    total_inserts = 0
    total_merged = 0
    total_r2_only = 0

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for sample_idx, row in yeast.iterrows():
            ensembl_chr = NC_TO_ENSEMBL.get(row["chr"])
            if ensembl_chr is None:
                continue

            gene_strand = row["strand"]
            # samples.pkl is 1-based inclusive; pysam uses 0-based half-open
            start_0 = int(row["start_coverage"]) - 1
            end_0 = int(row["end_coverage"])

            # Collect all usable reads in the window, grouped by query_name.
            # We need both R1 and R2 to attempt merging.
            by_name: dict[str, dict] = defaultdict(dict)
            for read in bam.fetch(ensembl_chr, start_0, end_0):
                if (
                    read.is_unmapped
                    or read.is_secondary
                    or read.is_supplementary
                    or read.mapping_quality < 10
                    or read.query_sequence is None
                ):
                    continue
                key = "R2" if read.is_read2 else "R1"
                # Keep first occurrence only (duplicates are rare in RNA-seq)
                if key not in by_name[read.query_name]:
                    by_name[read.query_name][key] = read

            # Build insert sequences in gene-sense direction.
            # One insert per R2 read (merged with its R1 mate if they overlap).
            inserts: list[str] = []
            n_merged = n_r2_only = 0

            for reads in by_name.values():
                r2 = reads.get("R2")
                if r2 is None:
                    continue  # R1-only: skip (antisense, not useful alone)

                r2_sense = to_gene_sense(r2.query_sequence, gene_strand)
                r1 = reads.get("R1")

                if r1 is not None:
                    r1_sense = to_gene_sense(r1.query_sequence, gene_strand)
                    insert = merge_pair(r2_sense, r1_sense)
                    if len(insert) > len(r2_sense):
                        n_merged += 1
                    else:
                        n_r2_only += 1
                else:
                    insert = r2_sense
                    n_r2_only += 1

                inserts.append(insert)

            total_inserts += len(inserts)
            total_merged += n_merged
            total_r2_only += n_r2_only

            # Reservoir-sample up to k inserts (Algorithm R).
            k = max_reads_per_window
            if len(inserts) <= k:
                sample = inserts
            else:
                sample = inserts[:k]
                for j in range(k, len(inserts)):
                    idx = rng.randint(0, j)
                    if idx < k:
                        sample[idx] = inserts[j]

            for seq in sample:
                records.append({
                    "sample_idx": int(sample_idx),
                    "read_seq": seq.replace("T", "U"),
                })

            if sample_idx % 5 == 0:
                print(
                    f"  {sample_idx}/{len(yeast)} | "
                    f"inserts={len(inserts)} merged={n_merged} r2_only={n_r2_only}"
                )

    pct_merged = 100 * total_merged / max(total_inserts, 1)
    print(f"\nTotal inserts: {total_inserts:,}")
    print(f"  merged pairs:  {total_merged:,} ({pct_merged:.1f}%)")
    print(f"  R2 only:       {total_r2_only:,} ({100 - pct_merged:.1f}%)")
    print(f"Total records saved: {len(records):,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pq.write_table(pa.Table.from_pylist(records), str(OUT_DIR / "reads.parquet"))

    keep_cols = [
        "chr", "strand", "start_seq", "end_seq",
        "start_coverage", "end_coverage", "input_sequence", "fold",
    ]
    yeast[keep_cols].to_parquet(str(OUT_DIR / "samples_yeast.parquet"), index=True)

    data_vol.commit()
    print("Saved reads.parquet and samples_yeast.parquet → coaster-data volume")


@app.local_entrypoint()
def main(max_reads: int = 1000) -> None:
    extract_reads.remote(max_reads_per_window=max_reads)
