#!/usr/bin/env python3
"""Modal pipeline: synthetic FASTQs → BAM → bigwigs, plus real coverage extraction.

Steps:
  build_index   build a STAR index from all window sequences (one "chromosome" per window)
  align         align synthetic FASTQs to the window genome → sorted BAM
  bigwigs       BAM → per-strand bigwig (deeptools bamCoverage)
  real_coverage extract per-window coverage vectors from the real sacCer3 BAM

Volumes:
  coaster-data     → /data/coaster    samples_yeast.parquet, reads.parquet
  coaster-aligned  → /data/aligned    real BAM (SRR21628668_Aligned.sortedByCoord.out.bam)
  coaster-evals    → /data/evals      FASTQs (input), index, BAM, bigwigs, coverage (output)

Usage:
  # Upload FASTQs first:
  modal volume put coaster-evals evals/output/reads /reads

  modal run evals/eval_modal.py --step build_index
  modal run evals/eval_modal.py --step align
  modal run evals/eval_modal.py --step bigwigs
  modal run evals/eval_modal.py --step real_coverage
  modal run evals/eval_modal.py --step all
"""

import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("coaster-eval")

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

data_vol = modal.Volume.from_name("coaster-data")
aligned_vol = modal.Volume.from_name("coaster-aligned")
evals_vol = modal.Volume.from_name("coaster-evals", create_if_missing=True)

DATA_DIR = Path("/data/coaster")
ALIGNED_DIR = Path("/data/aligned")
EVALS_DIR = Path("/data/evals")

READS_DIR = EVALS_DIR / "reads"        # input FASTQs (one per window)
INDEX_DIR = EVALS_DIR / "star_index"   # STAR index for window genome
WINDOW_FA = EVALS_DIR / "windows.fa"   # concatenated window sequences
BAM_DIR = EVALS_DIR / "bam"
BIGWIG_DIR = EVALS_DIR / "bigwigs"
REAL_COV_DIR = EVALS_DIR / "real_coverage"
SYNTH_COV_DIR = EVALS_DIR / "synth_coverage"

REAL_BAM = ALIGNED_DIR / "SRR21628668_Aligned.sortedByCoord.out.bam"

# ---------------------------------------------------------------------------
# Container images
# ---------------------------------------------------------------------------

bio_image = (
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install(
        "star=2.7.11a",
        "samtools=1.19",
        "deeptools=3.5.4",
        channels=["bioconda", "conda-forge"],
    )
    .pip_install("pandas", "pyarrow", "pysam==0.22.1", "numpy")
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(cmd: list, **kwargs) -> None:
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


# ---------------------------------------------------------------------------
# Step 1 — build STAR index from window sequences
# ---------------------------------------------------------------------------


@app.function(
    image=bio_image,
    volumes={
        str(DATA_DIR): data_vol,
        str(EVALS_DIR): evals_vol,
    },
    cpu=8,
    memory=16384,
    timeout=3600,
)
def build_index() -> None:
    """Write windows.fa (one seq per window) and build a STAR index."""
    import pandas as pd

    if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
        print("[skip] STAR index already exists")
        return

    samples = pd.read_parquet(DATA_DIR / "samples_yeast.parquet")
    print(f"Writing {len(samples)} window sequences to {WINDOW_FA}")

    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    with open(WINDOW_FA, "w") as fa:
        for sample_idx, row in samples.iterrows():
            fa.write(f">{sample_idx}\n{row['input_sequence']}\n")

    # genomeSAindexNbases: min(14, floor(log2(total_len) / 2 - 1))
    # ~8837 windows × 5000 bp ≈ 44M bp → log2(44M)/2 - 1 ≈ 11.7 → use 11
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    run([
        "STAR",
        "--runMode", "genomeGenerate",
        "--runThreadN", "8",
        "--genomeDir", str(INDEX_DIR),
        "--genomeFastaFiles", str(WINDOW_FA),
        "--genomeSAindexNbases", "11",
    ])

    evals_vol.commit()
    print("STAR index built.")


# ---------------------------------------------------------------------------
# Step 2 — align synthetic FASTQs → sorted BAM
# ---------------------------------------------------------------------------


@app.function(
    image=bio_image,
    volumes={str(EVALS_DIR): evals_vol},
    cpu=16,
    memory=32768,
    timeout=7200,
)
def align() -> None:
    """Merge per-window FASTQs and align to window genome with STAR (ENCODE params)."""
    import shutil

    out_bam = BAM_DIR / "synthetic_Aligned.sortedByCoord.out.bam"
    if out_bam.exists() and out_bam.stat().st_size > 0:
        print("[skip] BAM already exists")
        return

    if not INDEX_DIR.exists():
        sys.exit("Missing STAR index — run --step build_index first")

    fastqs = sorted(READS_DIR.glob("*.fastq"))
    if not fastqs:
        sys.exit(f"No FASTQs found in {READS_DIR} — upload them first")
    print(f"Found {len(fastqs)} FASTQs")

    # Merge all per-window FASTQs into one for STAR
    merged_fq = EVALS_DIR / "synthetic_merged.fastq"
    print(f"Merging FASTQs → {merged_fq}")
    with open(merged_fq, "wb") as out:
        for fq in fastqs:
            with open(fq, "rb") as inp:
                out.write(inp.read())

    local_out = Path("/tmp/star_out")
    local_out.mkdir(exist_ok=True)

    # ENCODE splicing params, matching the original alignment
    run([
        "STAR",
        "--runMode", "alignReads",
        "--runThreadN", "16",
        "--genomeDir", str(INDEX_DIR),
        "--readFilesIn", str(merged_fq),
        "--outSAMtype", "BAM", "SortedByCoordinate",
        "--outSAMattributes", "NH", "HI", "AS", "NM", "MD",
        "--outFileNamePrefix", str(local_out / "synthetic_"),
        "--outTmpDir", "/tmp/STARtmp",
        "--outBAMsortingThreadN", "8",
        "--limitBAMsortRAM", "17179869184",
        "--outFilterType", "BySJout",
        "--outFilterMultimapNmax", "20",
        "--alignSJoverhangMin", "8",
        "--alignSJDBoverhangMin", "1",
        "--outFilterMismatchNmax", "999",
        "--outFilterMismatchNoverReadLmax", "0.04",
        "--alignIntronMin", "20",
        "--alignIntronMax", "1000000",
        "--alignMatesGapMax", "1000000",
    ])

    local_bam = local_out / "synthetic_Aligned.sortedByCoord.out.bam"
    run(["samtools", "index", str(local_bam)])

    BAM_DIR.mkdir(parents=True, exist_ok=True)
    for f in local_out.iterdir():
        if f.is_file():
            shutil.copy2(f, BAM_DIR / f.name)

    evals_vol.commit()
    print(f"BAM written to {BAM_DIR}/")


# ---------------------------------------------------------------------------
# Step 3 — BAM → strand-specific bigwigs
# ---------------------------------------------------------------------------


@app.function(
    image=bio_image,
    volumes={str(EVALS_DIR): evals_vol},
    cpu=8,
    memory=16384,
    timeout=3600,
)
def make_bigwigs() -> None:
    """Generate forward and reverse strand bigwigs from the synthetic BAM."""
    bam = BAM_DIR / "synthetic_Aligned.sortedByCoord.out.bam"
    if not bam.exists():
        sys.exit("Missing BAM — run --step align first")

    BIGWIG_DIR.mkdir(parents=True, exist_ok=True)

    for strand, flag in [("forward", "forward"), ("reverse", "reverse")]:
        out_bw = BIGWIG_DIR / f"synthetic_{strand}.bw"
        if out_bw.exists():
            print(f"[skip] {out_bw.name} already exists")
            continue
        run([
            "bamCoverage",
            "--bam", str(bam),
            "--outFileName", str(out_bw),
            "--filterRNAstrand", flag,
            "--normalizeUsing", "CPM",
            "--binSize", "1",
            "--numberOfProcessors", "8",
        ])

    evals_vol.commit()
    print(f"Bigwigs written to {BIGWIG_DIR}/")


# ---------------------------------------------------------------------------
# Step 4 — extract real per-window coverage from sacCer3 BAM
# ---------------------------------------------------------------------------


@app.function(
    image=bio_image,
    volumes={
        str(DATA_DIR): data_vol,
        str(ALIGNED_DIR): aligned_vol,
        str(EVALS_DIR): evals_vol,
    },
    cpu=4,
    memory=8192,
    timeout=7200,
)
def real_coverage() -> None:
    """Extract per-window coverage vectors from the real BAM → numpy arrays.

    Output: REAL_COV_DIR/{sample_idx}.npy  — shape (window_len,), sense-strand counts.
    """
    import numpy as np
    import pandas as pd
    import pysam

    REAL_COV_DIR.mkdir(parents=True, exist_ok=True)

    samples = pd.read_parquet(DATA_DIR / "samples_yeast.parquet")
    # Ensembl chromosome names used in the real BAM
    NC_TO_ENSEMBL = {
        "NC_001133.9": "I",    "NC_001134.8": "II",   "NC_001135.5": "III",
        "NC_001136.10": "IV",  "NC_001137.3": "V",    "NC_001138.5": "VI",
        "NC_001139.9": "VII",  "NC_001140.6": "VIII", "NC_001141.2": "IX",
        "NC_001142.9": "X",    "NC_001143.9": "XI",   "NC_001144.5": "XII",
        "NC_001145.3": "XIII", "NC_001146.8": "XIV",  "NC_001147.6": "XV",
        "NC_001148.4": "XVI",
    }

    with pysam.AlignmentFile(str(REAL_BAM), "rb") as bam:
        for sample_idx, row in samples.iterrows():
            out_path = REAL_COV_DIR / f"{sample_idx}.npy"
            if out_path.exists():
                continue

            ensembl_chr = NC_TO_ENSEMBL.get(row["chr"])
            if ensembl_chr is None:
                continue

            start_0 = int(row["start_coverage"]) - 1
            end_0 = int(row["end_coverage"])
            window_len = end_0 - start_0

            coverage = np.zeros(window_len, dtype=np.float32)
            for read in bam.fetch(ensembl_chr, start_0, end_0):
                # Same strand filter as extract_reads_modal.py:
                # dUTP/RF library — R2 is sense strand
                if (
                    read.is_unmapped
                    or read.is_secondary
                    or read.is_supplementary
                    or not read.is_read2
                    or read.mapping_quality < 10
                    or read.query_sequence is None
                ):
                    continue
                # Accumulate coverage at aligned positions
                for pos in read.get_reference_positions():
                    local = pos - start_0
                    if 0 <= local < window_len:
                        coverage[local] += 1

            np.save(str(out_path), coverage)

    evals_vol.commit()
    print(f"Real coverage arrays written to {REAL_COV_DIR}/")


# ---------------------------------------------------------------------------
# Step 5 — extract synthetic per-window coverage from the synthetic BAM
# ---------------------------------------------------------------------------


@app.function(
    image=bio_image,
    volumes={
        str(DATA_DIR): data_vol,
        str(EVALS_DIR): evals_vol,
    },
    cpu=4,
    memory=8192,
    timeout=3600,
)
def synth_coverage() -> None:
    """Extract per-window coverage from the synthetic BAM → numpy arrays.

    The synthetic BAM uses sample_idx as chromosome names (one per window).
    Output: SYNTH_COV_DIR/{sample_idx}.npy  — shape (window_len,), raw counts.
    """
    import numpy as np
    import pandas as pd
    import pysam

    bam_path = BAM_DIR / "synthetic_Aligned.sortedByCoord.out.bam"
    if not bam_path.exists():
        sys.exit("Missing synthetic BAM — run --step align first")

    SYNTH_COV_DIR.mkdir(parents=True, exist_ok=True)

    samples = pd.read_parquet(DATA_DIR / "samples_yeast.parquet")

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for sample_idx, row in samples.iterrows():
            out_path = SYNTH_COV_DIR / f"{sample_idx}.npy"
            if out_path.exists():
                continue

            seq_len = len(row["input_sequence"])
            coverage = np.zeros(seq_len, dtype=np.float32)

            try:
                for read in bam.fetch(str(sample_idx), 0, seq_len):
                    if read.is_unmapped or read.is_secondary or read.is_supplementary:
                        continue
                    for pos in read.get_reference_positions():
                        if 0 <= pos < seq_len:
                            coverage[pos] += 1
            except ValueError:
                pass  # chromosome not present (no reads aligned to this window)

            np.save(str(out_path), coverage)

    evals_vol.commit()
    print(f"Synthetic coverage arrays written to {SYNTH_COV_DIR}/")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(step: str = "all") -> None:
    steps = {"build_index", "align", "bigwigs", "real_coverage", "synth_coverage", "all"}
    if step not in steps:
        sys.exit(f"Unknown step '{step}'. Choose: {' | '.join(sorted(steps))}")

    if step in ("build_index", "all"):
        print("=== [1/5] Building STAR index ===")
        build_index.remote()
    if step in ("align", "all"):
        print("=== [2/5] Aligning synthetic reads ===")
        align.remote()
    if step in ("bigwigs", "all"):
        print("=== [3/5] Generating bigwigs ===")
        make_bigwigs.remote()
    if step in ("real_coverage", "all"):
        print("=== [4/5] Extracting real coverage ===")
        real_coverage.remote()
    if step in ("synth_coverage", "all"):
        print("=== [5/5] Extracting synthetic coverage ===")
        synth_coverage.remote()
