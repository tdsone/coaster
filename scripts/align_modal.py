#!/usr/bin/env python3
"""Align SRR21628668 RNA-seq reads to sacCer3 (Ensembl R64-1-1) using Modal.

Volumes:
  coaster-ref     → /data/ref      genome FASTA, GTF, STAR index
  coaster-reads   → /data/reads    SRA blob, raw FASTQs
  coaster-aligned → /data/aligned  sorted BAM, gene counts

Usage:
  modal run scripts/align_modal.py              # run all 3 steps
  modal run scripts/align_modal.py --step index # build STAR index only
  modal run scripts/align_modal.py --step reads # download SRA reads only
  modal run scripts/align_modal.py --step align # align only (needs index + reads)
"""

import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("coaster-align")

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

ref_vol = modal.Volume.from_name("coaster-ref", create_if_missing=True)
reads_vol = modal.Volume.from_name("coaster-reads", create_if_missing=True)
aligned_vol = modal.Volume.from_name("coaster-aligned", create_if_missing=True)

REF_DIR = Path("/data/ref")
READS_DIR = Path("/data/reads")
ALIGNED_DIR = Path("/data/aligned")

# ---------------------------------------------------------------------------
# Reference URLs — Ensembl release 111, sacCer3 / R64-1-1
# ---------------------------------------------------------------------------

ENSEMBL = 111
GENOME_URL = (
    f"https://ftp.ensembl.org/pub/release-{ENSEMBL}/fasta/"
    "saccharomyces_cerevisiae/dna/"
    f"Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.gz"
)
GTF_URL = (
    f"https://ftp.ensembl.org/pub/release-{ENSEMBL}/gtf/"
    "saccharomyces_cerevisiae/"
    f"Saccharomyces_cerevisiae.R64-1-1.{ENSEMBL}.gtf.gz"
)

SRR_ID = "SRR21628668"

# ---------------------------------------------------------------------------
# Container image — bioinformatics tools via micromamba / bioconda
# ---------------------------------------------------------------------------

bio_image = (
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install(
        "star=2.7.11a",
        "samtools=1.19",
        "sra-tools=3.1.1",
        "pigz",
        channels=["bioconda", "conda-forge"],
    )
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(cmd: list, **kwargs) -> None:
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


# ---------------------------------------------------------------------------
# Step 1 — download sacCer3 genome + GTF and build STAR index
# ---------------------------------------------------------------------------


@app.function(
    image=bio_image,
    volumes={str(REF_DIR): ref_vol},
    cpu=8,
    memory=16384,
    timeout=3600,
)
def build_index():
    """Download sacCer3 genome + GTF from Ensembl and build a STAR index."""
    index_dir = REF_DIR / "star_index"
    genome_fa = REF_DIR / "genome.fa"
    gtf = REF_DIR / "annotation.gtf"

    if index_dir.exists() and any(index_dir.iterdir()):
        print("[skip] STAR index already exists")
        return

    REF_DIR.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    if not genome_fa.exists():
        print("Downloading genome FASTA…")
        gz = Path(str(genome_fa) + ".gz")
        run(["curl", "-fL", "--progress-bar", "-o", str(gz), GENOME_URL])
        run(["pigz", "-d", str(gz)])

    if not gtf.exists():
        print("Downloading GTF annotation…")
        gz = Path(str(gtf) + ".gz")
        run(["curl", "-fL", "--progress-bar", "-o", str(gz), GTF_URL])
        run(["pigz", "-d", str(gz)])

    print("Building STAR index…  (--genomeSAindexNbases 10 for small genome)")
    run([
        "STAR",
        "--runMode", "genomeGenerate",
        "--runThreadN", "8",
        "--genomeDir", str(index_dir),
        "--genomeFastaFiles", str(genome_fa),
        "--sjdbGTFfile", str(gtf),
        "--genomeSAindexNbases", "10",   # required for small genomes like yeast
    ])

    ref_vol.commit()
    print("Done — STAR index ready.")


# SRA open data on AWS S3 — accessible via HTTPS from Modal containers
SRA_S3_URL = f"https://sra-pub-run-odp.s3.amazonaws.com/sra/{SRR_ID}/{SRR_ID}"

# ---------------------------------------------------------------------------
# Step 2 — download SRR21628668 from NCBI SRA and extract paired FASTQs
# ---------------------------------------------------------------------------


@app.function(
    image=bio_image,
    volumes={str(READS_DIR): reads_vol},
    cpu=8,
    memory=16384,
    timeout=18000,  # 5 h — download + fasterq-dump can be slow
)
def download_reads():
    """Download SRR21628668 from NCBI's S3 open-data bucket and extract paired FASTQs."""
    fq1 = READS_DIR / f"{SRR_ID}_1.fastq.gz"
    fq2 = READS_DIR / f"{SRR_ID}_2.fastq.gz"

    if fq1.exists() and fq2.exists():
        print("[skip] FASTQs already exist on volume")
        return

    READS_DIR.mkdir(parents=True, exist_ok=True)
    sra_dir = READS_DIR / SRR_ID
    sra_dir.mkdir(exist_ok=True)
    sra_file = sra_dir / f"{SRR_ID}.sra"

    # Download .sra directly from NCBI's S3 open-data bucket via HTTPS
    if not sra_file.exists():
        print(f"Downloading {SRR_ID}.sra from S3 (~14 GB)…")
        run(["curl", "-fL", "--progress-bar", "-o", str(sra_file), SRA_S3_URL])
        reads_vol.commit()

    raw1 = READS_DIR / f"{SRR_ID}_1.fastq"
    raw2 = READS_DIR / f"{SRR_ID}_2.fastq"

    if not (raw1.exists() or fq1.exists()):
        print("Extracting paired FASTQs with fasterq-dump…")
        tmp_dir = READS_DIR / "fasterq_tmp"
        tmp_dir.mkdir(exist_ok=True)
        run([
            "fasterq-dump", str(sra_file),
            "--outdir", str(READS_DIR),
            "--temp", str(tmp_dir),
            "--split-files",
            "--threads", "8",
            "--progress",
        ])
        # clean up temp files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        reads_vol.commit()

    # compress
    for raw in [raw1, raw2]:
        if raw.exists():
            print(f"Compressing {raw.name}…")
            run(["pigz", "-p", "8", str(raw)])

    reads_vol.commit()
    print("Done — FASTQs ready.")


# ---------------------------------------------------------------------------
# Step 3 — STAR alignment → sorted BAM + per-gene read counts
# ---------------------------------------------------------------------------


@app.function(
    image=bio_image,
    volumes={
        str(REF_DIR): ref_vol,
        str(READS_DIR): reads_vol,
        str(ALIGNED_DIR): aligned_vol,
    },
    cpu=16,
    memory=32768,
    timeout=7200,
)
def align():
    """Align paired FASTQs to sacCer3 with STAR; output sorted BAM + gene counts."""
    import shutil

    bam = ALIGNED_DIR / f"{SRR_ID}_Aligned.sortedByCoord.out.bam"
    if bam.exists() and bam.stat().st_size > 0:
        print("[skip] BAM already exists")
        return

    fq1 = READS_DIR / f"{SRR_ID}_1.fastq.gz"
    fq2 = READS_DIR / f"{SRR_ID}_2.fastq.gz"
    index_dir = REF_DIR / "star_index"

    for p in [fq1, fq2]:
        if not p.exists():
            sys.exit(f"Missing input: {p}  — run --step reads first")
    if not index_dir.exists():
        sys.exit("Missing STAR index — run --step index first")

    # Run STAR entirely on local ephemeral storage to avoid network FS issues
    # with BAM sorting (which does heavy random I/O). Copy to volume at the end.
    local_out = Path("/tmp/star_out")
    local_out.mkdir(exist_ok=True)

    print(f"Aligning {SRR_ID} to sacCer3…  (writing to local /tmp first)")
    run([
        "STAR",
        "--runMode", "alignReads",
        "--runThreadN", "16",
        "--genomeDir", str(index_dir),
        "--readFilesIn", str(fq1), str(fq2),
        "--readFilesCommand", "zcat",
        "--outSAMtype", "BAM", "SortedByCoordinate",
        "--outSAMattributes", "NH", "HI", "AS", "NM", "MD",
        "--outFileNamePrefix", str(local_out / f"{SRR_ID}_"),
        "--outTmpDir", "/tmp/STARtmp",
        "--quantMode", "GeneCounts",
        "--outBAMsortingThreadN", "8",
        "--limitBAMsortRAM", "17179869184",  # 16 GB
        # ENCODE recommended splicing parameters
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

    local_bam = local_out / f"{SRR_ID}_Aligned.sortedByCoord.out.bam"
    print("Indexing BAM…")
    run(["samtools", "index", str(local_bam)])

    # Copy all outputs to the persistent volume
    print("Copying outputs to volume…")
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    for f in local_out.iterdir():
        if f.is_file():
            shutil.copy2(f, ALIGNED_DIR / f.name)

    aligned_vol.commit()
    print(f"\nDone! Output in {ALIGNED_DIR}:")
    for f in sorted(ALIGNED_DIR.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(step: str = "all"):
    """
    --step index   build STAR index
    --step reads   download SRA + extract FASTQs
    --step align   run alignment (requires index + reads)
    --step all     run all three in order (default)
    """
    if step == "index":
        build_index.remote()
    elif step == "reads":
        download_reads.remote()
    elif step == "align":
        align.remote()
    elif step == "all":
        print("=== [1/3] Building STAR index ===")
        build_index.remote()
        print("=== [2/3] Downloading SRA reads ===")
        download_reads.remote()
        print("=== [3/3] Aligning reads ===")
        align.remote()
    else:
        sys.exit(f"Unknown step '{step}'. Choose: index | reads | align | all")
