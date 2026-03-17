#!/usr/bin/env python3
"""Download SRA run SRR21628668 (BY UBR1 -469A>T, S. cerevisiae RNA-Seq).

Requires the NCBI SRA Toolkit to be installed and on PATH.
Install via:
    brew install sratoolkit          # macOS
    conda install -c bioconda sra-tools  # conda

Usage:
    python scripts/download_sra.py [--outdir data/raw] [--threads 4]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SRR_ID = "SRR21628668"


def check_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        print(
            f"Error: '{name}' not found on PATH.\n"
            "Install the SRA Toolkit first:\n"
            "  brew install sratoolkit\n"
            "  conda install -c bioconda sra-tools",
            file=sys.stderr,
        )
        sys.exit(1)
    return path


def prefetch(srr_id: str, outdir: Path) -> Path:
    """Download the .sra blob with prefetch (resumable)."""
    check_tool("prefetch")
    sra_file = outdir / srr_id / f"{srr_id}.sra"
    if sra_file.exists():
        print(f"  [skip] {sra_file} already exists")
        return sra_file

    cmd = ["prefetch", srr_id, "--output-directory", str(outdir)]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return sra_file


def fasterq_dump(srr_id: str, outdir: Path, threads: int) -> list[Path]:
    """Convert .sra → paired FASTQ files."""
    check_tool("fasterq-dump")
    expected = [outdir / f"{srr_id}_1.fastq", outdir / f"{srr_id}_2.fastq"]
    gz_expected = [p.with_suffix(".fastq.gz") for p in expected]

    if all(p.exists() for p in gz_expected):
        print(f"  [skip] compressed FASTQs already exist")
        return gz_expected
    if all(p.exists() for p in expected):
        print(f"  [skip] FASTQs already exist, skipping extraction")
        return expected

    cmd = [
        "fasterq-dump",
        srr_id,
        "--outdir",
        str(outdir),
        "--split-files",
        "--threads",
        str(threads),
        "--progress",
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return expected


def gzip_fastqs(paths: list[Path], threads: int) -> list[Path]:
    """Compress FASTQs with pigz (parallel gzip) or gzip."""
    compressed = []
    pigz = shutil.which("pigz")

    for p in paths:
        gz = p.with_suffix(".fastq.gz")
        if gz.exists():
            compressed.append(gz)
            continue
        if not p.exists():
            compressed.append(p)
            continue

        if pigz:
            cmd = ["pigz", "-p", str(threads), str(p)]
        else:
            cmd = ["gzip", str(p)]
        print(f"  Compressing: {p.name}")
        subprocess.run(cmd, check=True)
        compressed.append(gz)

    return compressed


def main():
    parser = argparse.ArgumentParser(description=f"Download {SRR_ID} from NCBI SRA")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw)",
    )
    parser.add_argument("--threads", type=int, default=4, help="Threads for fasterq-dump / pigz")
    parser.add_argument("--no-gzip", action="store_true", help="Skip gzip compression of FASTQs")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"=== Downloading {SRR_ID} ===")
    print(f"  Study: Variation in Ubiquitin System Genes (PRJNA882076)")
    print(f"  Sample: BY UBR1 -469A>T replicate 1, S. cerevisiae RNA-Seq")
    print(f"  ~152.6M paired-end spots, ~13.8 GB compressed\n")

    print("[1/3] Prefetching .sra file...")
    prefetch(SRR_ID, args.outdir)

    print("[2/3] Extracting paired-end FASTQs...")
    fastqs = fasterq_dump(SRR_ID, args.outdir, args.threads)

    if not args.no_gzip:
        print("[3/3] Compressing FASTQs...")
        fastqs = gzip_fastqs(fastqs, args.threads)
    else:
        print("[3/3] Skipping compression (--no-gzip)")

    print(f"\nDone! Files:")
    for f in fastqs:
        if f.exists():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f}  ({size_mb:.1f} MB)")
        else:
            print(f"  {f}  (expected)")


if __name__ == "__main__":
    main()
