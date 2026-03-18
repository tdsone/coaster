#!/usr/bin/env python3
"""Generate synthetic RNA-seq reads from a trained CoasterModel checkpoint.

For each genomic window in samples_yeast.parquet, generates `n_reads` reads
using temperature sampling (T=1.0 by default) and writes them as FASTQ files,
one per window, named by sample_idx.

Usage:
    uv run python evals/generate.py --checkpoint checkpoints/epoch_010.pt \\
        --samples data/samples_yeast.parquet \\
        --output-dir evals/output/reads \\
        --n-reads 100 \\
        --device cuda:0

Output: evals/output/reads/{sample_idx}.fastq
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from coaster.model import CoasterModel, load_config
from coaster.tokenizer import DNATokenizer, RNATokenizer

_DNA_TOK = DNATokenizer()
_RNA_TOK = RNATokenizer()

# Same trim/pad logic as RealRNADataset
def _trim_or_pad_dna(seq: str, target_len: int) -> str:
    n = len(seq)
    if n >= target_len:
        start = (n - target_len) // 2
        return seq[start : start + target_len]
    return seq + "N" * (target_len - n)


def generate_reads(
    model: CoasterModel,
    dna_seq: str,
    dna_len: int,
    n_reads: int,
    temperature: float,
    device: torch.device,
    batch_size: int = 64,
) -> list[str]:
    """Generate `n_reads` RNA reads for a single DNA window.

    Batches copies of the DNA sequence to amortise encoder cost.
    Returns RNA strings (with U's, no BOS/EOS).
    """
    dna = _trim_or_pad_dna(dna_seq, dna_len)
    dna_ids = torch.tensor(_DNA_TOK.encode(dna), dtype=torch.long, device=device)

    reads: list[str] = []
    remaining = n_reads
    while remaining > 0:
        b = min(batch_size, remaining)
        batch_dna = dna_ids.unsqueeze(0).expand(b, -1)  # (b, dna_len)
        batch_reads = model.generate(
            batch_dna,
            _RNA_TOK,
            max_len=_RNA_TOK.VOCAB_SIZE * 25,  # ~200 nt hard cap
            temperature=temperature,
            greedy=False,
        )
        reads.extend(batch_reads)
        remaining -= b
    return reads


def write_fastq(reads: list[str], path: Path, sample_idx: int) -> None:
    """Write reads as a FASTQ file with dummy quality scores (all 'I', PHRED+33=40)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, seq in enumerate(reads):
            # Convert U→T for alignment
            dna_seq = seq.replace("U", "T")
            f.write(f"@{sample_idx}_{i}\n")
            f.write(dna_seq + "\n")
            f.write("+\n")
            f.write("I" * len(dna_seq) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--samples", default="data/samples_yeast.parquet")
    parser.add_argument("--output-dir", default="evals/output/reads")
    parser.add_argument("--n-reads", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Reads to generate per encoder forward pass")
    parser.add_argument("--fold", default=None,
                        help="Only generate for windows in this fold (train/val/test)")
    args = parser.parse_args()

    device = torch.device(args.device)
    enc_cfg, dec_cfg, _ = load_config(args.config)

    model = CoasterModel(enc_cfg, dec_cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    samples = pd.read_parquet(args.samples)
    if args.fold is not None:
        samples = samples[samples["fold"] == args.fold]
    print(f"Windows to process: {len(samples):,}  (fold={args.fold or 'all'})")

    output_dir = Path(args.output_dir)
    n_done = 0
    for sample_idx, row in samples.iterrows():
        out_path = output_dir / f"{sample_idx}.fastq"
        if out_path.exists():
            n_done += 1
            continue

        reads = generate_reads(
            model,
            dna_seq=row["input_sequence"],
            dna_len=enc_cfg.dna_len,
            n_reads=args.n_reads,
            temperature=args.temperature,
            device=device,
            batch_size=args.batch_size,
        )
        write_fastq(reads, out_path, sample_idx)
        n_done += 1

        if n_done % 100 == 0 or n_done == len(samples):
            print(f"  {n_done}/{len(samples)} windows")

    print(f"Done. FASTQs written to {output_dir}/")


if __name__ == "__main__":
    main()
