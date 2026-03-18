#!/usr/bin/env python3
"""Generate synthetic RNA-seq reads from a trained CoasterModel checkpoint.

For each genomic window in samples_yeast.parquet, generates `n_reads` reads
using temperature sampling (T=1.0 by default) and writes them as FASTQ files,
one per window, named by sample_idx.

Two-phase approach for speed:
  1. Encode all windows upfront in batches (encoder runs once per window).
  2. Decode all reads in large cross-window batches (maximises GPU utilisation).

Usage:
    uv run python evals/generate.py --checkpoint checkpoints/last.pt \\
        --samples data/samples_yeast.parquet \\
        --output-dir evals/output/reads \\
        --n-reads 500 \\
        --device cuda:0

Output: evals/output/reads/{sample_idx}.fastq
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from coaster.model import CoasterModel, load_config
from coaster.tokenizer import DNATokenizer, RNATokenizer

_DNA_TOK = DNATokenizer()
_RNA_TOK = RNATokenizer()


def _trim_or_pad_dna(seq: str, target_len: int) -> str:
    n = len(seq)
    if n >= target_len:
        start = (n - target_len) // 2
        return seq[start : start + target_len]
    return seq + "N" * (target_len - n)


def write_fastq(reads: list[str], path: Path, sample_idx: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, seq in enumerate(reads):
            dna_seq = seq.replace("U", "T")
            f.write(f"@{sample_idx}_{i}\n{dna_seq}\n+\n{'I' * len(dna_seq)}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--samples", default="data/samples_yeast.parquet")
    parser.add_argument("--output-dir", default="evals/output/reads")
    parser.add_argument("--n-reads", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch", type=int, default=64,
                        help="Windows per encoder batch")
    parser.add_argument("--decode-batch", type=int, default=512,
                        help="Reads per decoder batch (across windows)")
    parser.add_argument("--fold", default=None)
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

    output_dir = Path(args.output_dir)
    # Skip windows already done
    todo = [(idx, row) for idx, row in samples.iterrows()
            if not (output_dir / f"{idx}.fastq").exists()]
    print(f"Windows to process: {len(todo):,}  (fold={args.fold or 'all'})")
    if not todo:
        print("All done.")
        return

    # -----------------------------------------------------------------------
    # Phase 1: encode all windows
    # -----------------------------------------------------------------------
    print("Phase 1: encoding windows…")
    memories: dict[int, torch.Tensor] = {}  # sample_idx → (S, d)

    with torch.no_grad():
        for chunk_start in range(0, len(todo), args.encode_batch):
            chunk = todo[chunk_start : chunk_start + args.encode_batch]
            dna_seqs = [_trim_or_pad_dna(row["input_sequence"], enc_cfg.dna_len)
                        for _, row in chunk]
            dna_ids = torch.tensor(
                [_DNA_TOK.encode(s) for s in dna_seqs],
                dtype=torch.long, device=device,
            )
            mem = model.encode(dna_ids)  # (chunk, S, d)
            for j, (idx, _) in enumerate(chunk):
                memories[idx] = mem[j]

            if (chunk_start // args.encode_batch + 1) % 10 == 0 or chunk_start + args.encode_batch >= len(todo):
                print(f"  encoded {min(chunk_start + args.encode_batch, len(todo))}/{len(todo)} windows")

    # -----------------------------------------------------------------------
    # Phase 2: decode all reads in large cross-window batches
    # -----------------------------------------------------------------------
    print("Phase 2: decoding reads…")

    # Flat list of (sample_idx, memory_tensor) for every read to generate
    all_jobs: list[tuple[int, torch.Tensor]] = [
        (idx, memories[idx])
        for idx, _ in todo
        for _ in range(args.n_reads)
    ]
    total = len(all_jobs)
    results: dict[int, list[str]] = {idx: [] for idx, _ in todo}

    with torch.no_grad():
        for batch_start in range(0, total, args.decode_batch):
            batch = all_jobs[batch_start : batch_start + args.decode_batch]
            memory_batch = torch.stack([m for _, m in batch])  # (B, S, d)
            reads = model.generate(
                dna_ids=None,
                rna_tokenizer=_RNA_TOK,
                temperature=args.temperature,
                greedy=False,
                memory=memory_batch,
            )
            for (idx, _), read in zip(batch, reads):
                results[idx].append(read)

            done = batch_start + len(batch)
            if done % (args.decode_batch * 10) == 0 or done == total:
                print(f"  decoded {done:,}/{total:,} reads")

    # -----------------------------------------------------------------------
    # Write FASTQs
    # -----------------------------------------------------------------------
    for idx, reads in results.items():
        write_fastq(reads, output_dir / f"{idx}.fastq", idx)

    print(f"Done. FASTQs written to {output_dir}/")


if __name__ == "__main__":
    main()
