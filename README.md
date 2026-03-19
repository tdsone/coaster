# coaster

A DNA-to-RNA encoder-decoder transformer that models yeast (*S. cerevisiae*) transcription. Given a ~5 kb genomic DNA window, the model learns to generate RNA-seq reads that would plausibly originate from that region.

## Table of contents
- [coaster](#coaster)
  - [Table of contents](#table-of-contents)
  - [Motivation](#motivation)
  - [Architecture](#architecture)
    - [Key design choices](#key-design-choices)
  - [Project status](#project-status)
    - [Done](#done)
    - [Next steps](#next-steps)
    - [Evaluation pipeline](#evaluation-pipeline)
  - [Repository layout](#repository-layout)
  - [Data](#data)
    - [Genomic windows (`samples.pkl` / `samples_yeast.parquet`)](#genomic-windows-samplespkl--samples_yeastparquet)
    - [Strand orientation and read extraction](#strand-orientation-and-read-extraction)
    - [Paired-end merging](#paired-end-merging)
  - [Setup](#setup)
  - [Training](#training)
  - [Hyperparameters (default, `configs/default.yaml`)](#hyperparameters-default-configsdefaultyaml)

## Motivation

RNA-seq read distributions encode everything that happens between DNA and mature RNA: transcription, splicing, degradation, and regulation. Rather than predicting a single summary statistic (e.g. expression level), coaster directly generates RNA-sequencing reads for a given DNA context window.

## Architecture

```
DNA string (~4992 bp)
  → char tokenizer (A/T/G/C/N, vocab=6)
  → Embedding(6, 256)
  → Conv1d(256, 256, kernel=8, stride=8)   # compress 4992 → 624 positions
  → SinusoidalPosEmb
  → TransformerEncoder × 4  (d=256, 4 heads, ffn=1024, pre-norm)
  → memory  [B, 624, 256]

RNA read (one per forward pass, ~151 nt)
  → RNATokenizer (PAD/BOS/EOS/A/U/G/C/N, vocab=8)
  → Embedding(8, 256) + LearnedPosEmb
  → TransformerDecoder × 4  (d=256, 4 heads, ffn=1024, pre-norm, cross-attn to memory)
  → RMSNorm → Linear(256, 8)
  → logits  [B, T, 8]
```

**~8M parameters.** Training objective: teacher-forced cross-entropy on RNA token sequences, one read per forward pass.

### Key design choices

| Choice | Rationale |
|---|---|
| Conv1d stride-8 downsampling | Reduces 4992 DNA positions to 624 before self-attention, keeping memory and compute manageable |
| Sinusoidal pos emb (encoder) | Fixed, no learned parameters needed for fixed-length DNA input |
| Learned pos emb (decoder) | Better at short sequences (~153 tokens); positions are absolute within a read |
| Pre-norm (`norm_first=True`) | More stable training at small batch sizes |
| One read per forward pass | Simplest formulation; the model is free to generate any plausible read from the DNA context |

## Project status

### Done
- **Alignment pipeline** — `scripts/align_modal.py`: Modal-based pipeline to download the sacCer3 reference genome (Ensembl R111), build a STAR index, download SRR21628668 paired-end RNA-seq (~14 GB from NCBI S3), and align to produce a sorted BAM + gene counts
- **Read extraction** — `scripts/extract_reads_modal.py`: extracts R2 reads (sense strand, dUTP library) from the BAM for each yeast genomic window using reservoir sampling (≤1000 reads/window), saves `reads.parquet` and `samples_yeast.parquet` to the `coaster-data` Modal volume
- **Model** — full encoder-decoder architecture (`coaster/model/`)
- **Paired-end merging** — `coaster/preprocessing.py`: `revcomp`, `to_gene_sense`, `merge_pair` helpers; `scripts/extract_reads_modal.py` now converts both R1 and R2 to gene-sense direction and merges overlapping pairs into full insert sequences before reservoir-sampling
- **Real data** — `RealRNADataset` pairs 8.65M extracted reads with their 5000 bp DNA windows across train/val/test folds
- **Trainer** — AdamW + cosine LR schedule with warmup; AMP on CUDA (autocast + grad scaler), float32 on MPS/CPU; checkpoint save/load
- **Evaluation pipeline** — `evals/generate.py` → `evals/eval_modal.py` (5-step Modal pipeline) → `evals/compare.py` / `evals/visualize.py`; local alternative via `evals/compute_coverage.py`
- **Tests** — 58 tests covering tokenizers, model shapes, forward/backward, data pipeline, training loop, and preprocessing helpers

### Next steps
- Unfreeze/replace encoder with NTv3 once real-data training is established

### Evaluation pipeline

**Stages** (each independently runnable):
1. `evals/generate.py` — local: model → per-window FASTQs (two-phase: encode all windows once, then decode in large cross-window batches)
2. `evals/eval_modal.py` — Modal, 5 steps:
   - `build_index`: concatenate window sequences as chromosomes → STAR index
   - `align`: synthetic FASTQs → sorted BAM (ENCODE STAR params, matching `scripts/align_modal.py`)
   - `bigwigs`: BAM → forward/reverse strand CPM bigwigs (deeptools bamCoverage, bin=1)
   - `real_coverage`: extract per-window sense-strand coverage from real sacCer3 BAM → `.npy` arrays
   - `synth_coverage`: extract per-window coverage from synthetic BAM → `.npy` arrays
3. `evals/compare.py` — local: synthetic bigwig vs real `.npy` coverage → per-window + aggregate Pearson/Spearman
4. `evals/visualize.py` — local: plot real vs synthetic coverage tracks per window

A lighter local alternative skips STAR alignment and instead uses exact substring search to compute coverage:
`evals/compute_coverage.py` — real `reads.parquet` + synthetic FASTQs → `.npy` coverage arrays for both (use with `evals/visualize.py`)

```bash
# 1. Generate reads
uv run python evals/generate.py --checkpoint checkpoints/epoch_010.pt --fold val --n-reads 100

# 2. Upload FASTQs to Modal and run pipeline
uv run modal volume put coaster-evals evals/output/reads /reads
uv run modal run evals/eval_modal.py --step all

# 3. Download results and compare
uv run modal volume get coaster-evals real_coverage evals/output/real_coverage
uv run modal volume get coaster-evals bigwigs evals/output/bigwigs
uv run python evals/compare.py --bigwig evals/output/bigwigs/synthetic_forward.bw --fold val
```

**Later tiers:**
- Sequence-level: edit distance to nearest real read, k-mer spectra (`evals/sequence_metrics.py`)
- Biological: splice junction usage, coverage at annotated features (`evals/bio_metrics.py`)

**Design notes:**
- Temperature sampling at T=1 is the principled choice: samples i.i.d. from P(read|DNA), reconstructing the learned distribution. Adjust upward only if empirical diversity is too low (overconfident model).
- The model was trained on reads 50–200 nt, so early EOS and runaway generation are not expected. Do not pre-emptively filter generated reads by length — only add filtering if this is actually observed, as filtering distorts the generated distribution.

## Repository layout

```
coaster/
  tokenizer.py          DNA (vocab=6) and RNA (vocab=8) tokenizers
  preprocessing.py      Strand helpers: revcomp, to_gene_sense, merge_pair
  model/
    config.py           EncoderConfig, DecoderConfig, TrainingConfig dataclasses + load_config()
    layers.py           RMSNorm, SinusoidalPosEmb
    encoder.py          DNAEncoder (embed → conv → transformer)
    decoder.py          RNADecoder (cross-attention, causal self-attention)
    transformer.py      CoasterModel (encoder + decoder + generate())
  data/
    dataset.py          collate_fn, make_dataloader()
    real_data.py        RealRNADataset (parquet-backed), center-crop/pad DNA to dna_len
  training/
    trainer.py          Trainer (train loop, eval, checkpointing, wandb logging)
scripts/
  train.py              Training entry point
  align_modal.py        Modal pipeline: genome download → STAR index → alignment
  extract_reads_modal.py  Modal pipeline: BAM → reads.parquet + samples_yeast.parquet
  download_sra.py       Local SRA download helper for SRR21628668
evals/
  generate.py           model → per-window FASTQs (two-phase encode + decode)
  eval_modal.py         Modal: STAR index → align → bigwigs + real/synth coverage extraction
  compute_coverage.py   Local alternative: reads → .npy coverage arrays via exact substring search
  compare.py            synthetic bigwig vs real .npy coverage → Pearson/Spearman metrics
  visualize.py          plot real vs synthetic coverage tracks per window
configs/
  default.yaml          Default hyperparameters
tests/                  58 pytest tests
```

## Data

**Real data:** SRR21628668 — BY UBR1 -469A>T *S. cerevisiae* RNA-seq, paired-end, ~152.6M read pairs (PRJNA882076). Aligned to sacCer3 / Ensembl R64-1-1 with STAR. Strand: reverse-stranded (dUTP/RF); R2 reads are sense-strand.

The processed data lives in the `coaster-data` Modal volume as two parquet files:

| File | Rows | Description |
|---|---|---|
| `samples_yeast.parquet` | 8,837 | Genomic windows: chr, strand, coordinates, 5000 bp DNA sequence, fold label |
| `reads.parquet` | 8,654,803 | Extracted reads: `sample_idx` (FK to samples), `read_seq` (151 nt RNA, T→U applied) |

To download them to `data/` locally or on a new cluster:

```bash
uv run modal volume get coaster-data samples_yeast.parquet data/samples_yeast.parquet
uv run modal volume get coaster-data reads.parquet data/reads.parquet
```

### Genomic windows (`samples.pkl` / `samples_yeast.parquet`)

`samples.pkl` was originally created for [Yorzoi](https://github.com/Tom-Ellis-Lab/yorzoi), a coverage-track predictor. We reuse it here because the DNA windows (centred on yeast genes with appropriate flanking) are non-trivial to compute. Relevant columns:

| Column | Description |
|---|---|
| `chr` | NCBI RefSeq chromosome name (e.g. `NC_001141.2`) |
| `strand` | Strand of `input_sequence`: `+` means the sequence runs 5′→3′ on the + genomic strand; `-` means it has been reverse-complemented so it still runs 5′→3′ along the gene |
| `start_seq` / `end_seq` | Genomic coordinates of the full 5000 bp window |
| `start_coverage` / `end_coverage` | Genomic coordinates of the inner ~3 kb coverage window |
| `input_sequence` | 5000 bp DNA in the 5′→3′ direction of the gene (always gene-sense, regardless of strand) |
| `fold` | `train` / `val` / `test` |
| `track_values` | Pointer to coverage track files from the yorzoi use-case — **ignored here** |

`hom_graph_id` and any other yorzoi-specific columns should also be ignored.

### Strand orientation and read extraction

Because `input_sequence` is always in the 5′→3′ gene-sense direction, training reads must also be in that direction. In a dUTP/RF paired-end library:

- **R2** is the sense read (same direction as the RNA transcript).
- **R1** is the antisense read; its reverse complement covers the 3′ end of the same fragment.

pysam stores reverse-strand-mapped reads as their reverse complement in the BAM (`query_sequence` is always in the + genomic strand direction). This means:

| Gene strand | R2 `query_sequence` | Action needed |
|---|---|---|
| `+` | + genomic = gene-sense ✓ | use as-is |
| `-` | + genomic = gene-**anti**sense ✗ | reverse-complement before saving |

`coaster/preprocessing.py` provides `to_gene_sense(query_sequence, gene_strand)` which applies the reverse-complement for `−` strand genes. This is called for both R1 and R2 in `scripts/extract_reads_modal.py`.

### Paired-end merging

Both reads of a pair span the same cDNA fragment from opposite ends. After converting to gene-sense direction, R2 covers the 5′ portion and R1 covers the 3′ portion. When the insert is short enough for them to overlap (common for yeast, where many fragments are <302 bp), they are merged into the full insert sequence:

```
Gene 5'→3':  [=======R2=======>]
                      [<=======R1=======]
Merged:      [===========full insert===========]
```

`coaster/preprocessing.py` provides `merge_pair(r2_sense, r1_sense)`, which finds the longest suffix/prefix overlap (minimum 10 nt) and returns the merged insert, or R2 alone if no overlap is found. `scripts/extract_reads_modal.py` fetches both R1 and R2 per pair, converts both to gene-sense, merges where possible, and applies reservoir sampling at the pair level.

**Note:** In practice ~42% of reads get merged and ~58% remain as R2-only (151 nt), so the training data is a mix of short (~151 nt) and longer (~150–300 nt) inserts. The decoder is autoregressive and length-agnostic, but the model implicitly learns two subtly different things without knowing which is which. Options to consider: only train on merged reads (drop 58% of data but cleaner), or condition the decoder on insert type. Revisit once baseline training is stable.


## Setup

Requires [uv](https://github.com/astral-sh/uv). A `.env` file must exist in the project root (can be empty).

```bash
uv sync
uv run pytest                   # run tests
```

## Training

```bash
# CUDA (specific GPU)
uv run python scripts/train.py --device cuda:0

# Override data paths
uv run python scripts/train.py \
    --samples /path/to/samples_yeast.parquet \
    --reads   /path/to/reads.parquet

# Sanity check: overfit a single batch to ~0 loss
uv run python scripts/train.py --overfit
```

The config device defaults to `cuda`. Override with `--device cuda:N` for a specific GPU, or `--device cpu`.

Checkpoints are written to `checkpoints/epoch_NNN.pt` after each epoch.

## Hyperparameters (default, `configs/default.yaml`)

| Parameter | Value |
|---|---|
| DNA input length | 4992 bp |
| Conv stride | 8 (4992 → 624 positions) |
| Encoder layers | 4 |
| Decoder layers | 4 |
| d\_model | 256 |
| Attention heads | 4 |
| FFN dim | 1024 |
| Max RNA length | 200 tokens |
| Batch size | 32 |
| Max steps | 500,000 |
| Eval / checkpoint interval | 10,000 steps |
| Learning rate | 3e-4 |
| LR schedule | Cosine with 500-step warmup |
| Weight decay | 0.01 |
| Grad clip | 1.0 |
