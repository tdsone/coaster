# coaster

A DNA-to-RNA encoder-decoder transformer that models yeast (*S. cerevisiae*) transcription. Given a ~4 kb genomic DNA window, the model learns to generate RNA-seq reads that would plausibly originate from that region.

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
    - [Paired-end merging (planned)](#paired-end-merging-planned)
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
- **Real data** — `RealRNADataset` pairs 8.65M extracted reads with their 5000 bp DNA windows across train/val/test folds
- **Trainer** — AdamW + cosine LR schedule with warmup; AMP on CUDA (autocast + grad scaler), float32 on MPS/CPU; checkpoint save/load
- **Tests** — 58 tests covering tokenizers, model shapes, forward/backward, data pipeline, and training loop

### Next steps
- **Evaluation pipeline** (see structure below)
- Unfreeze/replace encoder with NTv3 once real-data training is established

### Evaluation pipeline

**Stages** (each independently runnable):
1. `evals/generate.py` — local: model → per-window FASTQs via `generate_reads(model, dna_seq, n_reads, temperature=1.0)`
2. `evals/eval_modal.py` — Modal, 4 steps:
   - `build_index`: concatenate window sequences as chromosomes → STAR index
   - `align`: synthetic FASTQs → sorted BAM (ENCODE STAR params, matching `scripts/align_modal.py`)
   - `bigwigs`: BAM → forward/reverse strand CPM bigwigs (deeptools bamCoverage, bin=1)
   - `real_coverage`: extract per-window sense-strand coverage from real sacCer3 BAM → `.npy` arrays
3. `evals/compare.py` — local: synthetic bigwig vs real `.npy` coverage → per-window + aggregate Pearson/Spearman

```bash
# 1. Generate reads
uv run python evals/generate.py --checkpoint checkpoints/epoch_010.pt --fold val --n-reads 100

# 2. Upload FASTQs to Modal and run pipeline
modal volume put coaster-evals evals/output/reads /reads
uv run modal run evals/eval_modal.py --step all

# 3. Download results and compare
modal volume get coaster-evals real_coverage evals/output/real_coverage
modal volume get coaster-evals bigwigs evals/output/bigwigs
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
  train.py              Training entry point (synthetic or real data)
  align_modal.py        Modal pipeline: genome download → STAR index → alignment
  extract_reads_modal.py  Modal pipeline: BAM → reads.parquet + samples_yeast.parquet
  download_sra.py       Local SRA download helper for SRR21628668
evals/
  generate.py           model → per-window FASTQs (temperature sampling)
  eval_modal.py         Modal: STAR index → align → bigwigs + real coverage extraction
  compare.py            synthetic vs real coverage → Pearson/Spearman metrics
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
modal volume get coaster-data samples_yeast.parquet data/samples_yeast.parquet
modal volume get coaster-data reads.parquet data/reads.parquet
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

**Consequence:** the current `extract_reads_modal.py` has a bug — for `−` strand genes it stores reads in the gene-antisense direction, so the model trains on (gene-sense DNA, gene-antisense reads). This needs to be fixed in the next extraction run.

### Paired-end merging (planned)

Both reads of a pair span the same cDNA fragment from opposite ends. After converting to gene-sense direction, R2 covers the 5′ portion and RC(R1) covers the 3′ portion. When the insert is short enough that they overlap (common for yeast, where many fragments are <302 bp), they can be merged into the full insert sequence:

```
Gene 5'→3':  [=======R2=======>]
                      [<=======R1=======]   (R1 maps opposite strand; gene-sense = RC of query_sequence for + gene)
Merged:      [===========full insert===========]
```

The extraction script should be updated to:
1. Fetch both R1 and R2 for each pair (match by `query_name`).
2. Convert both to gene-sense direction (RC both if `strand == '-'`).
3. Merge overlapping pairs into one longer read; keep R2 only for non-overlapping pairs.
4. Apply reservoir sampling at the pair level.

**TODO:** In practice ~42% of reads get merged and ~58% remain as R2-only (151 nt), so the
training data is a mix of short (~151 nt) and longer (~150–300 nt) inserts. This is fine
architecturally (the decoder is autoregressive and length-agnostic) but the model is implicitly
learning two subtly different things without knowing which is which. Options to consider:
only train on merged reads (drop 58% of data but cleaner), always produce a fixed-length
representation, or condition the decoder on insert type. Revisit once baseline training is stable.


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
