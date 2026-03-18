# coaster

A DNA-to-RNA encoder-decoder transformer that models yeast (*S. cerevisiae*) transcription. Given a ~4 kb genomic DNA window, the model learns to generate RNA-seq reads that would plausibly originate from that region.

## Motivation

RNA-seq read distributions encode everything that happens between DNA and mature RNA: transcription, splicing, degradation, and regulation. Rather than predicting a single summary statistic (e.g. expression level), coaster generates the full read-level distribution directly from sequence. The long-term goal is a generative model of yeast transcription that can simulate RNA-seq experiments in silico.

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
- **Synthetic training** — lazy dataset generates random DNA + sampled RNA windows on the fly; used to validate architecture
- **Trainer** — AdamW + cosine LR schedule with warmup; AMP on CUDA (autocast + grad scaler), float32 on MPS/CPU; checkpoint save/load
- **Tests** — 58 tests covering tokenizers, model shapes, forward/backward, data pipeline, and training loop

### Next steps
- Evaluate read sequence accuracy (e.g. edit distance, motif recovery)
- Unfreeze/replace encoder with NTv3 once real-data training is established

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
    synthetic.py        random_dna(), dna_to_rna_window(), generate_synthetic_pair()
    dataset.py          SyntheticDataset (lazy numpy), collate_fn, make_dataloader()
    real_data.py        RealRNADataset (parquet-backed), center-crop/pad DNA to dna_len
  training/
    trainer.py          Trainer (train loop, eval, checkpointing)
scripts/
  train.py              Training entry point (synthetic or real data)
  generate.py           (stub) inference script
  evaluate.py           (stub) evaluation script
  align_modal.py        Modal pipeline: genome download → STAR index → alignment
  extract_reads_modal.py  Modal pipeline: BAM → reads.parquet + samples_yeast.parquet
  download_sra.py       Local SRA download helper for SRR21628668
configs/
  default.yaml          Default hyperparameters
tests/                  58 pytest tests
```

## Data

**Real data:** SRR21628668 — BY UBR1 -469A>T *S. cerevisiae* RNA-seq, paired-end, ~152.6M read pairs (PRJNA882076). Aligned to sacCer3 / Ensembl R64-1-1 with STAR. Strand: reverse-stranded (dUTP/RF); R2 reads are sense-strand and used as training targets.

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

**Synthetic data (Phase 1):** Random 4992 bp DNA sequences; target reads are 150 nt windows sampled uniformly and T→U converted. Used to validate architecture before real data is wired up.

## Setup

Requires [uv](https://github.com/astral-sh/uv). A `.env` file must exist in the project root (can be empty).

```bash
uv sync
uv run pytest                   # run tests
```

## Training

```bash
# Real data (recommended) — CUDA
uv run python scripts/train.py --real --device cuda

# Real data — Apple Silicon
uv run python scripts/train.py --real

# Synthetic data only (no parquet files needed)
uv run python scripts/train.py --device cuda

# Override the data paths
uv run python scripts/train.py --real \
    --samples /path/to/samples_yeast.parquet \
    --reads   /path/to/reads.parquet

# Sanity check: overfit one batch to ~0 loss before a full run
uv run python scripts/train.py --real --overfit
```

The config device defaults to `mps`. Override with `--device cuda` or `--device cpu`, or change `training.device` in `configs/default.yaml`.

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
| Batch size | 5 |
| Learning rate | 3e-4 |
| LR schedule | Cosine with 500-step warmup |
| Weight decay | 0.01 |
| Grad clip | 1.0 |
