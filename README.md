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
- **Trainer** — AdamW + cosine LR schedule with warmup; AMP on CUDA (autocast + grad scaler), float32 on MPS/CPU; checkpoint save/load
- **Tests** — 58 tests covering tokenizers, model shapes, forward/backward, data pipeline, and training loop

### Next steps
- ~~Switch training loop from epochs to total steps~~ — done: `max_steps` + `eval_interval` replace `num_epochs`
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
- Downstream: synthetic bigwigs as input to an expression predictor

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
