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

RNA read (one per forward pass, ~150 nt)
  → RNATokenizer (PAD/BOS/EOS/A/U/G/C/N, vocab=8)
  → Embedding(8, 256) + LearnedPosEmb
  → TransformerDecoder × 4  (d=256, 4 heads, ffn=1024, pre-norm, cross-attn to memory)
  → RMSNorm → Linear(256, 8)
  → logits  [B, T, 8]
```

**~5.8M parameters.** Training objective: teacher-forced cross-entropy on RNA token sequences, one read per forward pass.

### Key design choices

| Choice | Rationale |
|---|---|
| Conv1d stride-8 downsampling | Reduces 4992 DNA positions to 624 before self-attention, keeping memory and compute manageable |
| Sinusoidal pos emb (encoder) | Fixed, no learned parameters needed for fixed-length DNA input |
| Learned pos emb (decoder) | Better at short sequences (~152 tokens); positions are absolute within a read |
| Pre-norm (`norm_first=True`) | More stable training at small batch sizes |
| One read per forward pass | Simplest formulation; the model is free to generate any plausible read from the DNA context |
| `num_workers=0` | Required for MPS — fork-based DataLoader workers conflict with MPS device initialization |

## Project status

### Done
- **Data pipeline** — `scripts/align_modal.py`: full Modal-based pipeline to download the sacCer3 reference genome (Ensembl R111), build a STAR index, download SRR21628668 paired-end RNA-seq reads (~14 GB from NCBI S3), and align to produce a sorted BAM + gene counts
- **Model** — full encoder-decoder architecture (`coaster/model/`)
- **Synthetic training** — lazy dataset generates random DNA + sampled RNA windows on the fly; collate with BOS/EOS framing
- **Trainer** — AdamW + cosine LR schedule with warmup; AMP-gated (float32 on MPS, autocast on CUDA); checkpoint save/load
- **Tests** — 58 tests covering tokenizers, model shapes, forward/backward, data pipeline, and training loop

### In progress / next steps
- Wire up real data: extract reads from the aligned BAM (SRR21628668) and pair each read with its genomic DNA window
- Replace synthetic dataset with real `(DNA window, RNA read)` pairs
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
  training/
    trainer.py          Trainer (train loop, eval, checkpointing)
scripts/
  train.py              Training entry point
  generate.py           (stub) inference script
  evaluate.py           (stub) evaluation script
  align_modal.py        Modal pipeline: genome download → STAR index → alignment
  download_sra.py       Local SRA download helper for SRR21628668
configs/
  default.yaml          Default hyperparameters
tests/                  58 pytest tests
```

## Data

**Real data target:** SRR21628668 — BY UBR1 -469A>T *S. cerevisiae* RNA-seq, paired-end, ~152.6M read pairs (PRJNA882076). Aligned to sacCer3 / Ensembl R64-1-1 with STAR.

**Synthetic data (Phase 1):** Random 4992 bp DNA sequences; target reads are 150 nt windows sampled uniformly and T→U converted. Used to validate architecture before real data is wired up.

## Setup

Requires [uv](https://github.com/astral-sh/uv). A `.env` file must exist in the project root (can be empty).

```bash
uv sync
uv run pytest          # run tests
uv run python scripts/train.py --config configs/default.yaml
uv run python scripts/train.py --device cpu   # override device
```

Training defaults to `device: mps` (Apple Silicon). Falls back to CPU automatically if MPS is unavailable.

## Hyperparameters (default)

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
| Learning rate | 3e-4 |
| LR schedule | Cosine with 500-step warmup |
| Training samples | 50 000 (synthetic) |
