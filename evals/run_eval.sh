#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CHECKPOINT="checkpoints/last.pt"
OUTPUT_DIR="evals/output/reads"
FOLD="val"
N_READS=500
DEVICE="cuda:3"

echo "=== [1/2] Generating reads ==="
uv run python evals/generate.py \
    --checkpoint "$CHECKPOINT" \
    --fold "$FOLD" \
    --n-reads "$N_READS" \
    --device "$DEVICE" \
    --encode-batch 32 \
    --decode-batch 64 \
    --output-dir "$OUTPUT_DIR"

echo "=== [2/2] Plotting coverage ==="
uv run python evals/visualize.py \
    --real-coverage evals/output/real_coverage \
    --synth-coverage evals/output/synth_coverage \
    --samples data/samples_yeast.parquet \
    --fold "$FOLD" \
    --output evals/output/coverage.png

echo "=== Done ==="
