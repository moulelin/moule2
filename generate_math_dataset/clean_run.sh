#!/bin/bash
set -x

# ---- Activate environment ----
source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

# ---- Paths ----
SCRATCH=/anvil/scratch/x-qlan1/moule
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Storage ----
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache

# ---- Fix GLIBC for flash_attn ----
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# ============================================================
# LLM quality filter
#   Model: Qwen2.5-14B-Instruct (single GPU, ~28GB)
#   Input: evolved_raw.jsonl (from generate step)
#   Output: evolved_clean.jsonl (valid problems only)
# ============================================================

INPUT="$SCRIPT_DIR/evolved_raw.jsonl"
OUTPUT="$SCRIPT_DIR/evolved_clean.jsonl"

echo "=========================================="
echo "LLM Quality Filter"
echo "  Input:  $INPUT"
echo "  Output: $OUTPUT"
echo "=========================================="

python3 "$SCRIPT_DIR/clean_with_llm.py" \
    --model Qwen/Qwen2.5-14B-Instruct \
    --tp 1 \
    --max_model_len 4096 \
    --gpu_memory_utilization 0.90 \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --batch_size 512

echo "=========================================="
echo "Done!"
echo "Clean:   $(wc -l < "$OUTPUT")"
echo "Removed: $(wc -l < "${OUTPUT%.jsonl}_removed.jsonl")"
echo "=========================================="
