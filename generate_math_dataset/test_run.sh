#!/bin/bash
set -x

# ---- Activate environment ----
source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

# ---- Paths ----
SCRATCH=/anvil/scratch/x-qlan1/moule
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"

# ---- Storage ----
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache

# ---- Fix GLIBC for flash_attn ----
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# ============================================================
# TEST RUN: small model + small dataset + 1 GPU
#   Model: Qwen2.5-7B-Instruct (TP=1, single GPU)
#   Data:  AIME 2025 (~30) + HMMT Feb 2025 (~10)
#   ~40 seeds × 5 rounds = ~200 generations, ~5 min
# ============================================================

echo "=========================================="
echo "TEST RUN"
echo "=========================================="

python3 "$SCRIPT_DIR/generate.py" \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tp 1 \
    --max_model_len 4096 \
    --gpu_memory_utilization 0.90 \
    --datasets aime25,hmmt_feb25 \
    --temperature 0.7 \
    --top_p 0.95 \
    --max_tokens 1024 \
    --batch_size 64 \
    --output "$OUTPUT_DIR/test_evolved.jsonl" \
    --seed 42

echo "=========================================="
echo "Filtering"
echo "=========================================="

python3 "$SCRIPT_DIR/filter_dedup.py" \
    --input "$OUTPUT_DIR/test_evolved.jsonl" \
    --output "$OUTPUT_DIR/test_filtered.jsonl" \
    --min_len 30 \
    --max_len 2000 \
    --dedup \
    --dedup_threshold 0.7

echo "=========================================="
echo "Results"
echo "=========================================="
echo "Raw:      $(wc -l < "$OUTPUT_DIR/test_evolved.jsonl")"
echo "Filtered: $(wc -l < "$OUTPUT_DIR/test_filtered.jsonl")"
echo ""
echo "Sample outputs:"
head -5 "$OUTPUT_DIR/test_evolved.jsonl" | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(f\"[r{d['round']}|{d['strategy']}] {d['question'][:120]}...\")
    print()
"
