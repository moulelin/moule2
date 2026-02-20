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
# Model: Qwen2.5-72B-Instruct
#   - 72B params, TP=4 on 4x H100 (~36GB/GPU)
#   - MATH: 83.1%, GSM8K: 95.8%
# ============================================================

MODEL="Qwen/Qwen2.5-72B-Instruct"

echo "=========================================="
echo "Step 1: Generating evolved math problems"
echo "  Model: $MODEL"
echo "  Datasets: all tiers"
echo "=========================================="

python3 "$SCRIPT_DIR/generate.py" \
    --model "$MODEL" \
    --tp 4 \
    --max_model_len 4096 \
    --gpu_memory_utilization 0.90 \
    --datasets all \
    --temperature 0.7 \
    --top_p 0.95 \
    --max_tokens 1024 \
    --batch_size 256 \
    --output "$OUTPUT_DIR/evolved_raw.jsonl" \
    --seed 42

echo "=========================================="
echo "Step 2: Filtering and deduplication"
echo "=========================================="

python3 "$SCRIPT_DIR/filter_dedup.py" \
    --input "$OUTPUT_DIR/evolved_raw.jsonl" \
    --output "$OUTPUT_DIR/evolved_filtered.jsonl" \
    --min_len 30 \
    --max_len 2000 \
    --dedup \
    --dedup_threshold 0.7

echo "=========================================="
echo "Done!"
echo "=========================================="
echo "Raw:      $OUTPUT_DIR/evolved_raw.jsonl"
echo "Filtered: $OUTPUT_DIR/evolved_filtered.jsonl"
echo "Training: $OUTPUT_DIR/evolved_filtered_train.jsonl"
echo ""

echo "Raw count:      $(wc -l < "$OUTPUT_DIR/evolved_raw.jsonl")"
echo "Filtered count: $(wc -l < "$OUTPUT_DIR/evolved_filtered.jsonl")"
echo "Training count: $(wc -l < "$OUTPUT_DIR/evolved_filtered_train.jsonl")"

echo ""
echo "Per-difficulty distribution:"
python3 -c "
import json
from collections import Counter
with open('$OUTPUT_DIR/evolved_filtered.jsonl') as f:
    diffs = Counter(json.loads(l)['difficulty'] for l in f if l.strip())
for diff, cnt in sorted(diffs.items(), key=lambda x: -x[1]):
    print(f'  {diff}: {cnt}')
"
