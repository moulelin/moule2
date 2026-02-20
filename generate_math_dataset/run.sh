#!/bin/bash
set -x

# ---- Activate environment ----
source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

# ---- Paths ----
SCRATCH=/anvil/scratch/x-qlan1/moule
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRATCH/datasets/evolved_math"
mkdir -p "$OUTPUT_DIR"

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
#   - Non-reasoning model = faster output, no <think> overhead
# ============================================================

MODEL="Qwen/Qwen2.5-72B-Instruct"

# ============================================================
# Seed datasets (multi-tier difficulty):
#
#   easy (30K seeds, 1 round):
#     - ScaleQuest-Math: 1M general math
#
#   medium (32.5K seeds, 2 rounds):
#     - NuminaMath-CoT: 860K math + CoT
#     - MATH (Hendrycks): 12.5K competition
#
#   hard (9.4K seeds, 3 rounds):
#     - Omni-MATH: 4.4K Olympiad level
#     - OlympiadBench: 8.4K Olympiad math/physics
#
#   competition (~130 seeds, 5 rounds each):
#     - AIME 2024: 30 problems
#     - AIME 2025: ~30 problems
#     - HMMT Feb 2025: ~10 problems
#     - HMMT Nov 2025: ~10 problems
#     - AMO-Bench: 50 IMO+ level problems
#
# Total seeds: ~72K
# Total generation calls (with multi-round): ~124K
# Expected output after dedup: ~100K-120K evolved problems
# ============================================================

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

# Count results
echo "Raw count:      $(wc -l < "$OUTPUT_DIR/evolved_raw.jsonl")"
echo "Filtered count: $(wc -l < "$OUTPUT_DIR/evolved_filtered.jsonl")"
echo "Training count: $(wc -l < "$OUTPUT_DIR/evolved_filtered_train.jsonl")"

# Show per-source distribution
echo ""
echo "Per-source distribution:"
python3 -c "
import json
from collections import Counter
with open('$OUTPUT_DIR/evolved_filtered.jsonl') as f:
    sources = Counter(json.loads(l)['source'] for l in f if l.strip())
for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
    print(f'  {src}: {cnt}')
"
