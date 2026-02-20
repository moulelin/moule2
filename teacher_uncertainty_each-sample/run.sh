#!/bin/bash
set -x

# ---- Activate environment ----
source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

# ---- Model cache ----
SCRATCH=/anvil/scratch/x-qlan1/moule
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache

# ---- Fix GLIBC ----
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============================================================
# Offline teacher uncertainty computation
#   Teacher: Qwen3-8B (TP=1, 1x H100, no thinking)
#   Cluster: Qwen2.5-3B-Instruct (shares GPU, judges ambiguous pairs)
#   Dataset: 45K samples from evolved_clean.jsonl
#   N=8 responses per sample
#
#   Per sample: teacher generates 8 responses (vLLM, fast)
#               → extract \boxed{}, string-match + 3B judge → SE
#
#   Estimated time: ~1-2 hours
# ============================================================

echo "=========================================="
echo "Computing teacher uncertainty (SE)"
echo "=========================================="

python3 "$SCRIPT_DIR/compute_uncertainty.py" \
    --teacher_model Qwen/Qwen3-8B \
    --tp 1 \
    --max_model_len 4096 \
    --gpu_memory_utilization 0.65 \
    --cluster_model Qwen/Qwen2.5-3B-Instruct \
    --n_samples 8 \
    --temperature 0.7 \
    --max_gen_tokens 1024 \
    --input "$SCRIPT_DIR/evolved_clean.jsonl" \
    --output "$SCRIPT_DIR/evolved_with_se.jsonl" \
    --batch_size 32

echo "=========================================="
echo "Done!"
echo "=========================================="
echo "Output: $SCRIPT_DIR/evolved_with_se.jsonl"
echo "Count:  $(wc -l < "$SCRIPT_DIR/evolved_with_se.jsonl")"

echo ""
echo "SE distribution:"
python3 -c "
import json
with open('$SCRIPT_DIR/evolved_with_se.jsonl') as f:
    ses = [json.loads(l)['se'] for l in f if l.strip()]
n = len(ses)
if n == 0:
    print('  No samples produced — check for errors above.')
else:
    avg = sum(ses)/n
    low = sum(1 for s in ses if s < 0.1)
    mid = sum(1 for s in ses if 0.1 <= s < 0.5)
    high = sum(1 for s in ses if s >= 0.5)
    print(f'  Total: {n}')
    print(f'  Avg SE: {avg:.3f}, Avg weight: {1-avg:.3f}')
    print(f'  Confident (SE<0.1): {low} ({low/n:.1%})')
    print(f'  Moderate (0.1<=SE<0.5): {mid} ({mid/n:.1%})')
    print(f'  Uncertain (SE>=0.5): {high} ({high/n:.1%})')
"
