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

SCRIPT_DIR=/home/x-qlan1/code/moule2/teacher_uncertainty_each-sample
OUTPUT_DIR=$SCRIPT_DIR

# ============================================================
# Step 2: Cluster answers + compute probability-weighted SE
#   - Qwen2.5-3B-Instruct judges ambiguous pairs
#   - Clusters by \boxed{} values (string match + LLM judge)
#   - p(c) = sum(seq_prob_i) / Z for each cluster
#   - SE = -sum(p(c) * log(p(c)))
#   - Output: final JSONL with se, se_weight fields
# ============================================================

echo "=========================================="
echo "Step 2: Clustering + computing uncertainty"
echo "=========================================="

python3 "$SCRIPT_DIR/uncertainty_calculation.py" \
    --cluster_model Qwen/Qwen3-8B \
    --gpu_memory_utilization 0.85 \
    --input "$OUTPUT_DIR/teacher_responses.jsonl" \
    --output "$OUTPUT_DIR/dataset_output/evolved_with_se_new.jsonl" \
    --batch_size 1024

echo "=========================================="
echo "Step 2 Done!"
echo "=========================================="
echo "Output: $OUTPUT_DIR//dataset_output/evolved_with_se.jsonl"
echo "Count:  $(wc -l < "$OUTPUT_DIR/evolved_with_se.jsonl")"

echo ""
echo "SE distribution:"
python3 -c "
import json
with open('$OUTPUT_DIR/evolved_with_se.jsonl') as f:
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
