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
mkdir -p "$OUTPUT_DIR"

# ============================================================
# Step 1: Teacher generates 8 responses per question
#   - Qwen3-8B via vLLM (TP=1, 1x H100, no thinking)
#   - Extracts token-level logprobs + sequence probability
#   - Extracts \boxed{} answers
#   - Output: intermediate JSONL for step 2
# ============================================================

echo "=========================================="
echo "Step 1: Generating teacher responses"
echo "=========================================="

python3 "$SCRIPT_DIR/uncertainty_generate.py" \
    --teacher_model Qwen/Qwen3-8B \
    --tp 1 \
    --max_model_len 4096 \
    --gpu_memory_utilization 0.85 \
    --n_samples 8 \
    --temperature 0.7 \
    --max_gen_tokens 1024 \
    --input "$SCRIPT_DIR/evolved_clean.jsonl" \
    --output "$OUTPUT_DIR/teacher_responses.jsonl" \
    --batch_size 300

echo "=========================================="
echo "Step 1 Done!"
echo "=========================================="
echo "Output: $OUTPUT_DIR/teacher_responses.jsonl"
echo "Count:  $(wc -l < "$OUTPUT_DIR/teacher_responses.jsonl")"