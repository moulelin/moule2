#!/bin/bash
set -x

source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

SCRATCH=/anvil/scratch/x-qlan1/moule
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

SCRIPT_DIR=/home/x-qlan1/code/moule2/dataset/scripts/py
INPUT=/home/x-qlan1/code/moule2/dataset/cleaned/math_en_sample500_rewritten.jsonl
OUTPUT=/home/x-qlan1/code/moule2/dataset/cleaned/math_en_sample500_with_se.jsonl

python3 "$SCRIPT_DIR/compute_se_augmented.py" \
    --teacher_model Qwen/Qwen3-8B \
    --tp 2 \
    --max_model_len 4096 \
    --gpu_memory_utilization 0.80 \
    --cluster_model Qwen/Qwen3-4B \
    --n_samples 8 \
    --temperature 0.7 \
    --max_gen_tokens 4096 \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --batch_size 128
