#!/bin/bash
set -x

source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

SCRATCH=/anvil/scratch/x-qlan1/moule

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

SCRIPT_DIR=/home/x-qlan1/code/moule2/dataset/scripts

python3 "$SCRIPT_DIR/augment_math_csv.py" \
    --model Qwen/Qwen2.5-72B-Instruct \
    --input /home/x-qlan1/code/moule2/dataset/cleaned/math_cleaned.csv \
    --output_dir /home/x-qlan1/code/moule2/dataset/augmented \
    --tp 4 \
    --max_model_len 4096 \
    --max_tokens 1024 \
    --gpu_memory_utilization 0.90 \
    --batch_size 32
