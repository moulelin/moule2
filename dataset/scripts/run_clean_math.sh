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

python3 "$SCRIPT_DIR/clean_math_csv.py" \
    --model Qwen/Qwen2.5-72B-Instruct \
    --input /home/x-qlan1/code/moule2/dataset/raw/math.xlsx \
    --output_dir /home/x-qlan1/code/moule2/dataset/cleaned \
    --tp 4 \
    --max_model_len 6666 \
    --max_tokens 2048 \
    --gpu_memory_utilization 0.85 \
    --batch_size 64
