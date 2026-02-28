#!/bin/bash
set -x

source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

SCRATCH=/anvil/scratch/x-qlan1/moule2
SCRIPT_DIR=/home/x-qlan1/code/moule2/dataset/scripts

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

python3 "$SCRIPT_DIR/normalize_intervals.py" \
    --model Qwen/Qwen2.5-32B-Instruct \
    --input /home/x-qlan1/code/moule2/dataset/merged/math_merged_en_2.jsonl \
    --tp 2
