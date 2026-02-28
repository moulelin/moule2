#!/bin/bash
set -x

source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

SCRATCH=/anvil/scratch/x-qlan1/moule2
SCRIPT_DIR=/home/x-qlan1/code/moule2/dataset/scripts
DATA_DIR=/home/x-qlan1/code/moule2/dataset

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

python3 "$SCRIPT_DIR/translate_math.py" \
    --model Qwen/Qwen2.5-32B-Instruct \
    --clean_log "$DATA_DIR/cleaned/math_clean_log.jsonl" \
    --augment_log "$DATA_DIR/augmented/math_augment_log.jsonl" \
    --output_dir "$DATA_DIR/merged" \
    --tp 2 \
    --batch_size 64 \
    --max_tokens 2048 \
    --max_model_len 4096 \
    --gpu_memory_utilization 0.9
