#!/bin/bash
set -x

source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

SCRATCH=/anvil/scratch/x-qlan1/moule
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export CUDA_VISIBLE_DEVICES=0,1

SCRIPT_DIR=/home/x-qlan1/code/moule2/teacher_uncertainty_each-sample/pdf_extract

python3 "$SCRIPT_DIR/pdf_to_markdown.py" \
    --input "$SCRIPT_DIR/math.pdf" \
    --output_dir "$SCRIPT_DIR" \
    --force_ocr
