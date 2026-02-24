#!/bin/bash
#SBATCH --job-name=filter-70b
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --account=cis250976-ai
#SBATCH --time=00:30:00
#SBATCH --mem=400G
#SBATCH --exclude=h012
#SBATCH --output=logs/filter_70b_%j.log
#SBATCH --error=logs/filter_70b_%j.err

set -x

mkdir -p /home/x-qlan1/code/moule2/teacher_uncertainty_each-sample/logs

source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

SCRATCH=/anvil/scratch/x-qlan1/moule

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

SCRIPT_DIR=/home/x-qlan1/code/moule2/teacher_uncertainty_each-sample

python3 "$SCRIPT_DIR/filter_with_70b.py" \
    --model Qwen/Qwen2.5-72B-Instruct \
    --input "$SCRIPT_DIR/dataset_output/evolved_with_se_new.jsonl" \
    --output_dir "$SCRIPT_DIR/dataset_output" \
    --tp 4 \
    --max_model_len 4096 \
    --gpu_memory_utilization 0.9 \
    --batch_size 512
