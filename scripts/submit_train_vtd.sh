#!/bin/bash
# Submit VtD Ray training job to SLURM on Anvil
#
# Usage:
#   bash scripts/submit_train_vtd.sh
#
# All stdout/stderr → ./logs/vtd_train_<jobid>.log
# Eval results      → ./logs/vtd_ray_gsm8k/eval_results/gsm8k/eval_step_*.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=./logs
mkdir -p $LOG_DIR

sbatch \
    --partition=ai \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres=gpu:4 \
    --account=cis250976-ai \
    --time=1:00:00 \
    --exclusive \
    --overcommit \
    --job-name=vtd_train \
    --output=$LOG_DIR/vtd_train_%j.log \
    --error=$LOG_DIR/vtd_train_%j.log \
    "$SCRIPT_DIR/train_vtd_ray.sh"

echo "Submitted VtD training job."
echo "Logs:         $LOG_DIR/vtd_train_<jobid>.log"
echo "Eval results: ./logs/vtd_ray_gsm8k/eval_results/gsm8k/"
echo "Check status: squeue -u \$USER"
