#!/bin/bash
# Submit VtD Ray training job to SLURM on Anvil (2 nodes × 4 GPUs)
#
# Usage:
#   bash scripts/submit_train_vtd.sh
#
# All stdout/stderr → ./logs/vtd_train_<jobid>.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=./logs
mkdir -p $LOG_DIR

sbatch \
    --partition=ai \
    --nodes=2 \
    --ntasks-per-node=1 \
    --gres=gpu:4 \
    --account=cis250976-ai \
    --time=2:00:00 \
    --exclusive \
    --overcommit \
    --job-name=vtd_train_2node \
    --output=$LOG_DIR/vtd_train_%j.log \
    --error=$LOG_DIR/vtd_train_%j.log \
    "$SCRIPT_DIR/train_vtd_ray.sh"

echo "Submitted VtD training job (2 nodes × 4 GPUs)."
echo "Logs:  $LOG_DIR/vtd_train_<jobid>.log"
echo "Check: squeue -u \$USER"
