#!/bin/bash
# Submit VtD Ray training: 2 nodes × 4 GPUs = 8 GPUs
#
# Usage: bash scripts/submit_2node.sh

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
    --job-name=vtd_2node \
    --output=$LOG_DIR/vtd_2node_%j.log \
    --error=$LOG_DIR/vtd_2node_%j.log \
    "$SCRIPT_DIR/train_vtd_2node.sh"

echo "Submitted VtD training (2 nodes × 8 GPUs)."
echo "Logs:  $LOG_DIR/vtd_2node_<jobid>.log"
echo "Check: squeue -u \$USER"
