#!/bin/bash
# Submit VtD Ray training: 1 node × 4 GPUs
#
# Usage: bash scripts/submit_1node.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=./logs
mkdir -p $LOG_DIR

sbatch \
    --partition=ai \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres=gpu:4 \
    --account=cis250976-ai \
    --time=7:00:00 \
    --overcommit \
    --job-name=vtd_1node \
    --output=$LOG_DIR/vtd_1node_%j.log \
    --error=$LOG_DIR/vtd_1node_%j.log \
    "$SCRIPT_DIR/train_vtd_1node.sh"

echo "Submitted VtD training (1 node × 4 GPUs)."
echo "Logs:  $LOG_DIR/vtd_1node_<jobid>.log"
echo "Check: squeue -u \$USER"
