#!/bin/bash
# Submit GSM8K eval job to SLURM on Anvil
#
# Usage:
#   bash scripts/eval/submit_eval_gsm8k.sh

SCRIPT_DIR=/home/x-qlan1/code/moule/scripts/eval
LOG_DIR=./eval_logs
mkdir -p $LOG_DIR

sbatch \
    --partition=ai \
    --nodes=1 \
    --ntasks=32 \
    --gres=gpu:2 \
    --account=cis250976-ai \
    --time=04:00:00 \
    --job-name=eval_gsm8k \
    --output=$LOG_DIR/eval_gsm8k_%j.log \
    --error=$LOG_DIR/eval_gsm8k_%j.log \
    "$SCRIPT_DIR/eval_gsm8k.sh"

echo "Submitted GSM8K eval job."
echo "Logs:    $LOG_DIR/eval_gsm8k_<jobid>.log"
echo "Results: scripts/eval/results/"
echo "Check status: squeue -u \$USER"
