#!/bin/bash
# Submit Step 1: teacher generation job to SLURM on Anvil
#
# Usage:
#   bash teacher_uncertainty_each-sample/submit_generate.sh

LOG_DIR=/home/x-qlan1/code/moule2/teacher_uncertainty_each-sample/logs
mkdir -p $LOG_DIR

sbatch \
    --partition=ai \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --account=cis250976-ai \
    --time=08:00:00 \
    --job-name=unc_generate \
    --output=$LOG_DIR/unc_generate_%j.log \
    --error=$LOG_DIR/unc_generate_%j.log \
    /home/x-qlan1/code/moule2/teacher_uncertainty_each-sample/run_generate.sh

echo "Submitted Step 1: teacher generation job."
echo "Logs:    $LOG_DIR/unc_generate_<jobid>.log"
echo "Output:  teacher_uncertainty_each-sample/teacher_responses.jsonl"
echo "Check status: squeue -u \$USER"
