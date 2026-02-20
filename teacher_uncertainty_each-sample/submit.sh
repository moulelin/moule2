#!/bin/bash
# Submit SE computation job to SLURM on Anvil
#
# Usage:
#   bash teacher_uncertainty_each-sample/submit.sh

LOG_DIR=/home/x-qlan1/code/moule2/teacher_uncertainty_each-sample/logs
mkdir -p $LOG_DIR

sbatch \
    --partition=ai \
    --nodes=1 \
    --ntasks=32 \
    --gres=gpu:1 \
    --account=cis250976-ai \
    --time=02:00:00 \
    --job-name=se_compute \
    --output=$LOG_DIR/se_%j.log \
    --error=$LOG_DIR/se_%j.log \
    /home/x-qlan1/code/moule2/teacher_uncertainty_each-sample/run.sh

echo "Submitted SE computation job."
echo "Logs:    $LOG_DIR/se_<jobid>.log"
echo "Output:  teacher_uncertainty_each-sample/evolved_with_se.jsonl"
echo "Check status: squeue -u \$USER"
