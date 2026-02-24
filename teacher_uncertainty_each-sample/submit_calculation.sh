#!/bin/bash
# Submit Step 2: clustering + SE computation job to SLURM on Anvil
#
# Usage:
#   bash teacher_uncertainty_each-sample/submit_calculation.sh

LOG_DIR=/home/x-qlan1/code/moule2/teacher_uncertainty_each-sample/logs
mkdir -p $LOG_DIR

sbatch \
    --partition=ai \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --account=cis250976-ai \
    --time=02:00:00 \
    --job-name=unc_calc \
    --output=$LOG_DIR/unc_calc_%j.log \
    --error=$LOG_DIR/unc_calc_%j.log \
    /home/x-qlan1/code/moule2/teacher_uncertainty_each-sample/run_calculation.sh

echo "Submitted Step 2: clustering + SE computation job."
echo "Logs:    $LOG_DIR/unc_calc_<jobid>.log"
echo "Output:  teacher_uncertainty_each-sample/evolved_with_se.jsonl"
echo "Check status: squeue -u \$USER"
