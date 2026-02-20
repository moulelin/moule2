#!/bin/bash
# Submit 4 eval jobs to SLURM (one per dataset, each runs 3 models × 2 modes)
# Usage: bash scripts/eval/submit_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH=/anvil/scratch/x-qlan1/moule
SCRATCH_LOG=./
LOG_DIR=$SCRATCH_LOG/eval_logs

# ---- Clean up previous logs ----
rm -rf $LOG_DIR
mkdir -p $LOG_DIR

COMMON="--partition=ai --nodes=1 --ntasks=32 --gres=gpu:2 --account=cis250976-ai --time=02:00:00"

# sbatch $COMMON \
#     --job-name=eval_aime24 \
#     --output=$LOG_DIR/eval_aime24_%j.log \
#     --error=$LOG_DIR/eval_aime24_%j.log \
#     "$SCRIPT_DIR/eval_aime24.sh"

# sbatch $COMMON \
#     --job-name=eval_aime25 \
#     --output=$LOG_DIR/eval_aime25_%j.log \
#     --error=$LOG_DIR/eval_aime25_%j.log \
#     "$SCRIPT_DIR/eval_aime25.sh"

sbatch $COMMON \
    --job-name=eval_hmmt25 \
    --output=$LOG_DIR/eval_hmmt25_%j.log \
    --error=$LOG_DIR/eval_hmmt25_%j.log \
    "$SCRIPT_DIR/eval_hmmt25.sh"

# sbatch $COMMON \
#     --job-name=eval_amo_bench \
#     --output=$LOG_DIR/eval_amo_bench_%j.log \
#     --error=$LOG_DIR/eval_amo_bench_%j.log \
#     "$SCRIPT_DIR/eval_amo_bench.sh"

echo "Submitted 4 eval jobs. Logs: $LOG_DIR"
echo "Check status: squeue -u \$USER"
