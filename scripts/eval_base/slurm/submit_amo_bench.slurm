#!/bin/bash
#SBATCH --job-name=eval-amo-base
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --account=cis250976-ai
#SBATCH --time=08:00:00
#SBATCH --mem=200G
#SBATCH --exclude=h012
#SBATCH --output=logs/eval_amo_base_%j.log
#SBATCH --error=logs/eval_amo_base_%j.err

set -x

mkdir -p /home/x-qlan1/code/moule2/scripts/eval_base/logs

bash /home/x-qlan1/code/moule2/scripts/eval_base/sh/eval_amo_bench.sh
