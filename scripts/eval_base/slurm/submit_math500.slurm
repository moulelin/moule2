#!/bin/bash
#SBATCH --job-name=eval-math500-base
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --account=cis250976-ai
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --exclude=h012
#SBATCH --output=logs/eval_math500_base_%j.log
#SBATCH --error=logs/eval_math500_base_%j.err

set -x

mkdir -p /home/x-qlan1/code/moule2/scripts/eval_base/logs

bash /home/x-qlan1/code/moule2/scripts/eval_base/sh/eval_math500.sh
