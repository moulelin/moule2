#!/bin/bash
#SBATCH --job-name=augment-math
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --account=cis250976-ai
#SBATCH --time=01:00:00
#SBATCH --mem=400G
#SBATCH --exclude=h012
#SBATCH --output=logs/augment_math_%j.log
#SBATCH --error=logs/augment_math_%j.err

set -x

mkdir -p /home/x-qlan1/code/moule2/dataset/scripts/logs

bash /home/x-qlan1/code/moule2/dataset/scripts/run_augment_math.sh
