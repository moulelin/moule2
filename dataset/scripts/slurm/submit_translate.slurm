#!/bin/bash
#SBATCH --job-name=translate-math
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --account=cis250976-ai
#SBATCH --time=01:00:00
#SBATCH --mem=200G
#SBATCH --exclude=h012
#SBATCH --output=logs/translate_math_%j.log
#SBATCH --error=logs/translate_math_%j.err

set -x

mkdir -p /home/x-qlan1/code/moule2/dataset/scripts/logs

bash /home/x-qlan1/code/moule2/dataset/scripts/run_translate.sh
