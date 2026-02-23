#!/bin/bash
#SBATCH --job-name=unc-generate
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --account=cis250976-ai
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --output=logs/unc_generate_%j.log
#SBATCH --error=logs/unc_generate_%j.err

mkdir -p /home/x-qlan1/code/moule2/teacher_uncertainty_each-sample/logs

bash /home/x-qlan1/code/moule2/teacher_uncertainty_each-sample/run_generate.sh
