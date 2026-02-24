#!/bin/bash
#SBATCH --job-name=unc-calc
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --account=cis250976-ai
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --output=logs/unc_calc_%j.log
#SBATCH --error=logs/unc_calc_%j.err

mkdir -p /home/x-qlan1/code/moule2/teacher_uncertainty_each-sample/logs

bash /home/x-qlan1/code/moule2/teacher_uncertainty_each-sample/run_calculation.sh
