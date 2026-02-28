#!/bin/bash
#SBATCH --job-name=se-sample500
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=cis250976-ai
#SBATCH --time=02:00:00
#SBATCH --mem=400G
#SBATCH --exclude=h012
#SBATCH --output=/home/x-qlan1/code/moule2/dataset/scripts/logs/se_sample500_%j.log
#SBATCH --error=/home/x-qlan1/code/moule2/dataset/scripts/logs/se_sample500_%j.err

set -x

mkdir -p /home/x-qlan1/code/moule2/dataset/scripts/logs

source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

SCRATCH=/anvil/scratch/x-qlan1/moule
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

SCRIPT_DIR=/home/x-qlan1/code/moule2/dataset/scripts/py
INPUT=/home/x-qlan1/code/moule2/dataset/cleaned/math_en_sample500_rewritten.jsonl
OUTPUT=/home/x-qlan1/code/moule2/dataset/cleaned/math_en_sample500_with_se.jsonl

python3 "$SCRIPT_DIR/compute_se_augmented.py" \
    --teacher_model Qwen/Qwen3-8B \
    --tp 1 \
    --max_model_len 4096 \
    --gpu_memory_utilization 0.65 \
    --cluster_model Qwen/Qwen3-4B \
    --n_samples 8 \
    --temperature 0.7 \
    --max_gen_tokens 1024 \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --batch_size 128

echo "=========================================="
echo "Done!"
echo "=========================================="
echo "Output: $OUTPUT"
echo "Count:  $(wc -l < $OUTPUT)"

echo ""
echo "SE distribution:"
python3 -c "
import json
path = '$OUTPUT'
with open(path) as f:
    data = [json.loads(l) for l in f if l.strip()]
ses = [d['se'] for d in data if d.get('se') is not None]
n = len(ses)
if n == 0:
    print('  No valid samples.')
else:
    avg = sum(ses)/n
    low  = sum(1 for s in ses if s < 0.1)
    mid  = sum(1 for s in ses if 0.1 <= s < 0.5)
    high = sum(1 for s in ses if s >= 0.5)
    print(f'  Total valid: {n} / {len(data)}')
    print(f'  Avg SE: {avg:.3f}  Avg weight: {1-avg:.3f}')
    print(f'  Confident  (SE<0.1):        {low}  ({low/n:.1%})')
    print(f'  Moderate   (0.1<=SE<0.5):   {mid}  ({mid/n:.1%})')
    print(f'  Uncertain  (SE>=0.5):        {high} ({high/n:.1%})')
"
