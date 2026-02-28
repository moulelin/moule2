#!/bin/bash
set -x

source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

SCRATCH=/anvil/scratch/x-qlan1/moule2
SCRIPT_DIR=/home/x-qlan1/code/moule2/scripts/eval_base/py

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

OUTPUT_DIR=$SCRATCH/eval_results/base
mkdir -p $OUTPUT_DIR

run_model() {
    local GPUS=$1
    local MODEL=$2
    echo "============================================================"
    echo "[$GPUS] Evaluating: $MODEL on AMO-Bench"
    echo "============================================================"

    # ---- pass@1 (greedy) ----
    CUDA_VISIBLE_DEVICES=$GPUS python3 "$SCRIPT_DIR/eval_amo_bench.py" \
        --model "$MODEL" \
        --mode greedy \
        --max_tokens 38000 \
        --max_model_len 40960 \
        --tp 2 \
        --gpu_memory_utilization 0.9 \
        --output_dir "$OUTPUT_DIR"

    # ---- avg@16 ----
    CUDA_VISIBLE_DEVICES=$GPUS python3 "$SCRIPT_DIR/eval_amo_bench.py" \
        --model "$MODEL" \
        --mode average \
        --n_samples 16 \
        --temperature 1.2 \
        --top_p 0.95 \
        --max_tokens 38000 \
        --max_model_len 40960 \
        --tp 2 \
        --gpu_memory_utilization 0.9 \
        --output_dir "$OUTPUT_DIR"
}

# 2 models in parallel (tp=2 each, 2 GPUs per model)
run_model 0,1 Qwen/Qwen3-1.7B &
run_model 2,3 Qwen/Qwen3-4B &
wait

# 3rd model uses GPUs 0,1
run_model 0,1 Qwen/Qwen3-8B
wait
echo "All AMO-Bench evaluations completed."
