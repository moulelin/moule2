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

SURE_MATH_TEST=/home/x-qlan1/code/moule2/dataset/merged/math_merged_en.jsonl

run_model() {
    local GPU=$1
    local MODEL=$2
    echo "============================================================"
    echo "[$GPU] Evaluating: $MODEL on SURE-Math"
    echo "============================================================"

    # ---- pass@1 (greedy) ----
    CUDA_VISIBLE_DEVICES=$GPU python3 "$SCRIPT_DIR/eval_sure_math.py" \
        --model "$MODEL" \
        --input "$SURE_MATH_TEST" \
        --mode greedy \
        --max_tokens 4096 \
        --max_model_len 8192 \
        --tp 1 \
        --gpu_memory_utilization 0.9 \
        --output_dir "$OUTPUT_DIR"

    # # ---- avg@16 ----
    # CUDA_VISIBLE_DEVICES=$GPU python3 "$SCRIPT_DIR/eval_sure_math.py" \
    #     --model "$MODEL" \
    #     --input "$SURE_MATH_TEST" \
    #     --mode average \
    #     --n_samples 16 \
    #     --temperature 1.2 \
    #     --top_p 0.95 \
    #     --max_tokens 4096 \
    #     --max_model_len 8192 \
    #     --tp 1 \
    #     --gpu_memory_utilization 0.9 \
    #     --output_dir "$OUTPUT_DIR"
}

# Sequential execution
run_model 0 Qwen/Qwen3-1.7B
run_model 0 Qwen/Qwen3-4B
# run_model 0 Qwen/Qwen3-8B
echo "All SURE-Math evaluations completed."
