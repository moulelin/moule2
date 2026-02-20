#!/bin/bash
set -x

source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

SCRATCH=/anvil/scratch/x-qlan1/moule
SCRIPT_DIR=/home/x-qlan1/code/moule/scripts/eval

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

OUTPUT_DIR=$SCRATCH/eval_results
mkdir -p $OUTPUT_DIR

MODELS=(
    "Qwen/Qwen3-1.7B"
)

for MODEL in "${MODELS[@]}"; do
    echo "============================================================"
    echo "Evaluating: $MODEL on gsm8k"
    echo "============================================================"

    # ---- pass@1 (greedy) ----
    python3 "$SCRIPT_DIR/eval_gsm8k.py" \
        --model "$MODEL" \
        --mode greedy \
        --num_shots 2 \
        --max_tokens 2048 \
        --max_model_len 3072 \
        --tp 2 \
        --gpu_memory_utilization 0.9 \
        --output_dir "$OUTPUT_DIR"

    # ---- avg@16 ----
    # python3 "$SCRIPT_DIR/eval_gsm8k.py" \
    #     --model "$MODEL" \
    #     --mode average \
    #     --n_samples 16 \
    #     --num_shots 4 \
    #     --temperature 1.2 \
    #     --top_p 0.95 \
    #     --max_tokens 4096 \
    #     --max_model_len 8192 \
    #     --tp 2 \
    #     --gpu_memory_utilization 0.9 \
    #     --output_dir "$OUTPUT_DIR"
done
