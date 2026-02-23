#!/bin/bash
set -x

source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

SCRATCH=/anvil/scratch/x-qlan1/moule2
SCRIPT_DIR=/home/x-qlan1/code/moule2/scripts/eval

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

OUTPUT_DIR=$SCRATCH/eval_results
mkdir -p $OUTPUT_DIR

MODELS=(
    "/anvil/scratch/x-qlan1/moule/checkpoint/qwen3-1.7b-vtd-ray-evolved/checkpoints/global_step21_hf"
)

for MODEL in "${MODELS[@]}"; do
    echo "============================================================"
    echo "Evaluating: $MODEL on aime25"
    echo "============================================================"

    # ---- pass@1 (greedy) ----
    # python3 "$SCRIPT_DIR/eval_aime25.py" \
    #     --model "$MODEL" \
    #     --mode greedy \
    #     --max_tokens 38000 \
    #     --max_model_len 40960 \
    #     --tp 2 \
    #     --gpu_memory_utilization 0.9 \
    #     --output_dir "$OUTPUT_DIR"

    # # ---- avg@16 ----
    python3 "$SCRIPT_DIR/eval_aime25.py" \
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

    
done
