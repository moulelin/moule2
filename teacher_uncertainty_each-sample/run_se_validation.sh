#!/bin/bash
set -x

# ---- Activate environment ----
source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

# ---- Model cache ----
SCRATCH=/anvil/scratch/x-qlan1/moule
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache

# ---- Fix GLIBC ----
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

SCRIPT_DIR=/home/x-qlan1/code/moule2/teacher_uncertainty_each-sample

# ============================================================
# Step 1: Run Qwen3-8B on all questions (2x H100, TP=2)
# ============================================================

# echo "=========================================="
# echo "Step 1: Teacher inference (Qwen3-8B)"
# echo "=========================================="

# python3 "$SCRIPT_DIR/se_validation_infer.py" \
#     --model Qwen/Qwen3-8B \
#     --tp 2 \
#     --max_model_len 4096 \
#     --gpu_memory_utilization 0.85 \
#     --max_gen_tokens 2048 \
#     --input "$SCRIPT_DIR/dataset_output/solved_accepted.jsonl" \
#     --output "$SCRIPT_DIR/se_validation_results.jsonl" \
#     --batch_size 1024

# echo "=========================================="
# echo "Step 1 Done!"
# echo "=========================================="

# ============================================================
# Step 2: Generate plots + AUROC (CPU only)
# ============================================================

echo "=========================================="
echo "Step 2: Plotting SE vs accuracy + AUROC"
echo "=========================================="

python3 "$SCRIPT_DIR/se_validation_plot.py" \
    --input "$SCRIPT_DIR/se_validation_results.jsonl" \
    --output_dir "$SCRIPT_DIR/se_validation_plots" \
    --n_bins 10

echo "=========================================="
echo "All done! Plots saved to:"
echo "  $SCRIPT_DIR/se_validation_plots/"
echo "=========================================="
