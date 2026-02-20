#!/bin/bash
set -x

# ============================================================
# Baseline Evaluation: Qwen3 models on math competition benchmarks
# Models:   Qwen3-1.7B, Qwen3-4B, Qwen3-8B
# Datasets: AIME24, AIME25, HMMT25, AMO-Bench
# Modes:    pass@1 (greedy) + avg@16 (temp=1.2, Qwen3 recommended)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH=/anvil/scratch/x-qlan1/moule

MODELS=(
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
)

for MODEL in "${MODELS[@]}"; do
    bash "$SCRIPT_DIR/eval_aime24.sh" "$MODEL"
    bash "$SCRIPT_DIR/eval_aime25.sh" "$MODEL"
    bash "$SCRIPT_DIR/eval_hmmt25.sh" "$MODEL"
    bash "$SCRIPT_DIR/eval_amo_bench.sh" "$MODEL"
done

echo ""
echo "============================================================"
echo "All evaluations complete. Results in: $SCRATCH/eval_results"
echo "============================================================"
echo ""
python3 "$SCRIPT_DIR/generate_latex_table.py" \
    --results_dir "$SCRATCH/eval_results"
