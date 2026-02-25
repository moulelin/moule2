#!/bin/bash
set -x

source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

SCRATCH=/anvil/scratch/x-qlan1/moule
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export CUDA_VISIBLE_DEVICES=0,1

SCRIPT_DIR=/home/x-qlan1/code/moule2/teacher_uncertainty_each-sample/pdf_extract

# Step 1: Marker 提取 PDF -> Markdown
echo "===== Step 1: PDF -> Markdown ====="
python3 "$SCRIPT_DIR/pdf_to_markdown.py" \
    --input "$SCRIPT_DIR/math.pdf" \
    --output_dir "$SCRIPT_DIR" \
    --force_ocr

# 找到 Marker 输出的 md 文件
MD_FILE=$(find "$SCRIPT_DIR" -name "*.md" -newer "$SCRIPT_DIR/math.pdf" | head -1)
if [ -z "$MD_FILE" ]; then
    echo "错误: 未找到 Marker 输出的 md 文件"
    exit 1
fi
echo "Marker 输出: $MD_FILE"

# Step 2: LLM 提取 Q&A
echo "===== Step 2: Markdown -> CSV ====="
python3 "$SCRIPT_DIR/extract_qa_llm.py" \
    --input "$MD_FILE" \
    --output "$SCRIPT_DIR/extracted_qa.csv" \
    --model Qwen/Qwen2.5-32B-Instruct \
    --tp 2 \
    --max_model_len 8192 \
    --gpu_memory_utilization 0.9 \
    --max_tokens 4096 \
    --chunk_size 6000 \
    --verbose
