#!/bin/bash
# Debug script: test VtD prompt output
# Uses blending_datasets + VtDPromptDataset, same pipeline as real training
# No Ray, no GPU, no model

export HF_HOME=/anvil/scratch/x-qlan1/moule/cache/huggingface
export TRANSFORMERS_CACHE=/anvil/scratch/x-qlan1/moule/cache/huggingface

cd "$(dirname "$0")"

# ============ GSM8K + apply_chat_template ============
python OpenRLHF/openrlhf/cli/debug_vtd_prompt.py \
    --pretrain Qwen/Qwen2.5-0.5B-Instruct \
    --prompt_data openai/gsm8k \
    --prompt_split train \
    --input_key question \
    --label_key answer \
    --output_key answer \
    --apply_chat_template \
    --max_samples 3
