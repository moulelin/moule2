#!/bin/bash
set -x

# ---- Activate environment ----
source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

# ---- Clean up previous Ray session ----
ray stop --force 2>/dev/null || true
sleep 2
rm -rf /tmp/ray/session_* 2>/dev/null || true

# ---- Paths ----
PROJECT_DIR=/home/x-qlan1/code/moule
OPENRLHF_DIR="$PROJECT_DIR/OpenRLHF"
SCRATCH=/anvil/scratch/x-qlan1/moule

# ---- Storage (all cache to scratch, not current dir) ----
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export WANDB_DIR=$SCRATCH/wandb
export WANDB_CACHE_DIR=$SCRATCH/wandb/.cache

# ---- Fix GLIBC for flash_attn ----
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export NCCL_SOCKET_IFNAME=ib1
# ---- vLLM uses sync LLM() engine (no VLLM_USE_V1 needed) ----

# ============================================================
# VtD Ray Training — 1 Node × 4 H100 GPUs
# Phase 1 (Generate):  vLLM×4 active (student/teacher/ref sleep to CPU)
# Phase 2 (Collect):   vLLM sleeps; Teacher×4 + Ref×4 wake for logit collection (多GPU并行)
# Phase 3 (Train):     Teacher/Ref sleep; Student×4 wake for ZeRO-1 distributed training
# Phase 4 (Sync):      vLLM wakes briefly for weight broadcast, then sleeps
# ============================================================

# ---- Ray ----
ray stop --force || true
sleep 3
ray start --head --num-gpus=4
sleep 15

# Wait for Ray dashboard/agent to be ready
for i in $(seq 1 30); do
    if ray status &>/dev/null; then
        echo "Ray is ready."
        break
    fi
    echo "Waiting for Ray agent... ($i/30)"
    sleep 2
done

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{\"working_dir\": \"$OPENRLHF_DIR\", \"excludes\": [\"/checkpoints/\", \"/.git/\", \"/logs/\"], \"env_vars\": {\"TOKENIZERS_PARALLELISM\": \"true\", \"LD_PRELOAD\": \"$CONDA_PREFIX/lib/libstdc++.so.6\", \"HF_HOME\": \"$SCRATCH/hf_cache\", \"HF_DATASETS_CACHE\": \"$SCRATCH/hf_cache/datasets\", \"TRANSFORMERS_CACHE\": \"$SCRATCH/hf_cache\", \"TORCH_HOME\": \"$SCRATCH/torch_cache\", \"WANDB_DIR\": \"$SCRATCH/wandb\", \"WANDB_CACHE_DIR\": \"$SCRATCH/wandb/.cache\", \"VLLM_TARGET_DEVICE\": \"cuda\"}}" \
   -- python3 -m openrlhf.cli.train_vtd_ray \
   --pretrain Qwen/Qwen3-1.7B \
   --teacher_model Qwen/Qwen3-8B \
   --student_num_nodes 1 \
   --student_num_gpus_per_node 4 \
   --teacher_num_nodes 1 \
   --teacher_num_gpus_per_node 4 \
   --ref_num_nodes 1 \
   --output_key answer \
   --ref_num_gpus_per_node 4 \
   --colocate_all_models \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.6 \
   --vllm_enable_sleep \
   --enforce_eager \
   --vllm_sync_backend nccl \
   --prompt_data openai/gsm8k \
   --prompt_split train \
   --input_key question \
   --label_key answer \
   --apply_chat_template \
   --enable_thinking \
   --num_shots 2 \
   --max_input_len 1024 \
   --generate_max_len 2048 \
   --max_samples 10000 \
   --n_samples_per_prompt 4 \
   --vtd_distill_alpha 5.0 \
   --vtd_contrast_beta 0.1 \
   --temperature 0.7 \
   --top_p 0.95 \
   --teacher_generate \
   --zero_stage 1 \
   --deepspeed_enable_sleep \
   --packing_samples \
   --param_dtype bf16 \
   --attn_implementation flash_attention_2 \
   --gradient_checkpointing \
   --micro_train_batch_size 2 \
   --train_batch_size 256 \
   --rollout_batch_size 1024 \
   --num_episodes 1 \
   --max_epochs 1 \
   --actor_learning_rate 3e-6 \
   --lr_warmup_ratio 0.05 \
   --lr_scheduler cosine_with_min_lr \
   --adam_betas 0.9 0.95 \
   --max_norm 1.0 \
   --save_path $SCRATCH/checkpoint/qwen3-1.7b-vtd-ray-gsm8k \
   --ckpt_path $SCRATCH/checkpoint/qwen3-1.7b-vtd-ray-gsm8k/checkpoints \
   --save_steps 3 \
   --eval_batch_size 1320 \
   --save_hf_ckpt \
   --logging_steps 1 \
   --eval_steps 1 \
   --eval_dataset openai/gsm8k \
   --eval_split test \
   --use_wandb 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 \
   --wandb_project vtd_reasoning \
   --wandb_run_name vtd_ray_qwen3_1.7b_gsm8k \
   --log_dir ./logs/vtd_ray_gsm8k \
   --seed 42

ray stop
