#!/bin/bash
set -x

# ---- Activate environment ----
source /anvil/scratch/x-qlan1/moule/train-env/bin/activate

# ---- Paths ----
PROJECT_DIR=/home/x-qlan1/code/moule2
OPENRLHF_DIR="$PROJECT_DIR/OpenRLHF"
SCRATCH=/anvil/scratch/x-qlan1/moule

# ---- Storage ----
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export WANDB_DIR=$SCRATCH/wandb
export WANDB_CACHE_DIR=$SCRATCH/wandb/.cache

# ---- Fix GLIBC for flash_attn ----
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export NCCL_SOCKET_IFNAME=ib1

# ============================================================
# VtD Ray Training — 2 Nodes × 4 H100 GPUs = 8 GPUs
# ============================================================

TRAIN_DATA=$PROJECT_DIR/teacher_uncertainty_each-sample/dataset_output/evolved_with_se.jsonl
GPUS_PER_NODE=4

# ---- Multi-node Ray cluster via SLURM ----
if [ -z "$SLURM_JOB_NODELIST" ]; then
    echo "ERROR: Must run under SLURM (use submit_2node.sh)"
    exit 1
fi

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -i | awk '{print $1}')
echo "Head node: $HEAD_NODE ($HEAD_ADDR), total nodes: ${#NODES[@]}"

# Stop old Ray on all nodes
for node in "${NODES[@]}"; do
    srun --nodes=1 --ntasks=1 -w "$node" ray stop --force 2>/dev/null &
done
wait
sleep 3

# Start Ray head
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    ray start --head --num-gpus=$GPUS_PER_NODE --port=6379 &
sleep 10

# Start Ray workers on remaining nodes
for node in "${NODES[@]:1}"; do
    echo "Starting Ray worker on $node ..."
    srun --nodes=1 --ntasks=1 -w "$node" \
        ray start --address="$HEAD_ADDR:6379" --num-gpus=$GPUS_PER_NODE &
    sleep 5
done

# Wait for all GPUs to join
EXPECTED=$((${#NODES[@]} * GPUS_PER_NODE))
for i in $(seq 1 60); do
    NUM_GPUS=$(ray status 2>/dev/null | grep -oP '\d+\.\d+ GPU' | head -1 | grep -oP '[\d.]+' || echo "0")
    if [ "$(echo "$NUM_GPUS >= $EXPECTED" | bc)" -eq 1 ]; then
        echo "Ray cluster ready: $NUM_GPUS GPUs across ${#NODES[@]} nodes."
        break
    fi
    echo "Waiting for Ray cluster... ($i/60, GPUs: $NUM_GPUS/$EXPECTED)"
    sleep 5
done
ray status

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{\"working_dir\": \"$OPENRLHF_DIR\", \"excludes\": [\"/checkpoints/\", \"/.git/\", \"/logs/\"], \"env_vars\": {\"TOKENIZERS_PARALLELISM\": \"true\", \"LD_PRELOAD\": \"$CONDA_PREFIX/lib/libstdc++.so.6\", \"HF_HOME\": \"$SCRATCH/hf_cache\", \"HF_DATASETS_CACHE\": \"$SCRATCH/hf_cache/datasets\", \"TRANSFORMERS_CACHE\": \"$SCRATCH/hf_cache\", \"TORCH_HOME\": \"$SCRATCH/torch_cache\", \"WANDB_DIR\": \"$SCRATCH/wandb\", \"WANDB_CACHE_DIR\": \"$SCRATCH/wandb/.cache\", \"VLLM_TARGET_DEVICE\": \"cuda\"}}" \
   -- python3 -m openrlhf.cli.train_vtd_ray \
   --pretrain Qwen/Qwen3-1.7B \
   --teacher_model Qwen/Qwen3-8B \
   --student_num_nodes 2 \
   --student_num_gpus_per_node 4 \
   --teacher_num_nodes 2 \
   --teacher_num_gpus_per_node 4 \
   --colocate_all_models \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.7 \
   --vllm_enable_sleep \
   --vllm_sync_backend nccl \
   --prompt_data $TRAIN_DATA \
   --prompt_split train \
   --input_key question \
   --se_weight_key se_weight \
   --apply_chat_template \
   --enable_thinking \
   --num_shots 0 \
   --max_input_len 1024 \
   --generate_max_len 3280 \
   --max_samples 50000 \
   --n_samples_per_prompt 4 \
   --vtd_distill_alpha 5.0 \
   --temperature 0.7 \
   --top_p 0.95 \
   --zero_stage 1 \
   --deepspeed_enable_sleep \
   --packing_samples \
   --param_dtype bf16 \
   --attn_implementation flash_attention_2 \
   --gradient_checkpointing \
   --micro_train_batch_size 2 \
   --train_batch_size 256 \
   --rollout_batch_size 256 \
   --num_episodes 1 \
   --max_epochs 1 \
   --actor_learning_rate 3e-6 \
   --lr_warmup_ratio 0.05 \
   --lr_scheduler cosine_with_min_lr \
   --adam_betas 0.9 0.95 \
   --max_norm 1.0 \
   --save_path $SCRATCH/checkpoint/qwen3-1.7b-vtd-ray-evolved-2node \
   --ckpt_path $SCRATCH/checkpoint/qwen3-1.7b-vtd-ray-evolved-2node/checkpoints \
   --save_steps 3 \
   --eval_batch_size 1320 \
   --save_hf_ckpt \
   --logging_steps 1 \
   --eval_steps 1 \
   --eval_dataset "openai/gsm8k" \
   --eval_split "test" \
   --eval_input_key "question" \
   --eval_num_shots 4 \
   --eval_max_tokens 4096 \
   --use_wandb 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 \
   --wandb_project vtd_reasoning \
   --wandb_run_name vtd_2node_qwen3_1.7b \
   --log_dir ./logs/vtd_ray_2node \
   --seed 42

ray stop
