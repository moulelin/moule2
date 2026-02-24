"""
CLI entry point for VtD (Verify-then-Distill) training with Ray.

Supports multi-node training across multiple GPU nodes.
Uses vLLM for fast on-policy generation.
"""

import argparse
from datetime import datetime

import ray
from ray.util.placement_group import placement_group

from openrlhf.trainer.ray import create_vllm_engines
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.trainer.ray.vtd_actor import VtDStudentActor, VtDTeacherActor
from openrlhf.trainer.vtd_trainer_ray import VtDRayTrainer
from openrlhf.utils import get_strategy


def train(args):
    if not ray.is_initialized():
        ray.init()  # Inherit runtime_env from ray job submit to avoid conflicts

    strategy = get_strategy(args)
    strategy.print(args)

    # ============ Placement Groups ============
    pg = None
    if args.colocate_all_models:
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.student_num_nodes * args.student_num_gpus_per_node)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    # ============ vLLM Engines ============
    vllm_engines = None
    if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
        # vLLM max_model_len must accommodate both training generation and eval generation.
        # Training: short (max_input_len + generate_max_len = ~3072)
        # Eval: long thinking mode (few-shot prompt ~2000 + max_tokens up to 38000)
        # Match standalone eval script: max_model_len = 40960
        eval_max_tokens = getattr(args, "eval_max_tokens", args.generate_max_len)
        max_len = max(args.max_input_len + args.generate_max_len, 4096 + eval_max_tokens)
        max_len = min(max_len, 40960)  # cap at model's supported context length
        if args.colocate_all_models:
            assert (
                args.student_num_nodes * args.student_num_gpus_per_node
                == args.vllm_num_engines * args.vllm_tensor_parallel_size
            ), (
                f"student GPUs ({args.student_num_nodes * args.student_num_gpus_per_node}) "
                f"must equal vllm_num_engines * vllm_tensor_parallel_size "
                f"({args.vllm_num_engines * args.vllm_tensor_parallel_size}) when colocate_all_models"
            )
        vllm_engines = create_vllm_engines(
            args.vllm_num_engines,
            args.vllm_tensor_parallel_size,
            args.pretrain,
            args.seed,
            args.full_determinism,
            args.enable_prefix_caching,
            args.enforce_eager,
            max_len,
            pg if args.colocate_all_models else None,
            args.vllm_gpu_memory_utilization,
            args.vllm_enable_sleep,
        )

    # ============ Student Model (trainable) ============
    student_model = RayActorGroup(
        args.student_num_nodes,
        args.student_num_gpus_per_node,
        VtDStudentActor,
        pg=pg,
        num_gpus_per_actor=0.2 if pg else 1,
    )

    # ============ Teacher Model (frozen, for distillation + SE sampling) ============
    pg_teacher = pg if args.colocate_all_models else None
    teacher_model = RayActorGroup(
        args.teacher_num_nodes,
        args.teacher_num_gpus_per_node,
        VtDTeacherActor,
        pg=pg_teacher,
        num_gpus_per_actor=0.2 if pg_teacher else 1,
    )

    # ============ VtD Trainer (Single Controller) ============
    vtd_trainer = VtDRayTrainer.remote(
        args.pretrain,
        strategy,
        student_model,
        teacher_model,
        vllm_engines,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    max_steps = ray.get(vtd_trainer.get_max_steps.remote())

    # ============ Init Models ============
    refs = []
    refs.extend(student_model.async_init_model_from_pretrained(strategy, args.pretrain, max_steps, vllm_engines))
    refs.extend(teacher_model.async_init_model_from_pretrained(strategy, args.teacher_model))
    ray.get(refs)

    # ============ Train ============
    ray.get(vtd_trainer.fit.remote())

    # ============ Save ============
    ray.get(student_model.async_save_model())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ============ Ray / Multi-Node ============
    parser.add_argument("--student_num_nodes", type=int, default=1)
    parser.add_argument("--student_num_gpus_per_node", type=int, default=4)
    parser.add_argument("--teacher_num_nodes", type=int, default=1)
    parser.add_argument("--teacher_num_gpus_per_node", type=int, default=4)
    parser.add_argument("--colocate_all_models", action="store_true", default=False,
                        help="Colocate all models (Student, Teacher, vLLM) on same GPUs. Use with --vllm_enable_sleep.")

    # ============ vLLM ============
    parser.add_argument("--vllm_num_engines", type=int, default=4)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_sync_backend", type=str, default="nccl")
    parser.add_argument("--vllm_sync_with_ray", action="store_true", default=False)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true", default=False)
    parser.add_argument("--vllm_enable_sleep", action="store_true", default=False)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.95)

    # ============ Checkpoints ============
    parser.add_argument("--save_path", type=str, default="/anvil/scratch/x-qlan1/moule/checkpoint/vtd")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--ckpt_path", type=str, default="/anvil/scratch/x-qlan1/moule/checkpoint/vtd/checkpoints")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--use_ds_universal_ckpt", action="store_true", default=False)

    # ============ DeepSpeed ============
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--zero_stage", type=int, default=3)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument("--param_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--zpg", type=int, default=1)
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--use_liger_kernel", action="store_true", default=False)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--deepspeed_enable_sleep", action="store_true", default=False)
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full_determinism", action="store_true", default=False)

    # ============ LoRA ============
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # ============ Models ============
    parser.add_argument("--pretrain", type=str, required=True, help="Student model")
    parser.add_argument("--teacher_model", type=str, required=True, help="Teacher model")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Normalize reward (not used in VtD)")
    parser.add_argument("--value_head_prefix", type=str, default="score", help="Value head prefix (not used in VtD)")

    # ============ Training ============
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=64)
    parser.add_argument("--micro_train_batch_size", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--actor_learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95))

    # ============ Generation ============
    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument("--generate_max_len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--n_samples_per_prompt", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=100000)

    # ============ VtD Hyperparameters ============
    parser.add_argument("--vtd_distill_alpha", type=float, default=5.0)
    parser.add_argument("--vtd_distill_tau", type=float, default=1.0,
                        help="Temperature for token-level divergence weighting in distillation loss")
    parser.add_argument("--se_n_samples", type=int, default=8,
                        help="Number of teacher responses per prompt for semantic entropy estimation")
    parser.add_argument("--se_cluster_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Small model for pairwise semantic equivalence judging in SE clustering")
    parser.add_argument("--se_weight_key", type=str, default=None,
                        help="Key in dataset for precomputed SE weight. When set, skips online SE sampling")
    parser.add_argument("--teacher_micro_batch_size", type=int, default=2)
    parser.add_argument("--distill_topk", type=int, default=0,
                        help="Top-K teacher logits for distillation. 0 = full logits, >0 = top-K (e.g. 512)")

    # ============ Dataset ============
    parser.add_argument("--prompt_data", type=str, required=True)
    parser.add_argument("--prompt_data_probs", type=str, default=None)
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--eval_dataset", type=str, default=None,
                        help="Comma-separated eval datasets, e.g. 'openai/gsm8k,MathArena/hmmt_feb_2025'")
    parser.add_argument("--eval_split", type=str, default="test",
                        help="Comma-separated splits matching eval_dataset, e.g. 'test,train'")
    parser.add_argument("--eval_input_key", type=str, default="question",
                        help="Comma-separated input keys matching eval_dataset, e.g. 'question,problem'")
    parser.add_argument("--eval_num_shots", type=int, default=4,
                        help="Number of few-shot examples for eval (independent of training num_shots)")
    parser.add_argument("--eval_max_tokens", type=int, default=4096,
                        help="Max generation tokens for eval (independent of generate_max_len)")
    parser.add_argument("--input_key", type=str, default="question")
    parser.add_argument("--label_key", type=str, default="answer")
    parser.add_argument("--output_key", type=str, default=None,
                        help="Key for reference thinking chain (CoT). When set, enables CoT-guided VtD: "
                             "distillation on reference CoT, contrastive with reference as chosen.")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--apply_chat_template", action="store_true", default=False)
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="Append <think> after assistant turn (Qwen3 thinking mode)")
    parser.add_argument("--packing_samples", action="store_true", default=False)
    parser.add_argument("--num_shots", type=int, default=4,
                        help="Number of few-shot CoT examples in prompt (0 for zero-shot)")

    # ============ Logging ============
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_vtd_ray")
    parser.add_argument("--wandb_run_name", type=str,
                        default="vtd_ray_%s" % datetime.now().strftime("%m%dT%H:%M"))
    parser.add_argument("--use_tensorboard", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="/anvil/scratch/x-qlan1/moule/logs/vtd",
                        help="Directory for local txt training logs")

    # ============ ModelScope ============
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.colocate_all_models:
        if args.vllm_enable_sleep is False and args.vllm_num_engines and args.vllm_num_engines > 0:
            print("[Warning] --colocate_all_models with vLLM requires --vllm_enable_sleep for memory sharing")

    if args.vllm_enable_sleep and not args.colocate_all_models:
        print("Set args.vllm_enable_sleep to False when args.colocate_all_models is disabled.")
        args.vllm_enable_sleep = False

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub
        patch_hub()

    train(args)
