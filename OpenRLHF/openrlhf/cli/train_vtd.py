import argparse
import math
import os
from datetime import datetime

from transformers.trainer import get_scheduler

from openrlhf.datasets import VtDPromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.models import Actor
from openrlhf.trainer.vtd_trainer import VtDTrainer
from openrlhf.utils import get_strategy, get_tokenizer


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # ---- Student model (trainable) ----
    model = Actor(
        args.pretrain,
        attn_implementation=args.attn_implementation,
        param_dtype=args.param_dtype,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        use_liger_kernel=args.use_liger_kernel,
    )

    # ---- Teacher model (frozen, for distillation) ----
    teacher_model = Actor(
        args.teacher_model,
        attn_implementation=args.attn_implementation,
        param_dtype=args.param_dtype,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=args.teacher_offload),
    )
    if args.teacher_offload:
        teacher_model._offload = True

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    strategy.print(model)

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    # ---- Dataset: prompts with ground truth answers ----
    train_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        dataset_split=args.dataset_split,
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))

    train_dataset = VtDPromptDataset(
        train_data,
        tokenizer,
        strategy,
        input_template=args.input_template,
        max_length=args.max_input_len,
    )

    prompt_dataloader = strategy.setup_dataloader(
        train_dataset, args.micro_train_batch_size, True, True, train_dataset.collate_fn
    )

    # Eval dataset (optional)
    eval_dataloader = None
    if getattr(args, "eval_dataset", None):
        eval_data = blending_datasets(
            args.eval_dataset,
            None,
            strategy,
            dataset_split=args.eval_split,
        )
        eval_dataset = VtDPromptDataset(
            eval_data,
            tokenizer,
            strategy,
            input_template=args.input_template,
            max_length=args.max_input_len,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset, args.micro_train_batch_size, True, False, eval_dataset.collate_fn
        )

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # prepare models with strategy (DeepSpeed wrapping)
    ((model, optim, scheduler), teacher_model) = strategy.prepare(
        (model, optim, scheduler), teacher_model
    )

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model.model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    # ---- Configure VtD Trainer ----
    trainer = VtDTrainer(
        model=model,
        teacher_model=teacher_model,
        strategy=strategy,
        optim=optim,
        tokenizer=tokenizer,
        prompt_dataloader=prompt_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        max_epochs=args.max_epochs,
        num_samples_per_prompt=args.num_samples_per_prompt,
        max_gen_len=args.max_gen_len,
        max_input_len=args.max_input_len,
        vtd_distill_alpha=args.vtd_distill_alpha,
        vtd_distill_tau=args.vtd_distill_tau,
        se_n_samples=args.se_n_samples,
        generation_temperature=args.generation_temperature,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # save final model
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ============ Checkpoints ============
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_vtd")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--use_ds_universal_ckpt", action="store_true", default=False)

    # ============ DeepSpeed ============
    parser.add_argument("--micro_train_batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Global training batch size")
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full_determinism", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--zero_stage", type=int, default=3)
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
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=1)

    # ============ LoRA ============
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # ============ Models ============
    parser.add_argument("--pretrain", type=str, required=True, help="Student model path")
    parser.add_argument("--teacher_model", type=str, required=True, help="Teacher model path")
    parser.add_argument("--teacher_offload", action="store_true", default=False)

    # ============ VtD Hyperparameters ============
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95))

    parser.add_argument("--num_samples_per_prompt", type=int, default=4,
                        help="K: number of on-policy rollouts per prompt")
    parser.add_argument("--max_gen_len", type=int, default=1024,
                        help="Max generation length for on-policy sampling")
    parser.add_argument("--max_input_len", type=int, default=512,
                        help="Max input/prompt length")
    parser.add_argument("--vtd_distill_alpha", type=float, default=5.0,
                        help="Temperature for entropy-gap weighting in distillation")
    parser.add_argument("--vtd_distill_tau", type=float, default=1.0,
                        help="Temperature for token-level divergence weighting in distillation loss")
    parser.add_argument("--se_n_samples", type=int, default=8,
                        help="Number of teacher responses per prompt for semantic entropy estimation")
    parser.add_argument("--generation_temperature", type=float, default=0.7,
                        help="Temperature for on-policy sampling")

    # ============ Dataset ============
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_probs", type=str, default=None)
    parser.add_argument("--eval_dataset", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=100000)

    parser.add_argument("--input_key", type=str, default="question", help="JSON key for input/question")
    parser.add_argument("--output_key", type=str, default="answer", help="JSON key for output (unused)")
    parser.add_argument("--label_key", type=str, default="answer", help="JSON key for ground truth answer")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--apply_chat_template", action="store_true", default=False)

    # ============ Logging ============
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_vtd")
    parser.add_argument("--wandb_run_name", type=str,
                        default="vtd_%s" % datetime.now().strftime("%m%dT%H:%M"))
    parser.add_argument("--use_tensorboard", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="./logs/vtd",
                        help="Directory for local txt training logs")

    # ============ ModelScope ============
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub
        patch_hub()

    train(args)
