"""
VtD (Verify-then-Distill) Ray Actor for Student Model Training.

This actor manages the student model, handles VtD training steps
(semantic-entropy-weighted distillation on all rollouts),
and syncs weights to vLLM engines.
"""

import math
import os
import socket
from typing import Dict, List, Optional, Union

import deepspeed
import ray
import torch
import torch.distributed
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm
from transformers.trainer import get_scheduler

from openrlhf.models import Actor, VtDDistillLoss
from openrlhf.models.utils import masked_mean
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import offload_deepspeed_states, reload_deepspeed_states
from openrlhf.utils.distributed_util import stateless_init_process_group, torch_dist_barrier_and_cuda_sync
from openrlhf.utils.logging_utils import init_logger

from .launcher import BaseModelActor

logger = init_logger(__name__)


@ray.remote(num_gpus=1)
class VtDStudentActor(BaseModelActor):
    """Ray actor wrapping the trainable student model for VtD."""

    def init_model_from_pretrained(
        self, strategy: DeepspeedStrategy, pretrain, max_steps=None, vllm_engines=None
    ):
        args = strategy.args
        self.save_hf_ckpt = args.save_hf_ckpt
        self.disable_ds_ckpt = args.disable_ds_ckpt
        self.vllm_engines = vllm_engines
        self.max_steps = max_steps

        if getattr(args, "vllm_num_engines", 0) > 0:
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            attn_implementation=strategy.args.attn_implementation,
            param_dtype=strategy.args.param_dtype,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(actor)

        self.tokenizer = get_tokenizer(
            pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
        )
        actor_scheduler = get_scheduler(
            args.lr_scheduler,
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # Prepare with DeepSpeed
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler), is_rlhf=True,
        )

        # Loss functions
        self.distill_loss_fn = VtDDistillLoss(
            alpha=getattr(args, "vtd_distill_alpha", 5.0),
            tau=getattr(args, "vtd_distill_tau", 1.0),
        )
        self.max_gen_len = getattr(args, "max_gen_len", 1024)
        self.max_input_len = getattr(args, "max_input_len", 512)

        # Load checkpoint
        self.checkpoint_states = {}
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            strategy.print(f"Loading checkpoint: {ckpt_path}")
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.checkpoint_states = states

        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)

        # Init vLLM weight sync group
        self._init_vllm_sync()

    def _init_vllm_sync(self):
        """Initialize weight sync group for vLLM engines.

        When colocate_all_models: use CUDA IPC (shared memory, no NCCL needed).
        Otherwise: use NCCL process group for cross-GPU broadcast.
        """
        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = False
        if backend == "nccl" and getattr(self.strategy.args, "colocate_all_models", False):
            self.use_cuda_ipc = True

        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines = self.strategy.args.vllm_num_engines
            vllm_tensor_parallel_size = self.strategy.args.vllm_tensor_parallel_size
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
            group_name = "openrlhf_vtd"

            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            refs = [
                engine.init_process_group.remote(
                    master_address, master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size, group_name, backend="nccl", use_ray=use_ray,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            if use_ray:
                import ray.util.collective as collective
                collective.init_collective_group(
                    world_size=world_size, rank=0, backend=backend, group_name=group_name
                )
                self._model_update_group = group_name
            else:
                self._model_update_group = stateless_init_process_group(
                    master_address, master_port, 0, world_size, torch.cuda.current_device()
                )
            ray.get(refs)

        torch_dist_barrier_and_cuda_sync()

    @torch.no_grad()
    def generate(self, prompts: list[str], num_samples: int, temperature: float = 0.7,
                 top_p: float = 0.95, max_new_tokens: int = 512) -> list[list[str]]:
        """Generate responses using the student model directly (no vLLM)."""
        self.actor.model.eval()
        # self.actor.model = DeepSpeed engine, .module = HuggingFace model
        hf_model = self.actor.model.module
        all_responses = []

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                    max_length=self.max_input_len).to(torch.cuda.current_device())
            prompt_len = inputs["input_ids"].shape[1]
            responses = []
            for _ in range(num_samples):
                outputs = hf_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                response_ids = outputs[0][prompt_len:]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                responses.append(response_text)
            all_responses.append(responses)

        return all_responses

    def _pad_and_batch(self, tensors: list, pad_value=0):
        """Pad a list of (1, seq_len, ...) tensors to the same seq_len and stack into a batch."""
        max_len = max(t.shape[1] for t in tensors)
        padded = []
        for t in tensors:
            pad_size = max_len - t.shape[1]
            if pad_size > 0:
                pad_dims = [0] * (2 * (t.dim() - 2)) + [0, pad_size]
                t = F.pad(t, pad_dims, value=pad_value)
            padded.append(t)
        return torch.cat(padded, dim=0)

    def vtd_train(
        self,
        distill_items_flat: list,
    ) -> Dict[str, float]:
        """Train from flat collected buffers with rank-based data sharding.

        Each actor selects its data shard from the flat lists using rank::world_size stride,
        enabling true data parallelism across all student GPUs. Items padded with None ensure
        all ranks make identical backward+step calls → DeepSpeed allreduce stays in sync.

        Distillation items are processed in micro_bs batches. Each item includes a per-sample
        semantic entropy weight from the teacher's N-response sampling. None items trigger a
        zero-loss backward to advance DeepSpeed's internal accumulation counter.

        Args:
            distill_items_flat: flat list of (input_ids, attn, loss_mask, topk_vals, topk_ids, se_weight)
                or None, padded to multiple of world_size * micro_bs.
                se_weight is a float in [0, 1] representing teacher confidence (1 - normalized SE).
        """
        import time as _time
        self.actor.train()
        device = torch.cuda.current_device()
        micro_bs = self.strategy.micro_train_batch_size
        accum_steps = self.strategy.accumulated_gradient

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        # Each rank takes its stride slice (round-robin data sharding)
        distill_shard = distill_items_flat[rank::world_size]

        n_real_distill = sum(1 for x in distill_shard if x is not None)
        logger.info(
            f"[vtd_train rank={rank}/{world_size}] "
            f"distill={n_real_distill}/{len(distill_shard)} "
            f"micro_bs={micro_bs} accum_steps={accum_steps}"
        )

        distill_loss_val = 0.0
        se_weight_sum = 0.0
        n_distill_steps = 0
        total_backward_steps = 0
        n_weight_updates = 0

        t0 = _time.time()

        # ---- Distillation: process shard in micro_bs chunks ----
        distill_total = len(distill_shard)
        distill_num_chunks = (distill_total + micro_bs - 1) // micro_bs
        for chunk_idx, start in enumerate(range(0, distill_total, micro_bs)):
            end = min(start + micro_bs, distill_total)
            chunk = distill_shard[start:end]
            real = [x for x in chunk if x is not None]

            if real:
                seqs  = [x[0] for x in real]
                attns = [x[1] for x in real]
                masks = [x[2] for x in real]
                tvals = [x[3] for x in real]
                tids  = [x[4] for x in real]
                se_ws = [x[5] for x in real]  # per-sample SE weights

                seq_batch   = self._pad_and_batch(seqs,  pad_value=0).to(device)
                attn_batch  = self._pad_and_batch(attns, pad_value=0).to(device)
                mask_batch  = self._pad_and_batch(masks, pad_value=0).to(device)
                tvals_batch = self._pad_and_batch(tvals, pad_value=0).to(device)
                tids_batch  = self._pad_and_batch(tids,  pad_value=0).to(device)
                se_w_batch  = torch.tensor(se_ws, dtype=torch.float32, device=device)

                student_output = self.actor(seq_batch, attention_mask=attn_batch, return_output=True)
                student_logits = student_output["logits"][:, :-1, :]
                loss = self.distill_loss_fn(
                    student_logits,
                    tvals_batch[:, :-1, :],
                    tids_batch[:, :-1, :],
                    mask_batch[:, 1:],
                    se_weights=se_w_batch,
                )
                self.strategy.backward(loss, self.actor, self.actor_optim)
                distill_loss_val += loss.item()
                se_weight_sum += se_w_batch.mean().item()
                n_distill_steps += 1
            else:
                # Padding chunk: zero-loss backward to keep allreduce step counter in sync
                zero_loss = next(self.actor.parameters()).sum() * 0.0
                self.strategy.backward(zero_loss, self.actor, self.actor_optim)

            self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
            total_backward_steps += 1
            if total_backward_steps % accum_steps == 0:
                n_weight_updates += 1
            logger.info(f"[蒸馏训练 rank={rank}] 微批次 {chunk_idx + 1}/{distill_num_chunks} 完成 ({end}/{distill_total} 条样本, loss={distill_loss_val / max(n_distill_steps, 1):.4f})")

        # ---- Flush remaining accumulated gradients ----
        remainder = total_backward_steps % accum_steps
        if remainder > 0:
            pad_steps = accum_steps - remainder
            for _ in range(pad_steps):
                zero_loss = next(self.actor.parameters()).sum() * 0.0
                self.strategy.backward(zero_loss, self.actor, self.actor_optim)
                self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
                total_backward_steps += 1
            n_weight_updates += 1
            logger.info(f"[训练 rank={rank}] 填充 {pad_steps} 步以对齐梯度累积边界")

        total_time = _time.time() - t0
        logger.info(
            f"[训练 rank={rank}] 完成: 共 {total_backward_steps} 步, "
            f"{n_weight_updates} 次权重更新 (累积={accum_steps}), "
            f"lr={self.actor_scheduler.get_last_lr()[0]:.2e}, 耗时 {total_time:.1f}s"
        )

        avg_distill = distill_loss_val / n_distill_steps if n_distill_steps > 0 else 0.0
        avg_se_weight = se_weight_sum / n_distill_steps if n_distill_steps > 0 else 0.0
        return {
            "vtd_loss":         avg_distill,
            "distill_loss":     avg_distill,
            "avg_se_weight":    avg_se_weight,
            "actor_lr":         self.actor_scheduler.get_last_lr()[0],
            "n_distill_steps":  n_distill_steps,
            "n_weight_updates": n_weight_updates,
        }

    def compute_logps(self, input_ids, attention_mask, prompt_len):
        """Compute average log prob over response tokens."""
        device = torch.cuda.current_device()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        self.actor.eval()
        with torch.no_grad():
            output = self.actor(input_ids, attention_mask=attention_mask, return_output=True)
            logits = output["logits"]

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1, dtype=torch.float32)
        token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        response_mask = torch.zeros_like(token_logps)
        response_mask[:, prompt_len - 1:] = 1.0
        shift_attention = attention_mask[:, 1:]
        response_mask = response_mask * shift_attention

        num_tokens = response_mask.sum()
        if num_tokens > 0:
            avg_logp = (token_logps * response_mask).sum() / num_tokens
        else:
            avg_logp = torch.tensor(0.0, device=device)

        return avg_logp.cpu()

    def get_logits(self, input_ids, attention_mask):
        """Get full logits for distillation."""
        device = torch.cuda.current_device()
        self.actor.eval()
        with torch.no_grad():
            output = self.actor(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                return_output=True,
            )
        return output["logits"].cpu()

    def broadcast_to_vllm(self):
        """Broadcast trained weights to vLLM engines.

        Matches PPO's _broadcast_to_vllm pattern from STAR-main:
        - NCCL broadcast for non-colocated setups
        - CUDA IPC for colocate_all_models
        - Supports ds_tensor_parallel_size > 1
        - Handles prefix cache reset
        """
        from .utils import get_physical_gpu_id

        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        model = self.actor.model.module
        count, num_params = 0, len(list(model.named_parameters()))

        def _broadcast_param(param, count, num_params):
            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            if torch.distributed.get_rank() == 0:
                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(
                        name, dtype=param.dtype, shape=shape,
                        empty_cache=count == num_params
                    )
                    for engine in self.vllm_engines
                ]
                if use_ray:
                    import ray.util.collective as collective
                    collective.broadcast(param.data, 0, group_name=self._model_update_group)
                else:
                    self._model_update_group.broadcast(
                        param.data, src=0, stream=torch.cuda.current_stream()
                    )
                ray.get(refs)

        def _handle_cuda_ipc(param, count, num_params):
            from torch.multiprocessing.reductions import reduce_tensor

            weight = param.data.clone()
            ipc_handle = reduce_tensor(weight)

            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handle_list = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

            if torch.distributed.get_rank() == 0:
                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)

                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight_cuda_ipc.remote(
                        name, dtype=param.dtype, shape=shape,
                        ipc_handles=ipc_handles,
                        empty_cache=count == num_params,
                    )
                    for engine in self.vllm_engines
                ]
                ray.get(refs)
            torch_dist_barrier_and_cuda_sync()

        for name, param in model.named_parameters():
            count += 1

            # Temporarily clear ds_active_sub_modules to avoid assertion error
            # when tied weights (e.g. embed_tokens & lm_head) share the same param.
            saved_sub_modules = None
            if self.strategy.args.zero_stage == 3 and hasattr(param, "ds_active_sub_modules"):
                saved_sub_modules = param.ds_active_sub_modules
                param.ds_active_sub_modules = set()

            if not self.use_cuda_ipc:
                # NCCL broadcast path
                if self.strategy.args.ds_tensor_parallel_size > 1:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True):
                        _broadcast_param(param, count, num_params)
                else:
                    with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                        _broadcast_param(param, count, num_params)
            else:
                # CUDA IPC path (colocate_all_models)
                if self.strategy.args.ds_tensor_parallel_size > 1:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True):
                        _handle_cuda_ipc(param, count, num_params)
                else:
                    with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                        _handle_cuda_ipc(param, count, num_params)

            # Restore ds_active_sub_modules
            if saved_sub_modules is not None:
                param.ds_active_sub_modules = saved_sub_modules

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()

    def save_model(self):
        self.strategy.save_model(self.actor, self.tokenizer, self.strategy.args.save_path)

    def save_checkpoint(self, tag, client_states):
        args = self.strategy.args
        if not self.disable_ds_ckpt:
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, "_actor"),
                tag, args.max_ckpt_num, args.max_ckpt_mem, client_states,
            )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(self.actor, self.tokenizer, save_path)
        torch_dist_barrier_and_cuda_sync()

    def get_checkpoint_states(self):
        return self.checkpoint_states

    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)


@ray.remote(num_gpus=1)
class VtDTeacherActor(BaseModelActor):
    """Ray actor wrapping the frozen teacher model for VtD distillation."""

    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        model = Actor(
            pretrain,
            attn_implementation=strategy.args.attn_implementation,
            param_dtype=strategy.args.param_dtype,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(f"Teacher model: {model}")

        if strategy.args.ref_reward_offload:
            model._offload = True

        # Fix DeepSpeed batch assertion: teacher is eval-only, batch params don't matter
        # Original student params (e.g. train_batch_size=1024, micro=20) may not divide
        # evenly with teacher world_size, causing assertion failure.
        import torch.distributed as dist
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        orig_micro_bs = strategy.micro_train_batch_size
        orig_train_bs = strategy.train_batch_size
        strategy.micro_train_batch_size = 1
        strategy.train_batch_size = world_size  # satisfies: world_size == 1 * 1 * world_size

        self.model = self.strategy.prepare(model, is_rlhf=True)

        # Restore original values
        strategy.micro_train_batch_size = orig_micro_bs
        strategy.train_batch_size = orig_train_bs

        self.model.eval()

        self.tokenizer = get_tokenizer(
            pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_len = getattr(strategy.args, "max_input_len", 512)
        self.max_gen_len = getattr(strategy.args, "generate_max_len", 1024)

    @torch.no_grad()
    def generate(self, prompt: str, temperature: float = 0.7, top_p: float = 0.95,
                 max_new_tokens: int = None) -> str:
        """Generate a single response from the teacher model (for DPO chosen fallback)."""
        if max_new_tokens is None:
            max_new_tokens = self.max_gen_len
        device = torch.cuda.current_device()
        # DeepSpeed wraps: self.model -> .module (Actor) -> .model (HF PreTrainedModel)
        actor_obj = self.model.module if hasattr(self.model, "module") else self.model
        hf_model = actor_obj.model if hasattr(actor_obj, "model") else actor_obj
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=self.max_input_len).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        outputs = hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        response_ids = outputs[0][prompt_len:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)

    @torch.no_grad()
    def batch_generate(self, prompts: list[str], num_samples: int = 8,
                       temperature: float = 0.7, top_p: float = 0.95,
                       max_new_tokens: int = None) -> list[list[str]]:
        """Generate multiple responses per prompt for semantic entropy estimation.

        Args:
            prompts: list of prompt strings
            num_samples: N responses per prompt
            temperature: sampling temperature
            top_p: nucleus sampling threshold
            max_new_tokens: max generation length

        Returns:
            list of lists, each inner list has num_samples response strings.
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_gen_len
        device = torch.cuda.current_device()
        actor_obj = self.model.module if hasattr(self.model, "module") else self.model
        hf_model = actor_obj.model if hasattr(actor_obj, "model") else actor_obj

        all_responses = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                    max_length=self.max_input_len).to(device)
            prompt_len = inputs["input_ids"].shape[1]
            responses = []
            for _ in range(num_samples):
                outputs = hf_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                response_ids = outputs[0][prompt_len:]
                responses.append(self.tokenizer.decode(response_ids, skip_special_tokens=True))
            all_responses.append(responses)
        return all_responses

    def get_logits(self, input_ids, attention_mask, top_k: int = 512):
        """Get teacher top-K logits for distillation (memory-efficient)."""
        device = torch.cuda.current_device()
        with torch.no_grad():
            output = self.model(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                return_output=True,
            )
            logits = output["logits"]  # (1, seq_len, vocab_size)
            # Keep only top-K logits, set rest to -inf
            topk_vals, topk_ids = logits.topk(top_k, dim=-1)
        return topk_vals.cpu(), topk_ids.cpu()

    def batch_get_logits(self, input_ids_list, attention_mask_list, top_k=512, micro_batch_size=4):
        """Batch get teacher top-K logits with micro-batching to prevent OOM."""
        device = torch.cuda.current_device()
        total = len(input_ids_list)
        num_batches = (total + micro_batch_size - 1) // micro_batch_size
        logger.info(f"[教师前向] 共 {total} 条序列, micro_batch_size={micro_batch_size}, 共 {num_batches} 个微批次")
        results = []

        for batch_idx, start in enumerate(range(0, total, micro_batch_size)):
            end = min(start + micro_batch_size, total)
            batch_ids = input_ids_list[start:end]
            batch_masks = attention_mask_list[start:end]

            # Pad to same length within micro-batch
            max_len = max(ids.shape[1] for ids in batch_ids)
            bs = len(batch_ids)
            padded_ids = torch.zeros(bs, max_len, dtype=batch_ids[0].dtype)
            padded_mask = torch.zeros(bs, max_len, dtype=batch_masks[0].dtype)
            for i, (ids, mask) in enumerate(zip(batch_ids, batch_masks)):
                seq_len = ids.shape[1]
                padded_ids[i, :seq_len] = ids[0]
                padded_mask[i, :seq_len] = mask[0]

            with torch.no_grad():
                output = self.model(
                    padded_ids.to(device),
                    attention_mask=padded_mask.to(device),
                    return_output=True,
                )
                logits = output["logits"]
                topk_vals, topk_ids = logits.topk(top_k, dim=-1)

            # Split back, trim padding, move to CPU
            for i, ids in enumerate(batch_ids):
                seq_len = ids.shape[1]
                results.append((
                    topk_vals[i : i + 1, :seq_len, :].cpu(),
                    topk_ids[i : i + 1, :seq_len, :].cpu(),
                ))

            logger.info(f"[教师前向] 微批次 {batch_idx + 1}/{num_batches} 完成 ({end}/{total} 条序列)")

        return results

    def compute_logps(self, input_ids, attention_mask, prompt_len):
        """Compute average log prob over response tokens."""
        device = torch.cuda.current_device()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask=attention_mask, return_output=True)
            logits = output["logits"]

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1, dtype=torch.float32)
        token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        response_mask = torch.zeros_like(token_logps)
        response_mask[:, prompt_len - 1:] = 1.0
        shift_attention = attention_mask[:, 1:]
        response_mask = response_mask * shift_attention

        num_tokens = response_mask.sum()
        if num_tokens > 0:
            avg_logp = (token_logps * response_mask).sum() / num_tokens
        else:
            avg_logp = torch.tensor(0.0, device=device)

        return avg_logp.cpu()

    def offload_states(self):
        """Move HF model weights to CPU to free GPU memory (Phase 1/3 sleep)."""
        # self.model is DeepSpeed engine; navigate to the underlying HF model
        actor_obj = self.model.module if hasattr(self.model, "module") else self.model
        hf_model = actor_obj.model if hasattr(actor_obj, "model") else actor_obj
        hf_model.to("cpu")
        torch.cuda.empty_cache()

    def reload_states(self):
        """Reload HF model weights back to GPU (Phase 2 wake)."""
        device = torch.cuda.current_device()
        actor_obj = self.model.module if hasattr(self.model, "module") else self.model
        hf_model = actor_obj.model if hasattr(actor_obj, "model") else actor_obj
        hf_model.to(device)
