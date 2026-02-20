import math
import os
import re
from abc import ABC
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models import VtDDistillLoss
from openrlhf.models.utils import log_probs_from_logits, masked_mean
from openrlhf.utils.distributed_sampler import DistributedSampler


def _extract_last_boxed(text: str) -> str | None:
    """Extract content from the last \\boxed{...} using brace-depth counting.

    Handles nested braces like \\boxed{\\frac{1}{2}} correctly.
    """
    keyword = "\\boxed{"
    idx = text.rfind(keyword)
    if idx == -1:
        return None
    start = idx + len(keyword)
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i - 1]
    return None


def extract_answer(text: str) -> str:
    """Extract the final numerical answer from a model response.

    Supports common formats:
    - \\boxed{answer}
    - The answer is: answer
    - #### answer (GSM8K format)
    - Final answer at end of text
    """
    # Try \\boxed{...} with brace-depth counting for nested braces
    boxed = _extract_last_boxed(text)
    if boxed is not None:
        return boxed.strip()

    # Try #### format (GSM8K)
    hash_match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if hash_match:
        return hash_match.group(1).strip()

    # Try "the answer is" format
    answer_match = re.search(r"(?:the answer is|answer is)[:\s]*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip().rstrip(".")

    # Fallback: return last line stripped
    lines = text.strip().split("\n")
    return lines[-1].strip() if lines else ""


def normalize_answer(ans: str) -> str:
    """Normalize answer string for comparison."""
    ans = ans.strip().lower()
    # Remove $ and other common formatting
    ans = ans.replace("$", "").replace(",", "").replace("%", "").strip()
    # Try to parse as number
    try:
        return str(float(ans))
    except ValueError:
        return ans


def verify_answer(response: str, ground_truth: str) -> bool:
    """Check if the response contains the correct answer."""
    predicted = extract_answer(response)
    return normalize_answer(predicted) == normalize_answer(ground_truth)


from openrlhf.trainer.vtd_trainer_ray import SemanticCluster, semantic_entropy_to_weight


class VtDTrainer(ABC):
    """
    Trainer for Verify-then-Distill (VtD) with Semantic Entropy weighting.

    Implements on-policy sampling with semantic-entropy-weighted distillation:
    - Teacher generates N responses per prompt → compute semantic entropy
    - All student rollouts → distillation weighted by teacher confidence (1 - SE)

    Args:
        model: Student model to train
        teacher_model: Teacher model for distillation guidance + SE sampling
        strategy: Training strategy (DeepSpeed)
        optim: Optimizer
        tokenizer: Tokenizer
        prompt_dataloader: DataLoader yielding (prompts, labels)
        num_samples_per_prompt: K - number of rollouts per prompt
        max_gen_len: Maximum generation length
        vtd_distill_alpha: Temperature for entropy-gap weighting
        se_n_samples: Number of teacher responses for semantic entropy estimation
    """

    def __init__(
        self,
        model,
        teacher_model,
        strategy,
        optim: Optimizer,
        tokenizer,
        prompt_dataloader,
        eval_dataloader=None,
        scheduler=None,
        max_norm: float = 1.0,
        max_epochs: int = 1,
        num_samples_per_prompt: int = 4,
        max_gen_len: int = 1024,
        max_input_len: int = 512,
        vtd_distill_alpha: float = 5.0,
        vtd_distill_tau: float = 1.0,
        se_n_samples: int = 8,
        se_cluster_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        generation_temperature: float = 0.7,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.model = model
        self.teacher_model = teacher_model
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.prompt_dataloader = prompt_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.max_norm = max_norm
        self.epochs = max_epochs
        self.args = strategy.args

        self.num_samples_per_prompt = num_samples_per_prompt
        self.max_gen_len = max_gen_len
        self.max_input_len = max_input_len
        self.se_n_samples = se_n_samples
        self.semantic_cluster = SemanticCluster(model_name=se_cluster_model, device="cpu")
        self.generation_temperature = generation_temperature
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt

        # Loss functions
        self.distill_loss_fn = VtDDistillLoss(alpha=vtd_distill_alpha, tau=vtd_distill_tau)

        # Logging
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)

        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

        # Local txt log
        self.log_file = None
        log_dir = getattr(strategy.args, "log_dir", None)
        if log_dir and self.strategy.is_rank_0():
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"vtd_train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            self.log_file = log_path
            with open(self.log_file, "w") as f:
                f.write(f"VtD Training Log - {datetime.now().isoformat()}\n")
                f.write(f"Config: K={num_samples_per_prompt}, alpha={vtd_distill_alpha}, SE_N={se_n_samples}\n")
                f.write("=" * 80 + "\n")

    def _log_to_file(self, global_step, logs_dict, prefix="train"):
        if self.log_file is None:
            return
        with open(self.log_file, "a") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] {prefix} step={global_step}"
            for k, v in sorted(logs_dict.items()):
                if isinstance(v, float):
                    line += f" | {k}={v:.6f}"
                else:
                    line += f" | {k}={v}"
            f.write(line + "\n")

    @torch.no_grad()
    def _generate_responses(self, model, prompts: list[str], num_samples: int = 1) -> list[list[str]]:
        """Generate responses from a model for a batch of prompts.

        Returns: list of lists, each inner list has `num_samples` response strings.
        """
        model.eval()
        all_responses = []

        for prompt in prompts:
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_input_len,
                truncation=True,
                add_special_tokens=False,
            ).input_ids.to(torch.cuda.current_device())

            prompt_responses = []
            for _ in range(num_samples):
                with torch.no_grad():
                    output_ids = model.model.generate(
                        input_ids,
                        max_new_tokens=self.max_gen_len,
                        temperature=self.generation_temperature,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                    )
                # Decode only the generated part
                generated_ids = output_ids[0, input_ids.shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                prompt_responses.append(response)
            all_responses.append(prompt_responses)

        return all_responses

    def _compute_distill_loss_for_batch(
        self, prompts: list[str], responses_per_prompt: list[list[str]],
        se_weights: list[float] = None,
    ) -> torch.Tensor:
        """Compute SE-weighted distillation loss for all rollouts.

        Args:
            prompts: list of prompt strings
            responses_per_prompt: list of lists of response strings per prompt
            se_weights: per-prompt semantic entropy weights (1 = confident, 0 = uncertain)
        """
        total_loss = torch.tensor(0.0, device=torch.cuda.current_device())
        count = 0

        self.model.train()
        self.teacher_model.eval()

        for p_idx, (prompt, responses) in enumerate(zip(prompts, responses_per_prompt)):
            se_w = se_weights[p_idx] if se_weights is not None else 1.0
            for response in responses:
                full_text = prompt + response
                if not full_text.endswith(self.tokenizer.eos_token):
                    full_text += self.tokenizer.eos_token
                tokens = self.tokenizer(
                    full_text, return_tensors="pt",
                    max_length=self.max_input_len + self.max_gen_len,
                    truncation=True, add_special_tokens=False,
                )
                input_ids = tokens["input_ids"].to(torch.cuda.current_device())
                attention_mask = tokens["attention_mask"].to(torch.cuda.current_device())

                prompt_tokens = self.tokenizer(
                    prompt, return_tensors="pt", max_length=self.max_input_len,
                    truncation=True, add_special_tokens=False,
                )
                prompt_len = prompt_tokens["input_ids"].shape[1]

                student_output = self.model(input_ids, attention_mask=attention_mask, return_output=True)
                student_logits = student_output["logits"][:, :-1, :]

                with torch.no_grad():
                    teacher_output = self.teacher_model(input_ids, attention_mask=attention_mask, return_output=True)
                    teacher_logits = teacher_output["logits"][:, :-1, :]

                seq_len = student_logits.shape[1]
                loss_mask = torch.zeros(1, seq_len, device=input_ids.device)
                loss_mask[:, prompt_len - 1:] = 1.0
                shift_attention = attention_mask[:, 1:]
                loss_mask = loss_mask * shift_attention[:, :seq_len]

                loss = self.distill_loss_fn(student_logits, teacher_logits, loss_mask)
                total_loss = total_loss + se_w * loss
                count += 1

        if count > 0:
            return total_loss / count
        return total_loss

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch if num_update_steps_per_epoch else float("inf")
        if args.save_steps == -1:
            args.save_steps = float("inf")

        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // (num_update_steps_per_epoch or 1)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )

        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.prompt_dataloader.sampler, DistributedSampler):
                self.prompt_dataloader.sampler.set_epoch(epoch)

            step_bar = tqdm(
                range(len(self.prompt_dataloader)),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            for prompts, labels, _ref_outputs, _raw_inputs in self.prompt_dataloader:
                # ============ Step 1: On-policy sampling + Verification ============
                self.model.eval()
                student_responses = self._generate_responses(
                    self.model, prompts, num_samples=self.num_samples_per_prompt
                )

                batch_correct = []
                batch_incorrect = []
                total_correct = 0
                total_incorrect = 0

                for i, (prompt, responses) in enumerate(zip(prompts, student_responses)):
                    correct = []
                    incorrect = []
                    for resp in responses:
                        if verify_answer(resp, labels[i]):
                            correct.append(resp)
                        else:
                            incorrect.append(resp)
                    batch_correct.append(correct)
                    batch_incorrect.append(incorrect)
                    total_correct += len(correct)
                    total_incorrect += len(incorrect)

                # ============ Step 2: Teacher SE sampling ============
                teacher_se_responses = self._generate_responses(
                    self.teacher_model, prompts, num_samples=self.se_n_samples
                )
                se_weights = []
                for responses in teacher_se_responses:
                    se = self.semantic_cluster.compute_entropy(responses)
                    se_weights.append(semantic_entropy_to_weight(se))

                # ============ Step 3: SE-weighted distillation on all rollouts ============
                distill_loss = torch.tensor(0.0, device=torch.cuda.current_device())

                all_prompts = []
                all_responses = []
                all_se_weights = []
                for i, prompt in enumerate(prompts):
                    responses = batch_correct[i] + batch_incorrect[i]
                    if responses:
                        all_prompts.append(prompt)
                        all_responses.append(responses)
                        all_se_weights.append(se_weights[i])

                if all_prompts:
                    distill_loss = self._compute_distill_loss_for_batch(
                        all_prompts, all_responses, all_se_weights
                    )

                loss = distill_loss

                # ============ Step 4: Backward + Update ============
                self.model.train()
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # ============ Logging ============
                total = total_correct + total_incorrect
                accuracy = total_correct / (total if total > 0 else 1)
                avg_se_weight = sum(se_weights) / len(se_weights) if se_weights else 1.0
                logs_dict = {
                    "loss": loss.item(),
                    "distill_loss": distill_loss.item(),
                    "accuracy": accuracy,
                    "num_correct": total_correct,
                    "num_incorrect": total_incorrect,
                    "avg_se_weight": avg_se_weight,
                    "avg_semantic_entropy": 1.0 - avg_se_weight,
                }
                if self.scheduler:
                    logs_dict["lr"] = self.scheduler.get_last_lr()[0]

                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # Save logs/checkpoints
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
            self._log_to_file(global_step, logs_dict, prefix="train")

        if global_step % args.eval_steps == 0 and self.eval_dataloader is not None:
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
                )
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)

    @torch.no_grad()
    def evaluate(self, eval_dataloader, steps=0):
        """Evaluate: compute accuracy on eval set by sampling and verifying."""
        self.model.eval()
        total_correct = 0
        total_samples = 0

        step_bar = tqdm(
            range(len(eval_dataloader)),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )

        for prompts, labels, _ref_outputs, _raw_inputs in eval_dataloader:
            responses = self._generate_responses(self.model, prompts, num_samples=1)
            for i, (prompt, resps) in enumerate(zip(prompts, responses)):
                if verify_answer(resps[0], labels[i]):
                    total_correct += 1
                total_samples += 1

            step_bar.update()

        accuracy = total_correct / max(total_samples, 1)
        logs = {"eval_accuracy": accuracy, "eval_total": total_samples}
        logs = self.strategy.all_reduce(logs)
        step_bar.set_postfix(logs)

        if self.strategy.is_rank_0():
            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, steps)
            self._log_to_file(steps, logs, prefix="eval")

        self.model.train()
