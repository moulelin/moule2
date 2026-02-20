import os
import re
from abc import ABC
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models import VtDDistillLoss, VtDContrastLoss
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


class VtDTrainer(ABC):
    """
    Trainer for Verify-then-Distill (VtD).

    Implements on-policy sampling with verification-guided dual objectives:
    - Correct rollouts: confidence-weighted distillation from teacher
    - Incorrect rollouts: DPO-style contrastive learning

    Args:
        model: Student model to train
        teacher_model: Teacher model for distillation guidance
        ref_model: Reference model for contrastive loss (frozen copy of initial student)
        strategy: Training strategy (DeepSpeed)
        optim: Optimizer
        tokenizer: Tokenizer
        prompt_dataloader: DataLoader yielding (prompts, labels)
        num_samples_per_prompt: K - number of rollouts per prompt
        max_gen_len: Maximum generation length
        vtd_distill_alpha: Temperature for entropy-gap weighting
        vtd_contrast_beta: Beta coefficient for DPO-style contrastive loss
        teacher_generate: Whether teacher generates chosen responses when D+ is empty
    """

    def __init__(
        self,
        model,
        teacher_model,
        ref_model,
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
        vtd_contrast_beta: float = 0.1,
        teacher_generate: bool = True,
        generation_temperature: float = 0.7,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.model = model
        self.teacher_model = teacher_model
        self.ref_model = ref_model
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
        self.teacher_generate = teacher_generate
        self.generation_temperature = generation_temperature
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt

        # Loss functions
        self.distill_loss_fn = VtDDistillLoss(alpha=vtd_distill_alpha, tau=vtd_distill_tau)
        self.contrast_loss_fn = VtDContrastLoss(beta=vtd_contrast_beta)

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
                f.write(f"Config: K={num_samples_per_prompt}, alpha={vtd_distill_alpha}, beta={vtd_contrast_beta}\n")
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

    def _compute_sequence_logps(self, model, prompt: str, response: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Compute per-token log probs for a prompt+response and return (logits, log_probs_sum, prompt_len).

        Returns:
            logits: (1, seq_len, vocab_size)
            avg_logp: scalar tensor - average log prob over response tokens
            prompt_len: length of prompt in tokens
        """
        # Tokenize prompt to get prompt length
        prompt_tokens = self.tokenizer(
            prompt, return_tensors="pt", max_length=self.max_input_len,
            truncation=True, add_special_tokens=False,
        )
        prompt_len = prompt_tokens["input_ids"].shape[1]

        # Tokenize full sequence
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

        output = model(input_ids, attention_mask=attention_mask, return_output=True)
        logits = output["logits"]  # (1, seq_len, vocab)

        # Compute per-token log probs
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1, dtype=torch.float32)
        token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (1, seq_len-1)

        # Mask: only response tokens (after prompt)
        response_mask = torch.zeros_like(token_logps)
        response_start = prompt_len - 1  # shifted by 1 due to shift_logits
        response_mask[:, response_start:] = 1.0
        # Also mask padding
        shift_attention = attention_mask[:, 1:]
        response_mask = response_mask * shift_attention

        num_response_tokens = response_mask.sum()
        if num_response_tokens > 0:
            avg_logp = (token_logps * response_mask).sum() / num_response_tokens
        else:
            avg_logp = torch.tensor(0.0, device=input_ids.device)

        return logits, avg_logp, prompt_len

    def _compute_distill_loss_for_batch(
        self, prompts: list[str], correct_responses: list[list[str]]
    ) -> torch.Tensor:
        """Compute confidence-weighted distillation loss for correct rollouts."""
        total_loss = torch.tensor(0.0, device=torch.cuda.current_device())
        count = 0

        self.model.train()
        self.teacher_model.eval()

        for prompt, responses in zip(prompts, correct_responses):
            for response in responses:
                # Tokenize full sequence
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

                # Get prompt length for loss mask
                prompt_tokens = self.tokenizer(
                    prompt, return_tensors="pt", max_length=self.max_input_len,
                    truncation=True, add_special_tokens=False,
                )
                prompt_len = prompt_tokens["input_ids"].shape[1]

                # Student forward
                student_output = self.model(input_ids, attention_mask=attention_mask, return_output=True)
                student_logits = student_output["logits"][:, :-1, :]  # (1, seq_len-1, vocab)

                # Teacher forward
                with torch.no_grad():
                    teacher_output = self.teacher_model(input_ids, attention_mask=attention_mask, return_output=True)
                    teacher_logits = teacher_output["logits"][:, :-1, :]

                # Loss mask: only response tokens
                seq_len = student_logits.shape[1]
                loss_mask = torch.zeros(1, seq_len, device=input_ids.device)
                loss_mask[:, prompt_len - 1:] = 1.0
                shift_attention = attention_mask[:, 1:]
                loss_mask = loss_mask * shift_attention[:, :seq_len]

                loss = self.distill_loss_fn(student_logits, teacher_logits, loss_mask)
                total_loss = total_loss + loss
                count += 1

        if count > 0:
            return total_loss / count
        return total_loss

    def _compute_contrast_loss_for_batch(
        self,
        prompts: list[str],
        chosen_responses: list[str],
        rejected_responses: list[list[str]],
    ) -> tuple[torch.Tensor, float, float]:
        """Compute contrastive loss for incorrect rollouts.

        For each prompt, contrasts a chosen response against each rejected response.
        """
        all_chosen_logps = []
        all_rejected_logps = []
        all_ref_chosen_logps = []
        all_ref_rejected_logps = []

        self.model.train()

        for prompt, chosen, rejecteds in zip(prompts, chosen_responses, rejected_responses):
            # Compute chosen log probs
            _, chosen_logp, _ = self._compute_sequence_logps(self.model, prompt, chosen)
            with torch.no_grad():
                _, ref_chosen_logp, _ = self._compute_sequence_logps(self.ref_model, prompt, chosen)

            for rejected in rejecteds:
                _, rejected_logp, _ = self._compute_sequence_logps(self.model, prompt, rejected)
                with torch.no_grad():
                    _, ref_rejected_logp, _ = self._compute_sequence_logps(self.ref_model, prompt, rejected)

                all_chosen_logps.append(chosen_logp)
                all_rejected_logps.append(rejected_logp)
                all_ref_chosen_logps.append(ref_chosen_logp)
                all_ref_rejected_logps.append(ref_rejected_logp)

        if not all_chosen_logps:
            zero = torch.tensor(0.0, device=torch.cuda.current_device())
            return zero, 0.0, 0.0

        chosen_logps = torch.stack(all_chosen_logps)
        rejected_logps = torch.stack(all_rejected_logps)
        ref_chosen_logps = torch.stack(all_ref_chosen_logps)
        ref_rejected_logps = torch.stack(all_ref_rejected_logps)

        loss, chosen_rewards, rejected_rewards = self.contrast_loss_fn(
            chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps
        )
        return loss, chosen_rewards.mean().item(), rejected_rewards.mean().item()

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

                # Classify into correct (D+) and incorrect (D-)
                # Per-prompt: lists of correct and incorrect responses
                batch_correct = []  # list of lists
                batch_incorrect = []  # list of lists
                batch_has_correct = []  # whether this prompt has any correct response

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
                    batch_has_correct.append(len(correct) > 0)
                    total_correct += len(correct)
                    total_incorrect += len(incorrect)

                # ============ Step 2: Compute losses ============
                distill_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                contrast_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                contrast_chosen_reward = 0.0
                contrast_rejected_reward = 0.0

                # --- Distillation loss on correct rollouts ---
                distill_prompts = []
                distill_correct_responses = []
                for i, prompt in enumerate(prompts):
                    if batch_correct[i]:
                        distill_prompts.append(prompt)
                        distill_correct_responses.append(batch_correct[i])

                if distill_prompts:
                    distill_loss = self._compute_distill_loss_for_batch(
                        distill_prompts, distill_correct_responses
                    )

                # --- Contrastive loss on incorrect rollouts ---
                contrast_prompts = []
                contrast_chosen = []
                contrast_rejected = []

                for i, prompt in enumerate(prompts):
                    if not batch_incorrect[i]:
                        continue  # No incorrect rollouts for this prompt

                    # Determine chosen response
                    if batch_correct[i]:
                        # Use student's own correct response
                        chosen = batch_correct[i][0]
                    elif self.teacher_generate:
                        # Generate from teacher
                        teacher_responses = self._generate_responses(
                            self.teacher_model, [prompt], num_samples=1
                        )
                        chosen = teacher_responses[0][0]
                    else:
                        continue  # Skip if no chosen available

                    contrast_prompts.append(prompt)
                    contrast_chosen.append(chosen)
                    contrast_rejected.append(batch_incorrect[i])

                if contrast_prompts:
                    contrast_loss, contrast_chosen_reward, contrast_rejected_reward = \
                        self._compute_contrast_loss_for_batch(
                            contrast_prompts, contrast_chosen, contrast_rejected
                        )

                # ============ Step 3: Adaptive weighting ============
                total = total_correct + total_incorrect
                if total > 0:
                    lambda_1 = total_correct / total
                    lambda_2 = total_incorrect / total
                else:
                    lambda_1 = 0.5
                    lambda_2 = 0.5

                loss = lambda_1 * distill_loss + lambda_2 * contrast_loss

                # ============ Step 4: Backward + Update ============
                self.model.train()
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # ============ Logging ============
                accuracy = total_correct / (total if total > 0 else 1)
                logs_dict = {
                    "loss": loss.item(),
                    "distill_loss": distill_loss.item(),
                    "contrast_loss": contrast_loss.item(),
                    "lambda_distill": lambda_1,
                    "lambda_contrast": lambda_2,
                    "accuracy": accuracy,
                    "num_correct": total_correct,
                    "num_incorrect": total_incorrect,
                    "contrast_chosen_reward": contrast_chosen_reward,
                    "contrast_rejected_reward": contrast_rejected_reward,
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
