"""
VtD Ray Trainer — Single controller orchestrating multi-node VtD training.

Architecture:
  - vLLM engines: fast on-policy generation from student
  - Student (VtDStudentActor): trainable, VtD loss optimization
  - Teacher (VtDTeacherActor): frozen, provides logits for distillation + SE sampling

VtD Training Logic (Semantic Entropy weighted):
  For each prompt:
  1. Student generates K responses, verified against label
  2. Teacher generates N responses for semantic entropy estimation
  3. Semantic entropy measures teacher uncertainty per prompt
  4. All student responses → distillation with teacher logits, weighted by (1 - normalized SE)
     - Low SE (teacher confident) → high distillation weight
     - High SE (teacher uncertain) → low distillation weight
"""

import json
import math
import os
import re
import regex
import time
import multiprocessing
from math import isclose
from copy import deepcopy

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy
from datetime import datetime, timedelta

import ray
import torch
from tqdm import tqdm
try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None

from openrlhf.datasets import VtDPromptDataset
from openrlhf.datasets.vtd_dataset import build_teacher_system_prompt
from openrlhf.datasets.utils import blending_datasets
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import TensorboardLogger, WandbLogger, init_logger
from openrlhf.utils.utils import get_tokenizer

logger = init_logger(__name__)


# ============ Answer Verification Utilities ============

# ============ Math Answer Utilities (from Qwen2.5-Math) ============

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        new_str += "{" + a + "}" + b + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except:
        return string


def _fix_sqrt(string):
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def strip_string(string):
    string = str(string).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("\\{", "{").replace("\\}", "}")
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")
    string = string.replace("\\%", "").replace("\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if (
        string.startswith("{") and string.endswith("}") and string.isalnum()
        or string.startswith("(") and string.endswith(")") and string.isalnum()
        or string.startswith("[") and string.endswith("]") and string.isalnum()
    ):
        string = string[1:-1]
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")
    string = string.replace("and", "").replace("\\mathbf", "")
    string = re.sub(r"\\mbox{.*?}", "", string)
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    return string


def _extract_answer_value(raw):
    """Extract answer value from raw text after 'the answer is' / 'final answer is'."""
    m = re.match(r"\$?(-?\d[\d,]*\.?\d*)", raw)
    if m:
        return m.group(1).replace(",", "")
    parts = re.split(r"(?<=[^a-zA-Z])\s+(?=[A-Z][a-z])", raw)
    return parts[0].strip()


def extract_answer(pred_str, use_last_number=True):
    # Strip thinking block: closed <think>...</think> or unclosed <think>... (truncated)
    pred_str = re.sub(r"<think>.*?</think>", "", pred_str, flags=re.DOTALL).strip()
    pred_str = re.sub(r"<think>.*", "", pred_str, flags=re.DOTALL).strip()
    pred_str = pred_str.replace("\u043a\u0438", "")
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "final answer is $" in pred_str and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "the answer is" in pred_str:
        raw = pred_str.split("the answer is")[-1].strip().split("\n")[0].strip()
        raw = re.sub(r"^[\s:]+", "", raw)
        pred = _extract_answer_value(raw)
    elif "final answer is" in pred_str:
        raw = pred_str.split("final answer is")[-1].strip().split("\n")[0].strip()
        raw = re.sub(r"^[\s:]+", "", raw)
        pred = _extract_answer_value(raw)
    else:
        if use_last_number:
            pattern = r"-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred)
    return pred


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num):
    return parse_digits(num) is not None


def numeric_equal(prediction: float, reference: float):
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s
    a = _parse(a)
    b = _parse(b)
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass
    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass
    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass
    return False


def math_equal(prediction, reference, include_percentage=True, is_close=True):
    if prediction is None or reference is None:
        return False
    if str(prediction).strip().lower() == str(reference).strip().lower():
        return True
    try:
        if is_digit(prediction) and is_digit(reference):
            prediction_d = parse_digits(prediction)
            reference_d = parse_digits(reference)
            if include_percentage:
                gt_result = [reference_d / 100, reference_d, reference_d * 100]
            else:
                gt_result = [reference_d]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction_d, item):
                            return True
                    else:
                        if item == prediction_d:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass
    if not prediction and prediction not in [0, False]:
        return False
    reference = str(reference).strip()
    prediction = str(prediction).strip()
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")
    ) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close) for i in range(len(pred_parts))):
                return True
    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif prediction.count("=") == 1 and len(prediction.split("=")[0].strip()) <= 2 and "=" not in reference:
        if math_equal(prediction.split("=")[1], reference, include_percentage, is_close):
            return True
    elif reference.count("=") == 1 and len(reference.split("=")[0].strip()) <= 2 and "=" not in prediction:
        if math_equal(prediction, reference.split("=")[1], include_percentage, is_close):
            return True
    if symbolic_equal(prediction, reference):
        return True
    return False


def verify_answer(response: str, ground_truth: str) -> bool:
    """Extract answer from response and compare with ground truth using Qwen2.5-Math grading."""
    predicted = extract_answer(response)
    gt_stripped = strip_string(ground_truth)
    return math_equal(predicted, gt_stripped)


# ============ Semantic Entropy ============

def compute_semantic_entropy(responses: list[str], n_samples: int = None) -> float:
    """Compute semantic entropy from teacher's N responses.

    Groups responses by semantic equivalence (same extracted answer after normalization),
    then computes entropy over the cluster distribution.

    Args:
        responses: list of N response strings from the teacher
        n_samples: total number of samples (for normalization). If None, uses len(responses).

    Returns:
        Normalized semantic entropy in [0, 1].
        0 = all responses agree (teacher confident)
        1 = maximum disagreement (teacher uncertain)
    """
    if not responses:
        return 1.0  # no responses = maximum uncertainty

    N = n_samples or len(responses)
    if N <= 1:
        return 0.0

    # Extract and normalize answers, group into semantic clusters
    clusters = {}
    for resp in responses:
        answer = extract_answer(resp)
        normalized = strip_string(answer) if answer else "__EMPTY__"
        # Group semantically equivalent answers
        found_cluster = False
        for key in list(clusters.keys()):
            if math_equal(normalized, key):
                clusters[key] += 1
                found_cluster = True
                break
        if not found_cluster:
            clusters[normalized] = 1

    # Compute entropy: H = -Σ p_c * log(p_c)
    entropy = 0.0
    for count in clusters.values():
        p = count / N
        if p > 0:
            entropy -= p * math.log(p)

    # Normalize by max possible entropy log(N)
    max_entropy = math.log(N)
    if max_entropy > 0:
        return entropy / max_entropy
    return 0.0


def semantic_entropy_to_weight(se: float) -> float:
    """Convert normalized semantic entropy to a distillation weight.

    weight = 1 - se
    - Teacher confident (se ≈ 0) → weight ≈ 1 (full distillation)
    - Teacher uncertain (se ≈ 1) → weight ≈ 0 (skip distillation)
    """
    return 1.0 - se


# ============ Dataset Preparation ============

def prepare_vtd_datasets(strategy, tokenizer):
    args = strategy.args
    train_data = blending_datasets(
        args.prompt_data, args.prompt_data_probs, strategy,
        args.seed, max_count=args.max_samples, dataset_split=args.prompt_split,
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    train_dataset = VtDPromptDataset(
        train_data, tokenizer, strategy,
        input_template=args.input_template,
        max_length=getattr(args, "max_input_len", 512),
    )
    rollout_batch_size = getattr(args, "rollout_batch_size", 64)
    train_dataloader = strategy.setup_dataloader(
        train_dataset, rollout_batch_size, True, True, train_dataset.collate_fn
    )

    eval_dataloader = None
    if getattr(args, "eval_dataset", None):
        eval_data = blending_datasets(
            args.eval_dataset, None, strategy, dataset_split=args.eval_split,
        )
        eval_dataset = VtDPromptDataset(
            eval_data, tokenizer, strategy,
            input_template=args.input_template,
            max_length=getattr(args, "max_input_len", 1024),
            is_eval=True,
        )
        eval_batch_size = getattr(args, "eval_batch_size", 512)
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset, eval_batch_size, True, False, eval_dataset.collate_fn,
            drop_last=False
        )

    n_samples = getattr(args, "n_samples_per_prompt", 4)
    # All samples go through distillation (no separate contrast path).
    # Each micro-batch of micro_bs samples = 1 backward call per rank.
    total_samples = len(train_dataset) * n_samples * args.num_episodes * args.max_epochs
    micro_bs = getattr(args, "micro_train_batch_size", 2)
    world_size = args.student_num_nodes * args.student_num_gpus_per_node
    accum_steps = args.train_batch_size // micro_bs // world_size
    max_steps = total_samples // world_size // micro_bs // accum_steps
    return train_dataloader, eval_dataloader, max_steps


# ============ Main VtD Ray Trainer ============

@ray.remote(num_cpus=1)  # CPU-only controller, no GPU needed
class VtDRayTrainer:
    """
    Single controller for VtD training across multiple nodes.

    Orchestrates:
    1. On-policy generation via vLLM
    2. Answer verification (label matching)
    3. Teacher N-response sampling for semantic entropy estimation
    4. Distillation on all student responses (teacher logits), weighted by SE
    5. Student model VtD training step
    6. Weight sync back to vLLM
    """

    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        student_model_group: RayActorGroup,
        teacher_model_group: RayActorGroup,
        vllm_engines,
        **generate_kwargs,
    ):
        args = strategy.args
        self.strategy = strategy
        self.args = args

        if args.eval_steps == -1:
            args.eval_steps = float("inf")
        if args.save_steps == -1:
            args.save_steps = float("inf")

        self.student_model_group = student_model_group
        self.teacher_model_group = teacher_model_group
        self.vllm_engines = vllm_engines
        self.generate_kwargs = generate_kwargs

        # Whether we have reference CoT for guided distillation
        self.use_reference_cot = bool(getattr(args, "output_key", None))

        # Semantic entropy config
        self.se_n_samples = getattr(args, "se_n_samples", 8)

        # Tokenizer
        self.tokenizer = get_tokenizer(
            pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Datasets
        self.prompts_dataloader, self.eval_dataloader, self.max_steps = prepare_vtd_datasets(
            strategy, self.tokenizer
        )

        # VtD config
        self.n_samples_per_prompt = getattr(args, "n_samples_per_prompt", 4)
        self.max_gen_len = getattr(args, "generate_max_len", 1024)
        self.max_input_len = getattr(args, "max_input_len", 512)

        # Logging
        self.wandb_logger = WandbLogger(args) if args.use_wandb else None
        self.tensorboard_logger = TensorboardLogger(args) if args.use_tensorboard else None

        # Local txt log — write to project dir (always accessible)
        self.project_dir = "/home/x-qlan1/code/moule"
        self.log_file = os.path.join(self.project_dir, "vtd_metrics.txt")
        with open(self.log_file, "w") as f:
            f.write(f"VtD Training Log - {datetime.now().isoformat()}\n")
            f.write(f"Student: {pretrain}, Teacher: {args.teacher_model}\n")
            f.write(f"CoT-guided: {self.use_reference_cot}\n")
            f.write(f"Config: K={self.n_samples_per_prompt}, alpha={args.vtd_distill_alpha}, SE_N={self.se_n_samples}\n")
            f.write("=" * 80 + "\n")
        logger.info(f"Metrics log file: {self.log_file}")

    def get_max_steps(self):
        return self.max_steps

    def _generate(self, prompts: list[str], labels: list[str], greedy: bool = False) -> list[list[str]]:
        """Generate responses per prompt. Uses vLLM if available, else student model.

        Args:
            greedy: If True, generate 1 response per prompt with temperature=0 (for eval).
        """
        if self.vllm_engines is not None and len(self.vllm_engines) > 0:
            return self._generate_with_vllm(prompts, labels, greedy=greedy)
        else:
            return self._generate_with_student(prompts)

    def _generate_with_student(self, prompts: list[str]) -> list[list[str]]:
        """Generate responses using the student model directly (fallback when no vLLM)."""
        logger.info("Generating with student model (no vLLM engines)")
        refs = self.student_model_group.async_run_method(
            method_name="generate",
            prompts=prompts,
            num_samples=self.n_samples_per_prompt,
            temperature=self.generate_kwargs.get("temperature", 0.7),
            top_p=self.generate_kwargs.get("top_p", 0.95),
            max_new_tokens=self.max_gen_len,
        )
        results = ray.get(refs)
        return results[0] if results else [[] for _ in prompts]

    def _generate_with_vllm(self, prompts: list[str], labels: list[str] = None, greedy: bool = False) -> list[list[str]]:
        """Generate responses per prompt using vLLM engines."""
        if greedy:
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=self.max_gen_len,
            )
        else:
            sampling_params = SamplingParams(
                temperature=self.generate_kwargs.get("temperature", 0.7),
                top_p=self.generate_kwargs.get("top_p", 0.95),
                max_tokens=self.max_gen_len,
            )

        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        num_engines = len(self.vllm_engines)
        num_samples = 1 if greedy else self.n_samples_per_prompt

        # Partition prompts across engines via round-robin
        engine_prompts = [[] for _ in range(num_engines)]
        engine_indices = [[] for _ in range(num_engines)]  # original indices
        for i, prompt in enumerate(prompts):
            eidx = i % num_engines
            engine_prompts[eidx].append(prompt)
            engine_indices[eidx].append(i)

        # Send batches to each engine in parallel
        engine_refs = []
        for eidx in range(num_engines):
            if engine_prompts[eidx]:
                ref = self.vllm_engines[eidx].generate_batch.remote(
                    prompts=engine_prompts[eidx],
                    sampling_params=sampling_params,
                    num_samples=num_samples,
                )
                engine_refs.append((eidx, ref))

        # Collect results and reorder to original prompt order
        all_responses = [None] * len(prompts)
        for eidx, ref in engine_refs:
            batch_results = ray.get(ref)  # list[list[str]], one per prompt
            for local_idx, orig_idx in enumerate(engine_indices[eidx]):
                responses = []
                for sample in batch_results[local_idx]:
                    if isinstance(sample, str):
                        responses.append(sample)
                    else:
                        responses.append(str(sample))
                all_responses[orig_idx] = responses

        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        return all_responses

    def _tokenize_sequence(self, prompt: str, response: str):
        """Tokenize prompt+response and return (input_ids, attention_mask, prompt_len)."""
        prompt_tokens = self.tokenizer(
            prompt, return_tensors="pt", max_length=self.max_input_len,
            truncation=True, add_special_tokens=False,
        )
        prompt_len = prompt_tokens["input_ids"].shape[1]

        full_text = prompt + response
        if not full_text.endswith(self.tokenizer.eos_token):
            full_text += self.tokenizer.eos_token

        tokens = self.tokenizer(
            full_text, return_tensors="pt",
            max_length=self.max_input_len + self.max_gen_len,
            truncation=True, add_special_tokens=False,
        )
        return tokens["input_ids"], tokens["attention_mask"], prompt_len

    def _build_teacher_prompt(self, raw_input: str, label: str, reference_output: str) -> str:
        """Build teacher prompt with reference CoT + label in system prompt.

        The teacher sees the correct answer and reference reasoning in its system prompt,
        so its logits on the student's response tokens are answer-aware.
        """
        teacher_system = build_teacher_system_prompt(label, reference_output)
        messages = [
            {"role": "system", "content": teacher_system},
            {"role": "user", "content": raw_input},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Match student's thinking mode: append <think>\n
        if "<think>" not in prompt:
            prompt += "<think>\n"
        return prompt

    def _tokenize_teacher_sequence(self, raw_input: str, label: str, reference_output: str, response: str):
        """Tokenize teacher prompt (with ref CoT + label) + student response."""
        teacher_prompt = self._build_teacher_prompt(raw_input, label, reference_output)
        return self._tokenize_sequence(teacher_prompt, response)

    def fit(self):
        """Main VtD training loop (semantic-entropy weighted distillation).

        For each rollout batch of prompts:
        1. vLLM generates n_samples_per_prompt responses per prompt
        2. Verify against labels → correct / incorrect
        3. Teacher generates N responses per prompt → compute semantic entropy
        4. Collect distillation buffer (all responses + teacher logits + SE weights)
        5. Student trains from buffers in micro-batches (DeepSpeed handles accumulation)
        6. Sync weights to vLLM
        """
        global_step = 0

        for episode in range(self.args.num_episodes):
            pbar = tqdm(
                range(len(self.prompts_dataloader)),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
            )

            for prompts, labels, reference_outputs, raw_inputs in self.prompts_dataloader:
                start_time = time.time()

                # ============ Phase 1: Generate ============
                logger.info("========== 阶段1: 学生模型采样生成 ==========")
                if getattr(self.args, "colocate_all_models", False):
                    ray.get(self.teacher_model_group.async_run_method("offload_states"))

                student_responses = self._generate(prompts, labels)
                phase1_time = time.time() - start_time

                batch_correct = []
                batch_incorrect = []
                total_correct = 0
                total_incorrect = 0
                for i, responses in enumerate(student_responses):
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

                total = total_correct + total_incorrect
                batch_acc = total_correct / total if total > 0 else 0.0
                logger.info(
                    f"[采样] {len(prompts)} 个提示, "
                    f"正确 {total_correct}/{total} ({batch_acc:.1%})"
                )

                # ============ Phase 2: Collect teacher logits + Semantic Entropy ============
                logger.info("========== 阶段2: 教师语义熵采样 + logits收集 ==========")
                if getattr(self.args, "colocate_all_models", False):
                    ray.get(self.teacher_model_group.async_run_method("reload_states"))

                phase2_start = time.time()
                teacher_actors = self.teacher_model_group._actor_handlers
                num_teacher = len(teacher_actors)

                # ---- 2a: Teacher generates N responses per prompt for semantic entropy ----
                se_gen_refs = []
                for idx, i in enumerate(range(len(prompts))):
                    teacher_prompt = self._build_teacher_prompt(
                        raw_inputs[i], labels[i], reference_outputs[i]
                    )
                    actor_idx = idx % num_teacher
                    ref = teacher_actors[actor_idx].batch_generate.remote(
                        [teacher_prompt],
                        num_samples=self.se_n_samples,
                        temperature=self.generate_kwargs.get("temperature", 0.7),
                        top_p=self.generate_kwargs.get("top_p", 0.95),
                    )
                    se_gen_refs.append((i, ref))

                # Collect SE responses and compute per-prompt semantic entropy
                se_per_prompt = []
                se_weight_per_prompt = []
                for i, ref in se_gen_refs:
                    teacher_responses = ray.get(ref)[0]  # [0] because we sent [prompt]
                    se = compute_semantic_entropy(teacher_responses, self.se_n_samples)
                    se_w = semantic_entropy_to_weight(se)
                    se_per_prompt.append(se)
                    se_weight_per_prompt.append(se_w)

                avg_se = sum(se_per_prompt) / len(se_per_prompt) if se_per_prompt else 0.0
                avg_se_w = sum(se_weight_per_prompt) / len(se_weight_per_prompt) if se_weight_per_prompt else 1.0
                logger.info(
                    f"[阶段2] 语义熵: 平均 SE={avg_se:.3f}, 平均权重={avg_se_w:.3f} "
                    f"(N={self.se_n_samples} 采样/提示)"
                )

                # ---- 2b: Collect ALL student responses for distillation ----
                all_student_distill_ids = []
                all_student_distill_masks = []
                all_student_distill_loss_masks = []
                all_teacher_distill_ids = []
                all_teacher_distill_masks = []
                all_teacher_prompt_lens = []
                all_student_prompt_lens = []
                all_se_weights = []  # per-sample SE weight (same for all responses of same prompt)
                distill_prompt_indices = []
                for i, prompt in enumerate(prompts):
                    all_responses = batch_correct[i] + batch_incorrect[i]
                    for resp in all_responses:
                        s_ids, s_mask, s_plen = self._tokenize_sequence(prompt, resp)
                        s_seq_len = s_ids.shape[1]
                        loss_mask = torch.zeros(1, s_seq_len)
                        loss_mask[:, s_plen:] = 1.0
                        loss_mask = loss_mask * s_mask
                        all_student_distill_ids.append(s_ids)
                        all_student_distill_masks.append(s_mask)
                        all_student_distill_loss_masks.append(loss_mask)
                        all_student_prompt_lens.append(s_plen)

                        t_ids, t_mask, t_plen = self._tokenize_teacher_sequence(
                            raw_inputs[i], labels[i], reference_outputs[i], resp
                        )
                        all_teacher_distill_ids.append(t_ids)
                        all_teacher_distill_masks.append(t_mask)
                        all_teacher_prompt_lens.append(t_plen)

                        all_se_weights.append(se_weight_per_prompt[i])
                        distill_prompt_indices.append(i)

                total_distill_fired = len(all_student_distill_ids)
                logger.info(
                    f"[阶段2] 收集到 {total_distill_fired} 条序列, "
                    f"准备批量教师前向 (answer-aware)"
                )

                # ---- 2c: Batch teacher forward for logits (distributed across GPUs) ----
                if total_distill_fired > 0:
                    per_actor_ids = [[] for _ in range(num_teacher)]
                    per_actor_masks = [[] for _ in range(num_teacher)]
                    per_actor_indices = [[] for _ in range(num_teacher)]
                    for idx in range(total_distill_fired):
                        actor_idx = idx % num_teacher
                        per_actor_ids[actor_idx].append(all_teacher_distill_ids[idx])
                        per_actor_masks[actor_idx].append(all_teacher_distill_masks[idx])
                        per_actor_indices[actor_idx].append(idx)

                    teacher_ray_refs = []
                    for a in range(num_teacher):
                        if per_actor_ids[a]:
                            ref = teacher_actors[a].batch_get_logits.remote(
                                per_actor_ids[a], per_actor_masks[a],
                                micro_batch_size=getattr(self.args, "teacher_micro_batch_size", 4),
                            )
                            teacher_ray_refs.append((a, ref))
                    logger.info(
                        f"[阶段2] 教师批量前向: {total_distill_fired} 条序列, "
                        f"分发到 {len(teacher_ray_refs)}/{num_teacher} 个GPU"
                    )

                    all_teacher_results = [None] * total_distill_fired
                    for a, ref in teacher_ray_refs:
                        results = ray.get(ref)
                        for local_idx, global_idx in enumerate(per_actor_indices[a]):
                            all_teacher_results[global_idx] = results[local_idx]
                        logger.info(f"[阶段2] 教师GPU {a} 完成: {len(results)} 条序列")
                else:
                    all_teacher_results = []

                # ---- Build distill items with SE weights ----
                distill_per_prompt = [[] for _ in prompts]
                for idx in range(total_distill_fired):
                    prompt_idx = distill_prompt_indices[idx]
                    t_topk_vals, t_topk_ids = all_teacher_results[idx]
                    t_plen = all_teacher_prompt_lens[idx]
                    s_plen = all_student_prompt_lens[idx]
                    s_ids = all_student_distill_ids[idx]
                    s_seq_len = s_ids.shape[1]
                    top_k = t_topk_vals.shape[-1]

                    t_response_vals = t_topk_vals[:, t_plen:, :]
                    t_response_ids = t_topk_ids[:, t_plen:, :]
                    response_len = t_response_vals.shape[1]

                    aligned_vals = torch.zeros(1, s_seq_len, top_k, dtype=t_topk_vals.dtype)
                    aligned_ids = torch.zeros(1, s_seq_len, top_k, dtype=t_topk_ids.dtype)
                    actual_len = min(response_len, s_seq_len - s_plen)
                    aligned_vals[:, s_plen:s_plen + actual_len, :] = t_response_vals[:, :actual_len, :]
                    aligned_ids[:, s_plen:s_plen + actual_len, :] = t_response_ids[:, :actual_len, :]

                    # Include SE weight as 6th element
                    distill_per_prompt[prompt_idx].append((
                        s_ids, all_student_distill_masks[idx],
                        all_student_distill_loss_masks[idx], aligned_vals, aligned_ids,
                        all_se_weights[idx],
                    ))

                total_distill = sum(len(p) for p in distill_per_prompt)
                phase2_time = time.time() - phase2_start
                logger.info(
                    f"[阶段2] 完成: {total_distill} 条蒸馏样本 (SE加权), "
                    f"共 {len(prompts)} 个提示 (耗时 {phase2_time:.1f}s)"
                )

                # ============ Phase 3: Train from buffers ============
                logger.info("========== 阶段3: 学生模型训练 (SE加权蒸馏) ==========")
                phase3_start = time.time()

                if getattr(self.args, "colocate_all_models", False):
                    ray.get(self.teacher_model_group.async_run_method("offload_states"))

                if self.args.deepspeed_enable_sleep:
                    ray.get(self.student_model_group.async_run_method("reload_states"))

                # Flatten per-prompt lists into flat lists for distributed data sharding.
                distill_flat = [item for plist in distill_per_prompt for item in plist]
                world_size = self.args.student_num_nodes * self.args.student_num_gpus_per_node
                micro_bs = self.args.micro_train_batch_size

                # Pad distill to multiple of world_size * micro_bs
                d_chunk = world_size * micro_bs
                if distill_flat:
                    n_pad = (-len(distill_flat)) % d_chunk
                    distill_flat.extend([None] * n_pad)
                else:
                    distill_flat = [None] * d_chunk

                logger.info(
                    f"[阶段3] 蒸馏样本={len(distill_flat)} "
                    f"(有效 {sum(x is not None for x in distill_flat)}), "
                    f"world_size={world_size}"
                )

                train_refs = self.student_model_group.async_run_method(
                    "vtd_train",
                    distill_items_flat=distill_flat,
                )
                status = ray.get(train_refs)[0]

                if self.args.deepspeed_enable_sleep:
                    ray.get(self.student_model_group.async_run_method("offload_states"))

                phase3_time = time.time() - phase3_start

                # ============ Phase 4: Sync to vLLM ============
                logger.info("========== 阶段4: 同步权重到 vLLM ==========")
                if self.vllm_engines is not None:
                    if self.args.vllm_enable_sleep:
                        batch_vllm_engine_call(self.vllm_engines, "wake_up")
                    ray.get(self.student_model_group.async_run_method("broadcast_to_vllm"))

                    if self.args.vllm_enable_sleep:
                        batch_vllm_engine_call(self.vllm_engines, "sleep")

                # ============ Logging ============
                global_step += 1
                step_time = time.time() - start_time
                status["accuracy"] = batch_acc
                status["num_correct"] = total_correct
                status["num_incorrect"] = total_incorrect
                status["num_distill_items"] = total_distill
                status["avg_semantic_entropy"] = avg_se
                status["avg_se_weight"] = avg_se_w
                status["step_time"] = step_time
                status["phase1_generate_time"] = phase1_time
                status["phase2_collect_time"] = phase2_time
                status["phase3_train_time"] = phase3_time

                logger.info(
                    f"步骤 {global_step}: {status} "
                    f"[采样={phase1_time:.1f}s 收集={phase2_time:.1f}s 训练={phase3_time:.1f}s 总计={step_time:.1f}s]"
                )

                self._save_logs_and_checkpoints(global_step, status)

                if global_step % self.args.eval_steps == 0 and self.eval_dataloader:
                    self._evaluate(global_step)

                pbar.update(1)

        if self.wandb_logger:
            self.wandb_logger.close()
        if self.tensorboard_logger:
            self.tensorboard_logger.close()

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

    def _save_logs_and_checkpoints(self, global_step, logs_dict):
        if global_step % self.args.logging_steps == 0:
            if self.wandb_logger:
                self.wandb_logger.log_train(global_step, logs_dict)
            if self.tensorboard_logger:
                self.tensorboard_logger.log_train(global_step, logs_dict)
            self._log_to_file(global_step, logs_dict, prefix="train")

        if global_step % self.args.save_steps == 0:
            tag = f"global_step{global_step}"
            client_states = {"global_step": global_step}
            ray.get(self.student_model_group.async_run_method(
                "save_checkpoint", tag=tag, client_states=client_states
            ))

    @torch.no_grad()
    def _evaluate(self, global_step):
        eval_start = time.time()
        n_eval_batches = len(self.eval_dataloader)
        logger.info(f"========== 开始贪心评估 (步骤 {global_step}, 共 {n_eval_batches} 个批次) ==========")
        total_correct = 0
        total_samples = 0
        eval_details = []

        for batch_idx, (prompts, labels, _ref_outputs, _raw_inputs) in enumerate(self.eval_dataloader):
            logger.info(f"[评估] 批次 {batch_idx + 1}/{n_eval_batches} — 正在生成 {len(prompts)} 个提示...")
            responses = self._generate(prompts, labels, greedy=True)
            for i, resps in enumerate(responses):
                predicted = extract_answer(resps[0])
                gt_stripped = strip_string(str(labels[i]))
                is_correct = math_equal(predicted, gt_stripped)
                if is_correct:
                    total_correct += 1
                total_samples += 1
                eval_details.append({
                    "idx": total_samples - 1,
                    "label": labels[i],
                    "correct": is_correct,
                    "predicted": predicted,
                })
            running_acc = total_correct / max(total_samples, 1)
            logger.info(f"[评估] 批次 {batch_idx + 1}/{n_eval_batches} 完成 — "
                        f"当前准确率: {total_correct}/{total_samples} = {running_acc:.1%}")

        accuracy = total_correct / max(total_samples, 1)
        eval_time = time.time() - eval_start
        logger.info(f"[评估] 步骤 {global_step} 完成: 准确率 {accuracy:.1%} ({total_correct}/{total_samples}), 耗时 {eval_time:.1f}s")
        logs = {
            "eval_accuracy": accuracy,
            "eval_total": total_samples,
            "eval_time": eval_time,
        }

        if self.wandb_logger:
            self.wandb_logger.log_eval(global_step, logs)
        if self.tensorboard_logger:
            self.tensorboard_logger.log_eval(global_step, logs)
        self._log_to_file(global_step, logs, prefix="eval")

        # Save detailed eval results to txt
        self._save_eval_results(global_step, accuracy, total_correct, total_samples, eval_details)

        logger.info(f"[评估] 步骤 {global_step} 结果: {logs}")

    def _save_eval_results(self, global_step, accuracy, total_correct, total_samples, details):
        """Save per-sample eval results to a txt file under save_path/eval_results/<dataset_name>/."""
        eval_dataset = getattr(self.args, "eval_dataset", "unknown")
        # e.g. "openai/gsm8k" -> "gsm8k"
        dataset_name = eval_dataset.split("/")[-1] if eval_dataset else "unknown"

        eval_dir = os.path.join(self.project_dir, "eval_results", dataset_name)
        os.makedirs(eval_dir, exist_ok=True)

        txt_path = os.path.join(eval_dir, f"eval_step_{global_step}.txt")
        with open(txt_path, "w") as f:
            f.write(f"Eval at step {global_step}\n")
            f.write(f"Dataset: {eval_dataset}\n")
            f.write(f"Model: {self.args.pretrain}\n")
            f.write(f"Greedy Accuracy: {total_correct}/{total_samples} = {accuracy:.2%}\n")
            f.write(f"{'='*60}\n")
            for d in details:
                status = "O" if d["correct"] else "X"
                f.write(f"[{status}] #{d['idx']:3d}  gt={d['label']}  pred={d['predicted']}\n")

        logger.info(f"[评估] 结果已保存到 {txt_path}")
