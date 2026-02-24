import re
from typing import Callable

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from openrlhf.utils.utils import zero_pad_sequences


VTD_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}. If you are not sure about the answer, put the most likely answer within \\boxed{}"

# Teacher system prompt: includes reference CoT + label so teacher logits are answer-aware
VTD_TEACHER_SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert math teacher. You have access to the reference solution and correct answer "
    "for this problem. Use this knowledge to provide the most helpful token-level guidance.\n\n"
    "{reference_section}"
    "Correct answer: \\boxed{{{label}}}"
)


def build_teacher_system_prompt(label: str, reference_cot: str = "") -> str:
    """Build teacher system prompt with reference CoT and label embedded."""
    if reference_cot:
        reference_section = f"Reference solution:\n{reference_cot}\n\n"
    else:
        reference_section = ""
    return VTD_TEACHER_SYSTEM_PROMPT_TEMPLATE.format(
        reference_section=reference_section, label=label
    )

# MATH 4-shot CoT examples (competition-level, from Qwen2.5-Math)
MATH_FEW_SHOT_EXAMPLES = [
    (
        "Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.",
        "Let's think step by step\nKevin hops $1/3$ of the remaining distance with every hop.\nHis first hop takes $1/3$ closer.\nFor his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$.\nFor his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$.\nIn general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop.\nWe want to find how far he has hopped after five hops.\nThis is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms.\nThus, Kevin has hopped $\\frac{\\frac{1}{3}\\left(1-\\left(\\frac{2}{3}\\right)^5\\right)}{1-\\frac{2}{3}} = \\boxed{\\frac{211}{243}}$.",
    ),
    (
        "What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?",
        "Let's think step by step\nWe rewrite the equation as $x^2 + 14x + y^2 - 4y = 10$ and then complete the square,\nresulting in  $(x+7)^2-49 + (y-2)^2-4=10$,\nor $(x+7)^2+(y-2)^2=63$.\nThis is the equation of a circle with center $(-7, 2)$ and radius $\\sqrt{63},$\nso the area of this region is $\\pi r^2 = \\boxed{63\\pi}$",
    ),
    (
        "If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?",
        "Let's think step by step\nIf $(x,y)$ lies on the circle,\nso does $(x,-y),$ $(-x,-y),$ and $(-x,-y),$ (which all give the same value of $|x| + |y|$),\nso we can assume that $x \\ge 0$ and $y \\ge 0.$\nThen $|x| + |y| = x + y.$  Squaring, we get\n\\[(x + y)^2 = x^2 + 2xy + y^2 = 1 + 2xy.\\]\nNote that $(x - y)^2 \\ge 0.$\nExpanding, we get $x^2 - 2xy + y^2 \\ge 0,$ so $2xy \\le x^2 + y^2 = 1.$\nHence,\\[1 + 2xy \\le 2,\\]which means $x + y \\le \\sqrt{2}.$\nEquality occurs when $x = y = \\frac{1}{\\sqrt{2}},$\nso the maximum value of $|x| + |y|$ is $\\boxed{\\sqrt{2}}",
    ),
    (
        "If $f(x)=\\frac{ax+b}{cx+d}, abcd\\not=0$ and $f(f(x))=x$ for all $x$ in the domain of $f$, what is the value of $a+d$?",
        "Let's think step by step\nThe condition $f(f(x))$ means that $f$ is the inverse of itself,\nso its graph is symmetrical about the line $y = x$.\nWith a rational function of this form, we will have two asymptotes:\na vertical one at $x=-d/c$ if $cx+d$ does not divide $ax+b$,\nand a horizontal one at $y=a/c$,\nif we take the limit of $f(x)$ as $x$ goes to $\\pm\\infty$.\nIn order for $f$ to be its own inverse, the intersection of the asymptotes must lie on the line $y=x$\nso that it and its asymptotes reflect onto themselves.\nThis means that $-d/c=a/c$,\nand therefore $-d=a$ and $a+d=\\boxed{0}$",
    ),
]

# GSM8K 4-shot CoT examples (from Qwen2.5-Math)
GSM8K_FEW_SHOT_EXAMPLES = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is \\boxed{6}.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is \\boxed{5}.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is \\boxed{39}.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is \\boxed{8}.",
    ),
]


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


def extract_label(text: str) -> str:
    """Extract the final answer from an answer field.

    Handles:
      - GSM8K: "...#### 72"       -> "72"
      - boxed:  "...\\boxed{72}"  -> "72"
      - plain:  "72"              -> "72"
    """
    # GSM8K #### format
    m = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()
    # \boxed{} format with brace-depth counting for nested braces
    boxed = _extract_last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    # Already a plain answer
    return text.strip()


def clean_gsm8k_answer(text: str) -> str:
    """Clean GSM8K answer format to extract only the CoT reasoning.

    GSM8K raw:  "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n...#### 72"
    Cleaned:    "Natalia sold 48/2 = 24 clips in May.\n..."
    """
    # Remove <<calculation>> annotations: <<48/2=24>> -> ""
    text = re.sub(r"<<[^>]*>>", "", text)
    # Remove #### and everything after it (keep only CoT)
    text = re.sub(r"####.*", "", text, flags=re.DOTALL)
    return text.strip()


def preprocess_data(data, input_template=None, input_key="input", output_key=None,
                    label_key="answer", se_weight_key=None, apply_chat_template=None,
                    system_prompt=None, enable_thinking=False, few_shot_examples=None):
    """Preprocess data for VtD: extract prompt, label, optional reference CoT, raw input, and SE weight."""
    raw_input = data[input_key] if isinstance(data[input_key], str) else str(data[input_key])

    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            # Insert few-shot CoT examples
            if few_shot_examples:
                for q, a in few_shot_examples:
                    messages.append({"role": "user", "content": q})
                    messages.append({"role": "assistant", "content": a})
            messages.append({"role": "user", "content": chat})
            chat = messages
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # Force thinking mode: append <think>\n if not already present
        if "<think>" not in prompt:
            prompt += "<think>\n"
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)

    # Ground truth label: extract final answer only (e.g. "72" from "...#### 72")
    raw_label = data.get(label_key, "")
    label = extract_label(raw_label) if raw_label else ""

    # Reference thinking chain (optional, for CoT-guided distillation)
    reference_output = ""
    if output_key:
        raw = data.get(output_key, "")
        if raw:
            # Clean GSM8K format: remove <<calc>>, #### -> \boxed{}
            reference_output = clean_gsm8k_answer(raw)

    # Precomputed SE weight (optional, from offline uncertainty computation)
    se_weight = None
    if se_weight_key:
        se_weight = data.get(se_weight_key)

    return prompt, label, reference_output, raw_input, se_weight


class VtDPromptDataset(Dataset):
    """
    Dataset for VtD (Verify-then-Distill).

    Each item returns: (prompt_text, ground_truth_answer, reference_output)

    Expected dataset format:
        - input_key:  the question/prompt field
        - label_key:  the ground truth answer (for verification)
        - output_key: (optional) reference thinking chain + answer (for distillation guidance)

    When output_key is provided:
        - Distillation is done on the reference CoT (student learns HOW to think)
        - Contrastive loss uses reference CoT as chosen, student incorrect as rejected
    When output_key is absent:
        - Falls back to original VtD: distill on student's correct responses with teacher logits
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        strategy,
        input_template=None,
        max_length: int = 2048,
        is_eval: bool = False,
        eval_few_shot_examples=None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.input_template = input_template

        input_key = getattr(self.strategy.args, "input_key", "question")
        label_key = getattr(self.strategy.args, "label_key", "answer")
        output_key = getattr(self.strategy.args, "output_key", "answer")
        # Eval datasets don't have se_weight — only use for training
        se_weight_key = getattr(self.strategy.args, "se_weight_key", None) if not is_eval else None
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        enable_thinking = getattr(self.strategy.args, "enable_thinking", False)
        num_shots = getattr(self.strategy.args, "num_shots", 4)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        system_prompt = VTD_SYSTEM_PROMPT
        # For eval: use caller-provided few-shot examples (allows MATH vs GSM8K selection)
        # For training: no few-shot examples
        if is_eval and eval_few_shot_examples is not None:
            few_shot_examples = eval_few_shot_examples
        elif is_eval and num_shots > 0:
            few_shot_examples = GSM8K_FEW_SHOT_EXAMPLES[:num_shots]
        else:
            few_shot_examples = None

        self.prompts = []
        self.labels = []
        self.reference_outputs = []
        self.raw_inputs = []
        self.se_weights = []

        for data in tqdm(dataset, desc="Preprocessing VtD data", disable=not self.strategy.is_rank_0()):
            prompt, label, ref_output, raw_input, se_weight = preprocess_data(
                data, input_template, input_key,
                output_key=output_key,
                label_key=label_key,
                se_weight_key=se_weight_key,
                apply_chat_template=apply_chat_template,
                system_prompt=system_prompt,
                enable_thinking=enable_thinking,
                few_shot_examples=few_shot_examples,
            )
            # Skip entries with null se_weight (invalid samples from offline computation)
            if se_weight_key and se_weight is None:
                continue
            # When using precomputed SE, label is optional (no verification needed)
            if not prompt:
                continue
            if not se_weight_key and not label:
                continue
            self.prompts.append(prompt)
            self.labels.append(label)
            self.reference_outputs.append(ref_output)
            self.raw_inputs.append(raw_input)
            self.se_weights.append(se_weight if se_weight is not None else 1.0)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.labels[idx], self.reference_outputs[idx], self.raw_inputs[idx], self.se_weights[idx]

    def collate_fn(self, item_list):
        prompts = []
        labels = []
        reference_outputs = []
        raw_inputs = []
        se_weights = []
        for prompt, label, ref_output, raw_input, se_weight in item_list:
            prompts.append(prompt)
            labels.append(label)
            reference_outputs.append(ref_output)
            raw_inputs.append(raw_input)
            se_weights.append(se_weight)
        return prompts, labels, reference_outputs, raw_inputs, se_weights
