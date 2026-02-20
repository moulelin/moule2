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
                    label_key="answer", apply_chat_template=None, system_prompt=None,
                    enable_thinking=False, few_shot_examples=None):
    """Preprocess data for VtD: extract prompt, label, optional reference CoT, and raw input."""
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

    return prompt, label, reference_output, raw_input


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
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.input_template = input_template

        input_key = getattr(self.strategy.args, "input_key", "question")
        label_key = getattr(self.strategy.args, "label_key", "answer")
        output_key = getattr(self.strategy.args, "output_key", "answer")
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        enable_thinking = getattr(self.strategy.args, "enable_thinking", False)
        num_shots = getattr(self.strategy.args, "num_shots", 4)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        system_prompt = VTD_SYSTEM_PROMPT
        # Only use few-shot examples during evaluation, not training
        few_shot_examples = GSM8K_FEW_SHOT_EXAMPLES[:num_shots] if (is_eval and num_shots > 0) else None

        self.prompts = []
        self.labels = []
        self.reference_outputs = []
        self.raw_inputs = []

        for data in tqdm(dataset, desc="Preprocessing VtD data", disable=not self.strategy.is_rank_0()):
            prompt, label, ref_output, raw_input = preprocess_data(
                data, input_template, input_key,
                output_key=output_key,
                label_key=label_key,
                apply_chat_template=apply_chat_template,
                system_prompt=system_prompt,
                enable_thinking=enable_thinking,
                few_shot_examples=few_shot_examples,
            )
            if prompt and label:
                self.prompts.append(prompt)
                self.labels.append(label)
                self.reference_outputs.append(ref_output)
                self.raw_inputs.append(raw_input)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.labels[idx], self.reference_outputs[idx], self.raw_inputs[idx]

    def collate_fn(self, item_list):
        prompts = []
        labels = []
        reference_outputs = []
        raw_inputs = []
        for prompt, label, ref_output, raw_input in item_list:
            prompts.append(prompt)
            labels.append(label)
            reference_outputs.append(ref_output)
            raw_inputs.append(raw_input)
        return prompts, labels, reference_outputs, raw_inputs
