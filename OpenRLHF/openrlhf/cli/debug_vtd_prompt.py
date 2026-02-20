"""
Debug script: load VtD dataset and print prompts, labels, reference outputs.
Uses the same blending_datasets + VtDPromptDataset pipeline as real training.
No Ray, no GPU, no model loading — just data + tokenizer.
"""

import argparse
import sys
import os

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openrlhf.datasets.vtd_dataset import VtDPromptDataset, VTD_SYSTEM_PROMPT
from openrlhf.datasets.utils import blending_datasets
from openrlhf.trainer.vtd_trainer_ray import extract_answer, normalize_answer


class MockStrategy:
    """Minimal strategy mock for blending_datasets and VtDPromptDataset."""

    def __init__(self, args):
        self.args = args

    def is_rank_0(self):
        return True

    def print(self, *a, **kw):
        print(*a, **kw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt_data", type=str, default="openai/gsm8k")
    parser.add_argument("--prompt_data_probs", type=str, default=None)
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="question")
    parser.add_argument("--label_key", type=str, default="answer")
    parser.add_argument("--output_key", type=str, default="answer")
    parser.add_argument("--apply_chat_template", action="store_true", default=True)
    parser.add_argument("--enable_thinking", action="store_true", default=True,
                        help="Append <think> after assistant turn (Qwen3 thinking mode)")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ms", action="store_true", default=False)
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.pretrain}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrain, trust_remote_code=True,
        use_fast=not args.disable_fast_tokenizer,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    strategy = MockStrategy(args)

    # Load dataset via blending_datasets (same path as real training)
    print(f"\nLoading dataset via blending_datasets: {args.prompt_data}")
    raw = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        dataset_split=args.prompt_split,
    )
    raw = raw.select(range(min(args.max_samples, len(raw))))

    print(f"Dataset columns: {raw.column_names}")
    print(f"Dataset size: {len(raw)}")

    # Print raw data
    print("\n" + "=" * 80)
    print("RAW DATA (before processing)")
    print("=" * 80)
    for i in range(len(raw)):
        row = raw[i]
        print(f"\n--- Sample {i} ---")
        for col in raw.column_names:
            val = str(row[col])
            print(f"  [{col}]: {val[:200]}{'...' if len(val) > 200 else ''}")

    # Build VtDPromptDataset (same path as real training)
    print("\n" + "=" * 80)
    print("VtDPromptDataset OUTPUT")
    print(f"  apply_chat_template={args.apply_chat_template}")
    print(f"  system_prompt={VTD_SYSTEM_PROMPT}")
    print(f"  input_key={args.input_key}, label_key={args.label_key}, output_key={args.output_key}")
    print("=" * 80)

    dataset = VtDPromptDataset(
        raw, tokenizer, strategy,
        input_template=args.input_template,
        max_length=512,
    )

    for i in range(len(dataset)):
        prompt, label, ref_output, raw_input = dataset[i]
        print(f"\n--- Sample {i} ---")
        print(f"[PROMPT] ({len(prompt)} chars):")
        print(prompt)
        print(f"\n[LABEL] ({len(label)} chars):")
        print(label[:300])
        print(f"\n[REFERENCE_OUTPUT] ({len(ref_output)} chars):")
        print(ref_output[:300] if ref_output else "(empty)")

        expected = extract_answer(label)
        print(f"\n[EXTRACTED ANSWER from label]: '{expected}'")
        print(f"[NORMALIZED]: '{normalize_answer(expected)}'")

        if ref_output:
            ref_extracted = extract_answer(ref_output)
            print(f"[EXTRACTED ANSWER from ref_output]: '{ref_extracted}'")

        print()


if __name__ == "__main__":
    main()
