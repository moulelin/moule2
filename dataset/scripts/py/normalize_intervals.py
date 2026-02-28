"""
Convert set-builder notation answers to interval notation.

E.g., $\{x \mid -2 < x < 2\}$ → $(-2, 2)$

Usage:
  python normalize_intervals.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --input ../merged/math_merged_en_2.jsonl \
    --tp 2
"""

import argparse
import json
import re
import time

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


SYSTEM_PROMPT = """\
Convert the following math answer from set-builder notation to standard interval notation.

Rules:
1. Convert set-builder notation like {x | -2 < x < 2} to interval notation like (-2, 2).
2. Use ( ) for open endpoints (strict inequality < >).
3. Use [ ] for closed endpoints (inequality ≤ ≥ \\leqslant \\geqslant).
4. Use (-\\infty, a) or (a, +\\infty) for unbounded intervals.
5. Use \\cup for union of intervals.
6. Wrap the result in $...$ delimiters.
7. If the expression cannot be converted to interval notation (e.g., discrete sets, parametric sets), return it unchanged.
8. Output ONLY the converted answer, nothing else."""


def needs_conversion(answer: str) -> bool:
    """Check if answer uses set-builder notation."""
    return bool(re.search(r'\\mid|\\vert', answer)) and re.search(r'[<>]|leqslant|geqslant|\\le[^f]|\\ge[^n]', answer)


def main(args):
    # Load data
    entries = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"Total entries: {len(entries)}")

    # Find entries needing conversion
    to_convert = []
    for i, e in enumerate(entries):
        if needs_conversion(e["answer"]):
            to_convert.append((i, e["answer"]))

    print(f"Answers to convert: {len(to_convert)}")
    for i, a in to_convert:
        print(f"  [{i}] {a}")

    if not to_convert:
        print("Nothing to convert.")
        return

    # Build prompts
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = []
    for _, answer in to_convert:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": answer},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    # Generate
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    print("Generating conversions...")
    outputs = llm.generate(prompts, sampling_params)

    # Apply
    for j, (idx, orig) in enumerate(to_convert):
        raw = outputs[j].outputs[0].text
        # Remove thinking tags
        converted = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        # Take first line only (avoid extra explanation)
        converted = converted.split('\n')[0].strip()

        print(f"  [{idx}] {orig}  →  {converted}")
        entries[idx]["answer"] = converted

    # Write back
    with open(args.input, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nUpdated {len(to_convert)} answers in {args.input}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--tp", type=int, default=2)
    args = parser.parse_args()
    main(args)
