"""
Augment math dataset by generating a variant for each problem using Qwen2.5-72B-Instruct.

For each row, the 72B model creates a new problem that:
  - Tests the same math concept / knowledge point
  - Has noticeably different numbers, setup, or context
  - Comes with a verified answer

The output merges original + generated rows (2x dataset size).

Outputs:
  - math_augmented.csv       (original + generated, 2x rows)
  - math_augment_log.jsonl   (detailed log with generation info)

Usage:
  python augment_math_csv.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --input ../cleaned/math_cleaned.csv \
    --output_dir ../augmented \
    --tp 4 --batch_size 64
"""

import argparse
import csv
import json
import os
import re
import time

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


AUGMENT_SYSTEM_PROMPT = """\
You are a math problem designer. Given an existing math problem and its answer, \
create a NEW problem that is clearly different but tests the same mathematical concept.

Requirements for the new problem:
1. **Same concept, different problem**: Keep the same math topic and difficulty level, \
but change the scenario, numbers, conditions, or framing significantly.
2. **Not a trivial modification**: Do NOT just change one number. Change at least TWO of: \
the numbers/coefficients, the context/story, the structure of the problem, or add/remove a condition.
3. **Self-contained**: The new problem must include all necessary information.
4. **Correct answer**: You must solve the new problem and provide the correct answer.
5. **Same language**: If the original is in Chinese, write in Chinese. If English, write in English.
6. **Proper LaTeX**: Use $...$ for all math expressions.

You MUST output in EXACTLY this JSON format (no other text):
{
  "question": "<new problem>",
  "answer": "<answer to the new problem>"
}"""

AUGMENT_USER_TEMPLATE = """Original Problem: {question}

Original Answer: {answer}

Create a clearly different problem that tests the same math concept."""


def build_augment_prompts(questions, answers, tokenizer):
    """Build augmentation prompts."""
    prompts = []
    for q, a in zip(questions, answers):
        messages = [
            {"role": "system", "content": AUGMENT_SYSTEM_PROMPT},
            {"role": "user", "content": AUGMENT_USER_TEMPLATE.format(question=q, answer=a)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return prompts


def parse_augment_output(text):
    """Parse JSON output from the model. Returns dict or None."""
    # Strip thinking blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

    # Try to extract JSON block
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        json_str = json_match.group()
        # Try parsing as-is first, then fix trailing commas
        for candidate in [json_str, re.sub(r",\s*}", "}", json_str)]:
            try:
                result = json.loads(candidate)
                required = ["question", "answer"]
                if all(k in result for k in required):
                    return result
            except json.JSONDecodeError:
                continue

    return None


def main(args):
    print(f"Loading dataset from {args.input}")
    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Total rows: {len(rows)}")

    questions = [r["question"] for r in rows]
    answers = [r["answer"] for r in rows]

    # Load model
    print(f"Loading model: {args.model} (TP={args.tp})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_tokens,
        n=1,
    )

    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "math_augmented.csv")
    out_log = os.path.join(args.output_dir, "math_augment_log.jsonl")
    open(out_log, "w").close()

    generated_rows = []
    total_success = 0
    total_parse_fail = 0

    total_batches = (len(questions) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(questions))
        batch_q = questions[start:end]
        batch_a = answers[start:end]

        prompts = build_augment_prompts(batch_q, batch_a, tokenizer)

        # Retry with smaller sub-batches on failure
        outputs = None
        cur_batch_size = len(prompts)
        for attempt in range(3):
            try:
                t0 = time.time()
                if cur_batch_size >= len(prompts):
                    outputs = llm.generate(prompts, sampling_params)
                else:
                    outputs = []
                    for sub_start in range(0, len(prompts), cur_batch_size):
                        sub_prompts = prompts[sub_start:sub_start + cur_batch_size]
                        sub_out = llm.generate(sub_prompts, sampling_params)
                        outputs.extend(sub_out)
                elapsed = time.time() - t0
                break
            except Exception as e:
                cur_batch_size = max(1, cur_batch_size // 2)
                print(f"  Batch {batch_idx + 1} attempt {attempt + 1} failed: {e}")
                print(f"  Retrying with sub-batch size {cur_batch_size}...")
                if attempt == 2:
                    print(f"  FATAL: Batch {batch_idx + 1} failed after 3 attempts, skipping")
                    elapsed = 0.0

        f_log = open(out_log, "a", encoding="utf-8")

        for i in range(len(batch_q)):
            idx = start + i
            log_entry = {
                "index": idx,
                "original_question": batch_q[i],
                "original_answer": batch_a[i],
            }

            if outputs is not None and i < len(outputs):
                text = outputs[i].outputs[0].text
                result = parse_augment_output(text)
                log_entry["raw_output"] = text
            else:
                result = None
                log_entry["raw_output"] = ""

            if result is not None:
                generated_rows.append({
                    "question": result["question"],
                    "answer": result["answer"],
                })
                log_entry.update(result)
                total_success += 1
                f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            else:
                total_parse_fail += 1

        f_log.close()

        print(
            f"Batch {batch_idx + 1}/{total_batches}: {end}/{len(questions)} done "
            f"({elapsed:.1f}s) | success: {total_success}, parse_fail: {total_parse_fail}"
        )

    # Merge: original rows + generated rows
    all_rows = []
    for r in rows:
        all_rows.append({"question": r["question"], "answer": r["answer"]})
    all_rows.extend(generated_rows)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Original rows:  {len(rows)}")
    print(f"  Generated:      {total_success}")
    print(f"  Parse failed:   {total_parse_fail}")
    print(f"  Total output:   {len(all_rows)}")
    print(f"{'='*60}")
    print(f"Augmented CSV: {out_csv}")
    print(f"Detailed log:  {out_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment math dataset with 72B model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../augmented")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Max generation tokens per sample")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of rows per batch")
    args = parser.parse_args()
    main(args)
