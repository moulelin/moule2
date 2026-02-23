"""
Strict quality filtering using a 70B model (judge only, no answer generation).

For each question, the 70B model evaluates 5 quality criteria:
  1. Well-formed and complete
  2. Unambiguous (exactly one correct answer)
  3. Requires multi-step reasoning (not trivial)
  4. Appropriate difficulty (solvable, not domain-specific)
  5. Closed-form answer (number/expression, not proof/essay)

Only samples passing ALL criteria are kept.

Outputs:
  - filtered_accepted.jsonl   (accepted samples, original fields preserved)
  - filtered_rejected.jsonl   (rejected samples with reason)

Usage:
  python filter_with_70b.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --input evolved_with_se_new.jsonl \
    --output_dir dataset_output \
    --tp 4 --batch_size 512
"""

import argparse
import json
import os
import re
import time

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


JUDGE_SYSTEM_PROMPT = """\
You are a math dataset quality reviewer. Your job is to select good-quality math problems \
suitable for training a reasoning model. Accept problems that require meaningful reasoning.

Evaluate the problem on ALL of the following criteria:
1. Well-formed: The problem statement is complete and understandable. No missing critical information or broken formatting.
2. Unambiguous: There is a clear, deterministic answer. No multiple valid interpretations.
3. Requires reasoning: The problem requires at least 2 steps of mathematical reasoning. REJECT trivial one-step arithmetic or direct lookups.
4. Closed-form answer: The answer is a specific number, expression, or short mathematical object. REJECT proofs, essays, or open-ended "explain why" questions.
5. Self-contained: All necessary information is provided. REJECT if it references external figures, tables, or data not included.

You must output EXACTLY one line:
- If ALL 5 criteria are met: ACCEPT
- If ANY criterion fails: REJECT: <brief reason>

Do NOT solve the problem. Only judge its quality."""

JUDGE_USER_TEMPLATE = "Problem:\n{question}"


def build_judge_prompts(questions, tokenizer):
    """Build quality judgment prompts."""
    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(question=q)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return prompts


def parse_judgment(text):
    """Parse ACCEPT / REJECT from judge output. Returns (is_accept, reason)."""
    # Strip thinking blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

    # Look for ACCEPT/REJECT
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("ACCEPT"):
            return True, "ACCEPT"
        if line.upper().startswith("REJECT"):
            return False, line
    # Fallback: search anywhere
    if "ACCEPT" in text.upper():
        return True, "ACCEPT"
    if "REJECT" in text.upper():
        reason_match = re.search(r"REJECT[:\s]*(.*)", text, re.IGNORECASE)
        return False, reason_match.group(0) if reason_match else "REJECT: unknown"
    return False, "REJECT: no clear judgment"


def main(args):
    print(f"Loading dataset from {args.input}")
    samples = []
    with open(args.input) as f:
        for line in f:
            d = json.loads(line.strip())
            samples.append(d)
    print(f"Total samples: {len(samples)}")

    # Pre-filter: skip SE=None samples
    valid_samples = [s for s in samples if s.get("se") is not None]
    skipped_null = len(samples) - len(valid_samples)
    print(f"Skipped {skipped_null} samples with SE=None")
    print(f"Remaining: {len(valid_samples)} samples")

    # Pre-filter: remove multiple choice questions
    choice_markers = ["(A)", "(B)", "(C)", "(a)", "(b)", "(c)"]
    before = len(valid_samples)
    valid_samples = [
        s for s in valid_samples
        if not any(m in s["question"] for m in choice_markers)
    ]
    print(f"Removed {before - len(valid_samples)} multiple choice questions")
    print(f"Remaining: {len(valid_samples)} samples")

    # Optional: pre-filter by SE threshold
    if args.max_se is not None:
        before = len(valid_samples)
        valid_samples = [s for s in valid_samples if s["se"] <= args.max_se]
        print(f"Filtered SE>{args.max_se}: {before} -> {len(valid_samples)}")

    questions = [s["question"] for s in valid_samples]

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

    judge_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        n=1,
    )

    # Prepare output files
    os.makedirs(args.output_dir, exist_ok=True)
    out_accepted = os.path.join(args.output_dir, "filtered_accepted.jsonl")
    out_rejected = os.path.join(args.output_dir, "filtered_rejected.jsonl")
    for path in [out_accepted, out_rejected]:
        open(path, "w").close()

    total_batches = (len(questions) + args.batch_size - 1) // args.batch_size
    total_accepted = 0
    total_rejected = 0

    for batch_idx in range(total_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(questions))
        batch_questions = questions[start:end]
        batch_samples = valid_samples[start:end]

        t0 = time.time()
        judge_prompts = build_judge_prompts(batch_questions, tokenizer)
        judge_outputs = llm.generate(judge_prompts, judge_params)
        elapsed = time.time() - t0

        f_accepted = open(out_accepted, "a")
        f_rejected = open(out_rejected, "a")

        batch_accepted = 0
        batch_rejected = 0

        for i, output in enumerate(judge_outputs):
            text = output.outputs[0].text
            is_accept, reason = parse_judgment(text)
            sample = batch_samples[i]

            if is_accept:
                f_accepted.write(json.dumps(sample, ensure_ascii=False) + "\n")
                batch_accepted += 1
            else:
                record = dict(sample)
                record["reject_reason"] = reason
                f_rejected.write(json.dumps(record, ensure_ascii=False) + "\n")
                batch_rejected += 1

        f_accepted.close()
        f_rejected.close()

        total_accepted += batch_accepted
        total_rejected += batch_rejected
        total_processed = total_accepted + total_rejected

        print(f"Batch {batch_idx + 1}/{total_batches}: {end}/{len(questions)} done "
              f"({elapsed:.1f}s, batch {batch_accepted}/{len(batch_questions)} accepted, "
              f"total {total_accepted}/{total_processed} = {total_accepted/total_processed*100:.1f}%)")

    # Summary
    total_processed = total_accepted + total_rejected
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Input:    {len(questions)}")
    print(f"  Accepted: {total_accepted} ({total_accepted/total_processed*100:.1f}%)")
    print(f"  Rejected: {total_rejected} ({total_rejected/total_processed*100:.1f}%)")
    print(f"{'='*60}")
    print(f"Saved to: {out_accepted}")
    print(f"Saved to: {out_rejected}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict quality filter with 70B model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="dataset_output")
    parser.add_argument("--max_se", type=float, default=None,
                        help="Pre-filter: skip samples with SE > this value")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Number of questions per batch")
    args = parser.parse_args()
    main(args)
