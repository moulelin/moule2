"""
LLM-based quality filter for evolved math problems.

Uses a ~15B model to judge whether each generated problem is:
  1. A valid, well-formed math problem (not a solution/essay/gibberish)
  2. Mathematically coherent (no contradictions or impossible conditions)
  3. Solvable (has enough information to derive an answer)

Usage:
  python clean_with_llm.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --tp 1 \
    --input evolved_raw.jsonl \
    --output evolved_clean.jsonl
"""

import argparse
import json
import os
import time
import re


JUDGE_SYSTEM = (
    "You are a math problem quality judge. "
    "Your job is to decide whether a given text is a valid, well-formed math problem. "
    "Respond with EXACTLY one line in this format:\n"
    "VERDICT: KEEP or VERDICT: REMOVE\n"
    "followed by a one-sentence reason.\n\n"
    "Rules:\n"
    "- KEEP if it is a clear, solvable math problem with sufficient information.\n"
    "- REMOVE if it is:\n"
    "  * A solution/answer instead of a problem\n"
    "  * Incomplete or missing key information needed to solve it\n"
    "  * Mathematically impossible or self-contradictory\n"
    "  * Not a math problem (essay, instructions, code, etc.)\n"
    "  * Garbled, repetitive, or incoherent text\n"
    "  * A trivial or degenerate problem (e.g. 'What is 1+1?')\n"
)

JUDGE_TEMPLATE = "Is the following a valid math problem?\n\n{problem}"


def build_judge_prompts(problems):
    """Build chat prompts for judging."""
    prompts = []
    for prob in problems:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_TEMPLATE.format(problem=prob)},
        ]
        prompts.append(messages)
    return prompts


def parse_verdict(response):
    """Parse KEEP/REMOVE from judge response."""
    response = response.strip()
    # Look for VERDICT: KEEP or VERDICT: REMOVE
    match = re.search(r'VERDICT:\s*(KEEP|REMOVE)', response, re.IGNORECASE)
    if match:
        return match.group(1).upper() == "KEEP"
    # Fallback: check if response starts with or contains KEEP/REMOVE
    upper = response.upper()
    if "KEEP" in upper and "REMOVE" not in upper:
        return True
    if "REMOVE" in upper:
        return False
    # Default: keep (conservative)
    return True


def main(args):
    # ---- Load input ----
    print(f"Loading {args.input}...")
    entries = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"Loaded {len(entries)} entries")

    # ---- Resume support ----
    already_processed = 0
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            already_processed = sum(1 for l in f if l.strip())
    # Also count removed entries from the remove log
    remove_log = args.output.replace(".jsonl", "_removed.jsonl")
    if os.path.exists(remove_log):
        with open(remove_log, "r", encoding="utf-8") as f:
            already_processed += sum(1 for l in f if l.strip())

    if already_processed > 0:
        print(f"Resuming: {already_processed} already processed, skipping...")
        entries = entries[already_processed:]
        if not entries:
            print("All entries already processed!")
            return

    print(f"Entries to judge: {len(entries)}")

    # ---- Init vLLM ----
    print(f"Loading model: {args.model} (TP={args.tp})")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=0.0,  # deterministic judging
        max_tokens=128,   # short verdict + reason
    )

    # ---- Process in batches ----
    kept = 0
    removed = 0
    out_file = open(args.output, "a", encoding="utf-8")
    remove_file = open(remove_log, "a", encoding="utf-8")

    try:
        for batch_start in range(0, len(entries), args.batch_size):
            batch = entries[batch_start:batch_start + args.batch_size]
            t0 = time.time()

            # Build prompts
            problems = [e["question"] for e in batch]
            prompts = build_judge_prompts(problems)

            # Apply chat template
            formatted = []
            for msgs in prompts:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                formatted.append(text)

            # Generate verdicts
            outputs = llm.generate(formatted, sampling_params)

            # Parse and write results
            for entry, output in zip(batch, outputs):
                response = output.outputs[0].text
                keep = parse_verdict(response)

                if keep:
                    out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    # Log removed entries with reason
                    entry["_remove_reason"] = response.strip()[:200]
                    remove_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    removed += 1

            out_file.flush()
            remove_file.flush()

            elapsed = time.time() - t0
            progress = min(batch_start + args.batch_size, len(entries))
            total = kept + removed
            keep_rate = kept / total * 100 if total > 0 else 0
            print(
                f"[{progress}/{len(entries)}] "
                f"Kept: {kept} | Removed: {removed} | "
                f"Keep rate: {keep_rate:.1f}% | "
                f"Batch: {elapsed:.1f}s"
            )

    finally:
        out_file.close()
        remove_file.close()

    print("=" * 60)
    print(f"Done! Kept: {kept}, Removed: {removed}, Keep rate: {kept/(kept+removed)*100:.1f}%")
    print(f"Clean output: {args.output}")
    print(f"Removed log:  {remove_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-based math problem quality filter")

    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                        help="Judge model (~15B)")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)

    # IO
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    # Generation
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for vLLM judging")

    args = parser.parse_args()
    main(args)
