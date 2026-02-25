"""
SE Validation: Run Qwen3-8B on solved_accepted.jsonl, compare with ground truth.

For each question:
  1. Teacher (Qwen3-8B) generates a single response (temperature=0, greedy)
  2. Extract \boxed{} answer
  3. Compare with ground truth `answer` field using math_equal
  4. Save result with correctness label

Output: JSONL with fields: question, se, teacher_pred, ground_truth, correct
"""

import argparse
import json
import os
import sys
import time

# Add eval utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "eval"))
from math_eval_utils import extract_answer, math_equal, strip_string


def main(args):
    # ---- Load dataset ----
    print(f"Loading {args.input}...")
    entries = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"Loaded {len(entries)} entries")

    # ---- Resume support ----
    already_done = 0
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            already_done = sum(1 for l in f if l.strip())
    if already_done > 0:
        print(f"Resuming: {already_done} already done, skipping...")
        entries = entries[already_done:]
        if not entries:
            print("All done!")
            return

    print(f"Entries to process: {len(entries)}")

    # ---- Init model ----
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
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    sampling = SamplingParams(
        temperature=0.0,  # greedy decoding
        max_tokens=args.max_gen_tokens,
        n=1,
    )

    # ---- Process in batches ----
    out_file = open(args.output, "a", encoding="utf-8")
    total_processed = 0
    total_correct = 0

    try:
        for batch_start in range(0, len(entries), args.batch_size):
            batch = entries[batch_start:batch_start + args.batch_size]
            t0 = time.time()

            # Build prompts
            prompts = []
            for entry in batch:
                messages = [
                    {"role": "system", "content": "You are a helpful math tutor. Solve the problem step by step and put your final answer in \\boxed{}."},
                    {"role": "user", "content": entry["question"]},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompts.append(text)

            # Generate
            outputs = llm.generate(prompts, sampling)

            # Evaluate
            for entry, output in zip(batch, outputs):
                response_text = output.outputs[0].text
                teacher_pred = extract_answer(response_text)
                ground_truth = strip_string(str(entry["answer"]))

                correct = math_equal(teacher_pred, ground_truth)

                result = {
                    "question": entry["question"],
                    "se": entry.get("se"),
                    "se_raw": entry.get("se_raw"),
                    "difficulty": entry.get("difficulty"),
                    "source": entry.get("source"),
                    "teacher_pred": teacher_pred,
                    "ground_truth": ground_truth,
                    "correct": correct,
                }
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                total_processed += 1
                if correct:
                    total_correct += 1

            out_file.flush()
            elapsed = time.time() - t0
            progress = already_done + min(batch_start + args.batch_size, len(entries))
            acc = total_correct / total_processed * 100 if total_processed > 0 else 0
            print(
                f"[{progress}/{already_done + len(entries)}] "
                f"Acc={acc:.1f}% ({total_correct}/{total_processed}) | "
                f"Batch: {elapsed:.1f}s | "
                f"Speed: {len(batch)/elapsed:.1f} samples/s"
            )

    finally:
        out_file.close()

    acc = total_correct / total_processed * 100 if total_processed > 0 else 0
    print("=" * 60)
    print(f"Done! Processed: {total_processed}")
    print(f"Overall accuracy: {acc:.2f}% ({total_correct}/{total_processed})")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SE validation: run teacher on questions and check correctness"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel (2 for 2xH100)")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_gen_tokens", type=int, default=2048)
    parser.add_argument("--input", type=str,
                        default="dataset_output/solved_accepted.jsonl")
    parser.add_argument("--output", type=str, default="se_validation_results.jsonl")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    main(args)
