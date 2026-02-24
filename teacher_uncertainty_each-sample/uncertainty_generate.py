"""
Script 1: Generate teacher responses with token-level probabilities.

For each question, teacher model generates N responses via vLLM.
Each response includes:
  - Full text
  - Extracted \\boxed{} answer
  - Token-level log probabilities
  - Sequence log probability (sum of token logprobs = log of product)

Output: intermediate JSONL for uncertainty_calculation.py

Usage:
  python uncertainty_generate.py \
    --teacher_model Qwen/Qwen3-8B \
    --input evolved_clean.jsonl \
    --output teacher_responses.jsonl \
    --n_samples 8 --tp 1
"""

import argparse
import json
import os
import time


def extract_boxed(text):
    """Extract content from the last \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i - 1].strip()
    return None


def get_token_logprobs(output):
    """Extract token log probabilities from a vLLM CompletionOutput.

    Returns:
        list of float: log probability for each generated token
    """
    if output.logprobs is None:
        return []
    token_logprobs = []
    for token_id, logprob_dict in zip(output.token_ids, output.logprobs):
        if token_id in logprob_dict:
            token_logprobs.append(logprob_dict[token_id].logprob)
        else:
            # Fallback: use the first entry (should not happen with logprobs>=1)
            token_logprobs.append(list(logprob_dict.values())[0].logprob)
    return token_logprobs


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

    # ---- Init teacher model ----
    print(f"Loading teacher: {args.teacher_model} (TP={args.tp})")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    teacher_llm = LLM(
        model=args.teacher_model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model, trust_remote_code=True
    )
    teacher_sampling = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_gen_tokens,
        n=args.n_samples,
        logprobs=1,  # return logprob of each generated token
    )

    # ---- Process in batches ----
    out_file = open(args.output, "a", encoding="utf-8")
    total_processed = 0

    try:
        for batch_start in range(0, len(entries), args.batch_size):
            batch = entries[batch_start:batch_start + args.batch_size]
            t0 = time.time()

            # Build teacher prompts
            teacher_prompts = []
            for entry in batch:
                question = entry["question"]
                messages = [
                    {"role": "system", "content": "You are a helpful math tutor. Solve the problem step by step and put your final answer in \\boxed{}."},
                    {"role": "user", "content": question},
                ]
                text = teacher_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                teacher_prompts.append(text)

            # Generate N responses per prompt with logprobs
            outputs = teacher_llm.generate(teacher_prompts, teacher_sampling)

            # Extract responses with probabilities
            for entry, output in zip(batch, outputs):
                responses = []
                for o in output.outputs:
                    token_lps = get_token_logprobs(o)
                    length = len(token_lps)
                    seq_log_prob = sum(token_lps) if token_lps else None
                    boxed = extract_boxed(o.text)

                    resp_data = {
                        "text": o.text,
                        "boxed": boxed,
                        "length": length,
                        "seq_log_prob": round(seq_log_prob, 6) if seq_log_prob is not None else None,
                    }
                    if args.save_token_logprobs:
                        resp_data["token_logprobs"] = [round(lp, 6) for lp in token_lps]
                    responses.append(resp_data)

                result = dict(entry)  # copy original fields
                result["teacher_responses"] = responses
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                total_processed += 1

            out_file.flush()
            elapsed = time.time() - t0
            progress = min(batch_start + args.batch_size, len(entries))
            print(
                f"[{progress}/{len(entries)}] "
                f"Processed: {total_processed} | "
                f"Batch: {elapsed:.1f}s | "
                f"Speed: {len(batch)/elapsed:.1f} samples/s"
            )

    finally:
        out_file.close()

    print("=" * 60)
    print(f"Done! Processed: {total_processed}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate teacher responses with token-level probabilities"
    )

    # Teacher model
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel for teacher")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)

    # Sampling
    parser.add_argument("--n_samples", type=int, default=8, help="N responses per question")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_gen_tokens", type=int, default=1024)

    # IO
    parser.add_argument("--input", type=str, default="evolved_clean.jsonl")
    parser.add_argument("--output", type=str, default="./teacher_responses.jsonl")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_token_logprobs", action="store_true", default=False,
                        help="Save per-token logprobs (increases file size)")

    args = parser.parse_args()
    main(args)
