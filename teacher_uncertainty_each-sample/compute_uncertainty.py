"""
Offline computation of teacher uncertainty (Semantic Entropy) for each sample.

Pipeline:
  1. Load dataset (evolved_clean.jsonl)
  2. For each question, teacher (Qwen3-8B via vLLM) generates N responses
  3. A small 0.5B model judges pairwise equivalence → Union-Find clustering
  4. Compute normalized SE per sample, store se_weight = 1 - SE
  5. Output dataset with "se" and "se_weight" fields appended

Usage:
  python compute_uncertainty.py \
    --teacher_model Qwen/Qwen3-8B \
    --cluster_model Qwen/Qwen2.5-0.5B-Instruct \
    --input evolved_clean.jsonl \
    --output evolved_with_se.jsonl \
    --n_samples 8 --tp 4
"""

import argparse
import json
import math
import os

import time
from collections import defaultdict


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


# ============ Pairwise Clustering via Small LLM ============

PAIRWISE_SYSTEM = (
    "You judge whether two math solutions arrive at the same final answer. "
    'Reply ONLY "YES" or "NO".'
)

PAIRWISE_TEMPLATE = (
    "Do these two solutions give the same final answer?\n\n"
    "Solution A:\n{resp_a}\n\n"
    "Solution B:\n{resp_b}"
)


class SemanticCluster:
    """Pairwise LLM-based clustering for SE computation."""

    def __init__(self, model_name, max_model_len=2048):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        print(f"[Cluster] Loading {model_name}...")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.12,  # small model, leave room for teacher
            trust_remote_code=True,
            dtype="bfloat16",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=3)
        print(f"[Cluster] Ready.")

    def _build_pair_prompt(self, resp_a, resp_b):
        messages = [
            {"role": "system", "content": PAIRWISE_SYSTEM},
            {"role": "user", "content": PAIRWISE_TEMPLATE.format(resp_a=resp_a, resp_b=resp_b)},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def compute_entropy_batch(self, all_responses: list[list[str]]) -> list[float]:
        """Compute SE for multiple prompts at once.

        Args:
            all_responses: list of N-response lists, one per prompt.
                e.g. [[r1,r2,...,r8], [r1,r2,...,r8], ...]

        Returns:
            list of SE values, one per prompt.
        """
        # Build ALL pair prompts across all prompts
        pair_prompts = []
        pair_meta = []  # (prompt_idx, i, j) for each pair

        for prompt_idx, responses in enumerate(all_responses):
            N = len(responses)
            for i in range(N):
                for j in range(i + 1, N):
                    pair_prompts.append(self._build_pair_prompt(responses[i], responses[j]))
                    pair_meta.append((prompt_idx, i, j))

        if not pair_prompts:
            return [1.0] * len(all_responses)

        # Batch judge ALL pairs in one vLLM call
        outputs = self.llm.generate(pair_prompts, self.sampling_params)

        # Parse verdicts
        verdicts = []
        for output in outputs:
            text = output.outputs[0].text.strip().upper()
            verdicts.append(text.startswith("YES"))

        # Union-Find per prompt
        results = []
        for prompt_idx, responses in enumerate(all_responses):
            N = len(responses)
            if N <= 1:
                results.append(0.0)
                continue

            parent = list(range(N))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            # Apply verdicts for this prompt
            for (pidx, i, j), equiv in zip(pair_meta, verdicts):
                if pidx == prompt_idx and equiv:
                    union(i, j)

            # Count clusters
            groups = defaultdict(int)
            for i in range(N):
                groups[find(i)] += 1

            # Entropy
            entropy = 0.0
            for count in groups.values():
                p = count / N
                if p > 0:
                    entropy -= p * math.log(p)
            max_entropy = math.log(N)
            se = entropy / max_entropy if max_entropy > 0 else 0.0
            results.append(se)

        return results


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

    # ---- Init teacher (vLLM, fast generation) ----
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
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    teacher_sampling = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_gen_tokens,
        n=args.n_samples,  # vLLM generates N responses per prompt in one call
    )

    # ---- Init cluster model ----
    cluster = SemanticCluster(args.cluster_model, max_model_len=2048)

    # ---- Process in batches ----
    out_file = open(args.output, "a", encoding="utf-8")
    total_processed = 0
    se_sum = 0.0

    try:
        for batch_start in range(0, len(entries), args.batch_size):
            batch = entries[batch_start:batch_start + args.batch_size]
            t0 = time.time()

            # Step 1: Build teacher prompts
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

            # Step 2: Teacher generates N responses per prompt (vLLM, n=8)
            outputs = teacher_llm.generate(teacher_prompts, teacher_sampling)

            # Debug: dump first batch teacher outputs
            if batch_start == 0:
                debug_path = os.path.join(os.path.dirname(args.output), "debug_teacher_outputs.jsonl")
                with open(debug_path, "w", encoding="utf-8") as df:
                    for entry_d, output_d in zip(batch, outputs):
                        df.write(json.dumps({
                            "question": entry_d["question"],
                            "responses": [o.text for o in output_d.outputs],
                            "boxed": [extract_boxed(o.text) for o in output_d.outputs],
                        }, ensure_ascii=False) + "\n")
                print(f"[Debug] First batch teacher outputs written to {debug_path}")

            # Collect N responses per prompt, extract \boxed{} answers
            all_answers = []  # extracted boxed answers for clustering
            skip_flags = []   # True if sample should be skipped
            for output in outputs:
                answers = []
                for o in output.outputs:
                    ans = extract_boxed(o.text)
                    if ans is not None and len(ans) <= 200:
                        answers.append(ans)
                # Need at least 2 valid answers to compute SE
                all_answers.append(answers)
                skip_flags.append(len(answers) < 2)

            # Step 3: Cluster + compute SE (only for valid samples)
            valid_answers = [a for a, s in zip(all_answers, skip_flags) if not s]
            if valid_answers:
                valid_se = cluster.compute_entropy_batch(valid_answers)
            else:
                valid_se = []

            # Step 4: Write results
            se_iter = iter(valid_se)
            skipped_in_batch = 0
            for entry, skip in zip(batch, skip_flags):
                if skip:
                    skipped_in_batch += 1
                    continue
                se = next(se_iter)
                entry["se"] = round(se, 4)
                entry["se_weight"] = round(1.0 - se, 4)
                out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                se_sum += se
                total_processed += 1

            out_file.flush()
            elapsed = time.time() - t0
            avg_se = se_sum / total_processed if total_processed > 0 else 0
            progress = min(batch_start + args.batch_size, len(entries))
            print(
                f"[{progress}/{len(entries)}] "
                f"Avg SE={avg_se:.3f} | "
                f"Avg weight={1-avg_se:.3f} | "
                f"Skipped: {skipped_in_batch} | "
                f"Batch: {elapsed:.1f}s | "
                f"Speed: {len(batch)/elapsed:.1f} samples/s"
            )

    finally:
        out_file.close()

    avg_se = se_sum / total_processed if total_processed > 0 else 0
    print("=" * 60)
    print(f"Done! Processed: {total_processed}")
    print(f"Average SE: {avg_se:.3f}, Average weight: {1-avg_se:.3f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline teacher uncertainty computation")

    # Teacher model (generates N responses per question)
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel for teacher")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.60)

    # Cluster model (pairwise judging)
    parser.add_argument("--cluster_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")

    # SE sampling
    parser.add_argument("--n_samples", type=int, default=8, help="N responses per question")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_gen_tokens", type=int, default=1024)

    # IO
    parser.add_argument("--input", type=str, default="evolved_clean.jsonl")
    parser.add_argument("--output", type=str, default="evolved_with_se.jsonl")
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()
    main(args)
