"""
Compute probability-weighted Semantic Entropy (SE) for augmented math dataset.

Two-phase pipeline (following uncertainty_generate.py + uncertainty_calculation.py):
  Phase 1: Teacher (Qwen3-8B) generates N responses with token-level logprobs
           → saves intermediate cache (teacher_responses)
  Phase 2: Cluster (Qwen3-4B) judges answer equivalence + computes weighted SE
           → outputs final JSONL matching solved_accepted.jsonl format

Supports resume: if teacher cache exists, skips Phase 1.

Usage:
  python compute_se_augmented.py \
    --teacher_model Qwen/Qwen3-8B \
    --cluster_model Qwen/Qwen3-4B \
    --input ../augmented/math_augment_log.jsonl \
    --output ../augmented/math_augmented_with_se.jsonl \
    --n_samples 8 --tp 4
"""

import argparse
import csv
import json
import math
import os
import re
import time
from collections import defaultdict


# ============ Utility functions ============

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
    """Extract token log probabilities from a vLLM CompletionOutput."""
    if output.logprobs is None:
        return []
    token_logprobs = []
    for token_id, logprob_dict in zip(output.token_ids, output.logprobs):
        if token_id in logprob_dict:
            token_logprobs.append(logprob_dict[token_id].logprob)
        else:
            token_logprobs.append(list(logprob_dict.values())[0].logprob)
    return token_logprobs


def logsumexp(values):
    """Numerically stable log-sum-exp."""
    if not values:
        return float("-inf")
    max_val = max(values)
    if max_val == float("-inf"):
        return float("-inf")
    return max_val + math.log(sum(math.exp(v - max_val) for v in values))


def compute_weighted_entropy(cluster_ids, seq_log_probs, lengths):
    """Compute probability-weighted semantic entropy.

    Each response's log prob is length-normalized: seq_log_prob / length.
    Cluster probability: p(c) = sum(exp(norm_log_prob_i)) / Z
    Entropy: SE = -sum(p(c) * log(p(c)))
    """
    N = len(cluster_ids)
    if N <= 1:
        return 0.0

    norm_log_probs = [
        slp / length if length > 0 else slp
        for slp, length in zip(seq_log_probs, lengths)
    ]

    log_Z = logsumexp(norm_log_probs)

    cluster_log_probs = defaultdict(list)
    for cid, nlp in zip(cluster_ids, norm_log_probs):
        cluster_log_probs[cid].append(nlp - log_Z)

    entropy = 0.0
    for log_ps in cluster_log_probs.values():
        log_pc = logsumexp(log_ps)
        pc = math.exp(log_pc)
        if pc > 0:
            entropy -= pc * log_pc

    return entropy


# ============ Hybrid Clustering ============

PAIRWISE_SYSTEM = (
    "You judge whether two math answers are equivalent. "
    'Reply ONLY "YES" or "NO".'
)

PAIRWISE_TEMPLATE = (
    "Are these two answers equivalent?\n\n"
    "Answer A: {ans_a}\n\n"
    "Answer B: {ans_b}"
)


class SemanticCluster:
    """Hybrid clustering: exact string match + LLM judge for ambiguous pairs."""

    def __init__(self, model_name, max_model_len=2048, gpu_util=0.80):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        print(f"[Cluster] Loading {model_name}...")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_util,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=3)
        print(f"[Cluster] Ready.")

    def _build_pair_prompt(self, ans_a, ans_b):
        messages = [
            {"role": "system", "content": PAIRWISE_SYSTEM},
            {"role": "user", "content": PAIRWISE_TEMPLATE.format(ans_a=ans_a, ans_b=ans_b)},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def cluster_batch(self, all_answers: list[list[str]]) -> list[list[int]]:
        """Cluster answers for multiple prompts. Returns cluster IDs per response."""
        pair_prompts = []
        pair_meta = []
        for prompt_idx, answers in enumerate(all_answers):
            N = len(answers)
            for i in range(N):
                for j in range(i + 1, N):
                    if answers[i] != answers[j]:
                        pair_prompts.append(self._build_pair_prompt(answers[i], answers[j]))
                        pair_meta.append((prompt_idx, i, j))

        llm_verdicts = {}
        if pair_prompts:
            outputs = self.llm.generate(pair_prompts, self.sampling_params)
            for (pidx, i, j), output in zip(pair_meta, outputs):
                text = output.outputs[0].text.strip().upper()
                llm_verdicts[(pidx, i, j)] = text.startswith("YES")

        all_cluster_ids = []
        for prompt_idx, answers in enumerate(all_answers):
            N = len(answers)
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

            for i in range(N):
                for j in range(i + 1, N):
                    if answers[i] == answers[j]:
                        union(i, j)
                    elif llm_verdicts.get((prompt_idx, i, j), False):
                        union(i, j)

            cluster_ids = [find(i) for i in range(N)]
            all_cluster_ids.append(cluster_ids)

        return all_cluster_ids


# ============ Data loading ============

def load_entries(path):
    """Load entries from jsonl or csv. Returns list of dicts with 'question' and 'answer'."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append({"question": row["question"], "answer": row["answer"]})
        return entries
    else:
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if "question" in d and "answer" in d:
                    entries.append({"question": d["question"], "answer": d["answer"]})
        return entries


# ============ Main ============

def main(args):
    print(f"Loading {args.input}...")
    entries = load_entries(args.input)
    print(f"Loaded {len(entries)} entries")

    # Resume support for final output
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

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import gc, torch

    cache_path = args.output + ".teacher_cache.jsonl"
    total_batches = (len(entries) + args.batch_size - 1) // args.batch_size

    # ========== Phase 1: Teacher generates responses with logprobs ==========

    if os.path.exists(cache_path):
        print(f"Found teacher cache: {cache_path}, skipping Phase 1")
        # Validate cache line count
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_lines = sum(1 for l in f if l.strip())
        if cache_lines < len(entries):
            print(f"WARNING: Cache has {cache_lines} lines but need {len(entries)}, regenerating...")
            os.remove(cache_path)
        else:
            print(f"Cache has {cache_lines} entries, proceeding to Phase 2")

    if not os.path.exists(cache_path):
        print(f"Loading teacher: {args.teacher_model} (TP={args.tp})")
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
        teacher_llm = LLM(
            model=args.teacher_model,
            tensor_parallel_size=args.tp,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        teacher_sampling = SamplingParams(
            temperature=args.temperature,
            top_p=0.95,
            max_tokens=args.max_gen_tokens,
            n=args.n_samples,
            logprobs=1,
        )

        cache_file = open(cache_path, "w", encoding="utf-8")
        for batch_start in range(0, len(entries), args.batch_size):
            batch = entries[batch_start:batch_start + args.batch_size]
            t0 = time.time()

            teacher_prompts = []
            for entry in batch:
                messages = [
                    {"role": "system", "content": "You are a helpful math tutor. Solve the problem step by step and put your final answer in \\boxed{}."},
                    {"role": "user", "content": entry["question"]},
                ]
                text = teacher_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                teacher_prompts.append(text)

            outputs = teacher_llm.generate(teacher_prompts, teacher_sampling)

            for output in outputs:
                responses = []
                for o in output.outputs:
                    token_lps = get_token_logprobs(o)
                    length = len(token_lps)
                    seq_log_prob = sum(token_lps) if token_lps else None
                    boxed = extract_boxed(o.text)
                    responses.append({
                        "boxed": boxed,
                        "length": length,
                        "seq_log_prob": round(seq_log_prob, 6) if seq_log_prob is not None else None,
                    })
                cache_file.write(json.dumps({"responses": responses}, ensure_ascii=False) + "\n")

            cache_file.flush()
            elapsed = time.time() - t0
            batch_idx = batch_start // args.batch_size + 1
            print(f"[Teacher {batch_idx}/{total_batches}] {min(batch_start + args.batch_size, len(entries))}/{len(entries)} done ({elapsed:.1f}s)")

        cache_file.close()
        print(f"Teacher cache saved: {cache_path}")

        # Release teacher GPU memory
        print("Releasing teacher model...")
        del teacher_llm
        gc.collect()
        torch.cuda.empty_cache()

    # ========== Phase 2: Cluster + probability-weighted SE ==========

    # Load teacher cache
    print("Loading teacher cache...")
    cached_responses = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cached_responses.append(json.loads(line))

    # Extract answers, logprobs, lengths per entry
    all_answers = []
    all_seq_log_probs = []
    all_lengths = []
    skip_flags = []

    for cached in cached_responses:
        answers = []
        seq_log_probs = []
        lengths = []
        null_count = 0

        for resp in cached["responses"]:
            boxed = resp.get("boxed")
            slp = resp.get("seq_log_prob")
            length = resp.get("length", 1)
            if boxed is None:
                null_count += 1
            elif len(boxed) <= 200 and slp is not None:
                answers.append(boxed)
                seq_log_probs.append(slp)
                lengths.append(length)

        if null_count > 2 or len(answers) < 2:
            skip_flags.append(True)
            all_answers.append([])
            all_seq_log_probs.append([])
            all_lengths.append([])
        else:
            skip_flags.append(False)
            all_answers.append(answers)
            all_seq_log_probs.append(seq_log_probs)
            all_lengths.append(lengths)

    print(f"Valid: {sum(1 for s in skip_flags if not s)}, Skipped: {sum(skip_flags)}")

    # Load cluster model
    print(f"Loading cluster model: {args.cluster_model}")
    cluster = SemanticCluster(args.cluster_model, max_model_len=2048, gpu_util=0.80)

    out_file = open(args.output, "a", encoding="utf-8")
    total_processed = 0
    se_sum = 0.0
    skipped_total = 0

    try:
        for batch_start in range(0, len(entries), args.batch_size):
            batch_entries = entries[batch_start:batch_start + args.batch_size]
            batch_answers = all_answers[batch_start:batch_start + args.batch_size]
            batch_slps = all_seq_log_probs[batch_start:batch_start + args.batch_size]
            batch_lengths = all_lengths[batch_start:batch_start + args.batch_size]
            batch_skip = skip_flags[batch_start:batch_start + args.batch_size]
            t0 = time.time()

            # Cluster only valid entries
            valid_answers = [a for a, s in zip(batch_answers, batch_skip) if not s]
            if valid_answers:
                all_cluster_ids = cluster.cluster_batch(valid_answers)
            else:
                all_cluster_ids = []

            # Compute SE and write results
            skipped_in_batch = 0
            valid_idx = 0

            for entry, skip, ans_list, slps, lens in zip(
                batch_entries, batch_skip, batch_answers, batch_slps, batch_lengths
            ):
                if skip:
                    skipped_in_batch += 1
                    skipped_total += 1
                    record = {
                        "question": entry["question"],
                        "answer": entry["answer"],
                        "se": None,
                        "se_weight": None,
                        "se_raw": None,
                        "n_valid_responses": 0,
                    }
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                cluster_ids = all_cluster_ids[valid_idx]
                valid_idx += 1

                se_raw = compute_weighted_entropy(cluster_ids, slps, lens)
                N = len(cluster_ids)
                max_entropy = math.log(N)
                se_normalized = se_raw / max_entropy if max_entropy > 0 else 0.0

                record = {
                    "question": entry["question"],
                    "answer": entry["answer"],
                    "se": round(se_normalized, 4),
                    "se_weight": round(1.0 - se_normalized, 4),
                    "se_raw": round(se_raw, 4),
                    "n_valid_responses": N,
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                se_sum += se_normalized
                total_processed += 1

            out_file.flush()
            elapsed = time.time() - t0
            avg_se = se_sum / total_processed if total_processed > 0 else 0
            batch_idx = batch_start // args.batch_size + 1
            progress = min(batch_start + args.batch_size, len(entries))
            print(
                f"[Cluster {batch_idx}/{total_batches}] {progress}/{len(entries)} | "
                f"Avg SE={avg_se:.3f} | "
                f"Skipped: {skipped_in_batch} | "
                f"{elapsed:.1f}s"
            )

    finally:
        out_file.close()

    avg_se = se_sum / total_processed if total_processed > 0 else 0
    print("=" * 60)
    print(f"Done! Processed: {total_processed}, Skipped: {skipped_total}")
    print(f"Average SE: {avg_se:.3f}, Average weight: {1-avg_se:.3f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SE for augmented math dataset")

    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.80)

    parser.add_argument("--cluster_model", type=str, default="Qwen/Qwen3-4B")

    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_gen_tokens", type=int, default=1024)

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()
    main(args)
