"""
Script 2: Compute probability-weighted semantic entropy from teacher responses.

Pipeline:
  1. Load intermediate file from uncertainty_generate.py
  2. Extract \\boxed{} answers from teacher responses
  3. Cluster answers: exact string match + 3B LLM judge for ambiguous pairs
  4. Accumulate sequence probabilities per cluster: p(c) = sum(p_i) / Z
  5. Compute SE = -sum(p(c) * log(p(c)))
  6. Output dataset with "se" and "se_weight" fields

Usage:
  python uncertainty_calculation.py \
    --cluster_model Qwen/Qwen2.5-3B-Instruct \
    --input teacher_responses.jsonl \
    --output evolved_with_se.jsonl
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict


# ============ Hybrid Clustering: string match + LLM judge ============

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

    def __init__(self, model_name, max_model_len=2048, gpu_util=0.30):
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=3)
        print(f"[Cluster] Ready.")

    def _build_pair_prompt(self, ans_a, ans_b):
        messages = [
            {"role": "system", "content": PAIRWISE_SYSTEM},
            {"role": "user", "content": PAIRWISE_TEMPLATE.format(
                ans_a=ans_a, ans_b=ans_b
            )},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def cluster_batch(self, all_answers: list[list[str]]) -> list[list[int]]:
        """Cluster answers for multiple prompts via Union-Find.

        Args:
            all_answers: list of answer lists, one per prompt

        Returns:
            list of cluster ID lists. result[i][j] = cluster id for
            the j-th answer of the i-th prompt.
        """
        # Build pair prompts ONLY for pairs with different strings
        pair_prompts = []
        pair_meta = []
        for prompt_idx, answers in enumerate(all_answers):
            N = len(answers)
            for i in range(N):
                for j in range(i + 1, N):
                    if answers[i] != answers[j]:
                        pair_prompts.append(
                            self._build_pair_prompt(answers[i], answers[j])
                        )
                        pair_meta.append((prompt_idx, i, j))

        # Batch judge ambiguous pairs
        llm_verdicts = {}
        if pair_prompts:
            outputs = self.llm.generate(pair_prompts, self.sampling_params)
            for (pidx, i, j), output in zip(pair_meta, outputs):
                text = output.outputs[0].text.strip().upper()
                llm_verdicts[(pidx, i, j)] = text.startswith("YES")

        # Union-Find per prompt
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
    Then cluster probability: p(c) = sum(exp(norm_log_prob_i)) / Z
    where Z = sum over all i ensures normalization.
    Entropy: SE = -sum(p(c) * log(p(c)))

    Args:
        cluster_ids: cluster assignment for each response
        seq_log_probs: log(product of token probs) for each response
        lengths: token count for each response (for length normalization)

    Returns:
        entropy: -sum(p(c) * log(p(c)))
    """
    N = len(cluster_ids)
    if N <= 1:
        return 0.0

    # Length-normalize: log_prob / length
    norm_log_probs = [
        slp / length if length > 0 else slp
        for slp, length in zip(seq_log_probs, lengths)
    ]

    # Normalize across all responses: log(p_i / Z) = norm_log_prob_i - log(Z)
    log_Z = logsumexp(norm_log_probs)

    # Accumulate normalized log probs per cluster via logsumexp
    cluster_log_probs = defaultdict(list)
    for cid, nlp in zip(cluster_ids, norm_log_probs):
        cluster_log_probs[cid].append(nlp - log_Z)

    # p(c) = sum of normalized probs in cluster, SE = -sum(p(c) * log(p(c)))
    entropy = 0.0
    for log_ps in cluster_log_probs.values():
        log_pc = logsumexp(log_ps)  # log(p(c))
        pc = math.exp(log_pc)
        if pc > 0:
            entropy -= pc * log_pc

    return entropy


def main(args):
    # ---- Load intermediate file ----
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

    # ---- Init cluster model ----
    cluster = SemanticCluster(
        args.cluster_model,
        max_model_len=2048,
        gpu_util=args.gpu_memory_utilization,
    )

    # ---- Process in batches ----
    out_file = open(args.output, "a", encoding="utf-8")
    total_processed = 0
    se_sum = 0.0
    skipped_total = 0

    try:
        for batch_start in range(0, len(entries), args.batch_size):
            batch = entries[batch_start:batch_start + args.batch_size]
            t0 = time.time()

            # Extract boxed answers, seq_log_probs, and lengths
            all_answers = []
            all_seq_log_probs = []
            all_lengths = []
            skip_flags = []

            for entry in batch:
                responses = entry.get("teacher_responses", [])
                answers = []
                seq_log_probs = []
                lengths = []

                null_count = 0
                for resp in responses:
                    boxed = resp.get("boxed")
                    slp = resp.get("seq_log_prob")
                    length = resp.get("length", 1)
                    if boxed is None:
                        null_count += 1
                    elif len(boxed) <= 200 and slp is not None:
                        answers.append(boxed)
                        seq_log_probs.append(slp)
                        lengths.append(length)

                # Skip if too many null boxed (>2) or too few valid answers
                if null_count > 2 or len(answers) < 2:
                    skip_flags.append(True)
                else:
                    skip_flags.append(False)
                    all_answers.append(answers)
                    all_seq_log_probs.append(seq_log_probs)
                    all_lengths.append(lengths)

            # Cluster valid entries
            if all_answers:
                all_cluster_ids = cluster.cluster_batch(all_answers)
            else:
                all_cluster_ids = []

            # Compute SE and write results
            skipped_in_batch = 0
            valid_iter = iter(range(len(all_cluster_ids)))

            for entry, skip in zip(batch, skip_flags):
                out_entry = {
                    k: v for k, v in entry.items()
                    if k != "teacher_responses"
                }

                if skip:
                    skipped_in_batch += 1
                    skipped_total += 1
                    # Still write to output (with null se) so resume stays aligned
                    out_entry["se"] = None
                    out_entry["se_weight"] = None
                    out_entry["se_raw"] = None
                    out_entry["n_valid_responses"] = 0
                    out_file.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
                    continue

                vi = next(valid_iter)
                cluster_ids = all_cluster_ids[vi]
                seq_log_probs = all_seq_log_probs[vi]
                lengths = all_lengths[vi]

                se = compute_weighted_entropy(cluster_ids, seq_log_probs, lengths)

                # Normalize by max entropy log(N)
                N = len(cluster_ids)
                max_entropy = math.log(N)
                se_normalized = se / max_entropy if max_entropy > 0 else 0.0

                out_entry["se"] = round(se_normalized, 4)
                out_entry["se_weight"] = round(1.0 - se_normalized, 4)
                out_entry["se_raw"] = round(se, 4)
                out_entry["n_valid_responses"] = N

                out_file.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
                se_sum += se_normalized
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
                f"Batch: {elapsed:.1f}s"
            )

    finally:
        out_file.close()

    avg_se = se_sum / total_processed if total_processed > 0 else 0
    print("=" * 60)
    print(f"Done! Processed: {total_processed}, Skipped: {skipped_total}")
    print(f"Average SE: {avg_se:.3f}, Average weight: {1-avg_se:.3f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute probability-weighted semantic entropy"
    )

    # Cluster model (3B judge)
    parser.add_argument("--cluster_model", type=str,
                        default="Qwen/Qwen3-4B")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.30)

    # IO
    parser.add_argument("--input", type=str, default="teacher_responses.jsonl",
                        help="Intermediate file from uncertainty_generate.py")
    parser.add_argument("--output", type=str, default="evolved_with_se.jsonl")
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()
    main(args)
