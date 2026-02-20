"""
Filter and deduplicate evolved math problems.

Steps:
  1. Remove too-short or too-long problems
  2. Remove problems that look like solutions (contain "therefore", "answer is", etc.)
  3. Remove near-duplicates via MinHash LSH
  4. Output clean JSONL

Usage:
  python filter_dedup.py \
    --input evolved_math_problems.jsonl \
    --output evolved_math_filtered.jsonl \
    --min_len 30 --max_len 2000
"""

import argparse
import json
import re
import hashlib
from collections import defaultdict


def is_likely_solution(text):
    """Check if text looks like a solution rather than a problem."""
    lower = text.lower()
    solution_indicators = [
        "therefore, the answer is",
        "the final answer is",
        "\\boxed{",
        "hence, the answer",
        "so the answer is",
        "the solution is",
        "we get the answer",
        "step 1:",
        "step 2:",
    ]
    count = sum(1 for ind in solution_indicators if ind in lower)
    return count >= 2  # Multiple solution indicators = likely a solution


def is_likely_problem(text):
    """Check if text looks like a math problem."""
    lower = text.lower()
    problem_indicators = [
        "find", "compute", "calculate", "determine", "prove",
        "how many", "what is", "evaluate", "solve", "show that",
        "if ", "given ", "let ", "suppose",
        "?",  # Questions often end with ?
    ]
    return any(ind in lower for ind in problem_indicators)


def ngram_shingles(text, n=3):
    """Get character n-gram shingles for near-dedup."""
    text = re.sub(r'\s+', ' ', text.lower().strip())
    return set(text[i:i+n] for i in range(len(text) - n + 1))


def jaccard_similarity(set_a, set_b):
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def minhash_signature(shingles, num_hashes=128):
    """Compute MinHash signature for a set of shingles."""
    sig = []
    for i in range(num_hashes):
        min_hash = float('inf')
        for shingle in shingles:
            h = int(hashlib.md5(f"{i}_{shingle}".encode()).hexdigest(), 16)
            min_hash = min(min_hash, h)
        sig.append(min_hash)
    return tuple(sig)


def lsh_buckets(signature, bands=16):
    """Split MinHash signature into LSH bands for approximate matching."""
    rows_per_band = len(signature) // bands
    buckets = []
    for b in range(bands):
        start = b * rows_per_band
        band_slice = signature[start:start + rows_per_band]
        bucket_hash = hash(band_slice)
        buckets.append((b, bucket_hash))
    return buckets


def main(args):
    # ---- Load ----
    print(f"Loading {args.input}...")
    entries = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"Loaded {len(entries)} entries")

    # ---- Length filter ----
    before = len(entries)
    entries = [e for e in entries if args.min_len <= len(e["query"]) <= args.max_len]
    print(f"Length filter: {before} -> {len(entries)} ({before - len(entries)} removed)")

    # ---- Solution filter ----
    before = len(entries)
    entries = [e for e in entries if not is_likely_solution(e["query"])]
    print(f"Solution filter: {before} -> {len(entries)} ({before - len(entries)} removed)")

    # ---- Problem check (optional soft filter) ----
    if args.require_problem_format:
        before = len(entries)
        entries = [e for e in entries if is_likely_problem(e["query"])]
        print(f"Problem format filter: {before} -> {len(entries)} ({before - len(entries)} removed)")

    # ---- Near-dedup via MinHash LSH ----
    if args.dedup:
        print("Running MinHash LSH deduplication...")
        # Build signatures
        signatures = []
        shingle_sets = []
        for e in entries:
            shingles = ngram_shingles(e["query"], n=args.ngram_size)
            sig = minhash_signature(shingles, num_hashes=args.num_hashes)
            signatures.append(sig)
            shingle_sets.append(shingles)

        # LSH bucketing
        bucket_map = defaultdict(list)  # bucket -> [indices]
        for idx, sig in enumerate(signatures):
            for bucket_key in lsh_buckets(sig, bands=args.lsh_bands):
                bucket_map[bucket_key].append(idx)

        # Find duplicates
        to_remove = set()
        checked_pairs = set()
        for indices in bucket_map.values():
            if len(indices) < 2:
                continue
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    a, b = indices[i], indices[j]
                    if a in to_remove or b in to_remove:
                        continue
                    pair = (min(a, b), max(a, b))
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)

                    sim = jaccard_similarity(shingle_sets[a], shingle_sets[b])
                    if sim >= args.dedup_threshold:
                        to_remove.add(b)  # Remove later one

        before = len(entries)
        entries = [e for i, e in enumerate(entries) if i not in to_remove]
        print(f"Dedup: {before} -> {len(entries)} ({len(to_remove)} duplicates removed)")

    # ---- Save ----
    print(f"Saving {len(entries)} problems to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # ---- Also save query-only version for training ----
    train_output = args.output.replace(".jsonl", "_train.jsonl")
    with open(train_output, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps({"query": e["query"]}, ensure_ascii=False) + "\n")
    print(f"Training-ready (query only): {train_output}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and deduplicate evolved math problems")

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    # Length filters
    parser.add_argument("--min_len", type=int, default=30,
                        help="Minimum problem length in characters")
    parser.add_argument("--max_len", type=int, default=2000,
                        help="Maximum problem length in characters")

    # Content filters
    parser.add_argument("--require_problem_format", action="store_true", default=False,
                        help="Require problem to contain question-like keywords")

    # Dedup
    parser.add_argument("--dedup", action="store_true", default=True,
                        help="Enable near-duplicate removal")
    parser.add_argument("--no_dedup", action="store_false", dest="dedup")
    parser.add_argument("--dedup_threshold", type=float, default=0.7,
                        help="Jaccard similarity threshold for dedup (0.7 = quite similar)")
    parser.add_argument("--ngram_size", type=int, default=3)
    parser.add_argument("--num_hashes", type=int, default=128)
    parser.add_argument("--lsh_bands", type=int, default=16)

    args = parser.parse_args()
    main(args)
