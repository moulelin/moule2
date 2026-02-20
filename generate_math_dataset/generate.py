"""
Evol-Instruct style math problem evolution using vLLM.

Pipeline:
  1. Load seed problems from multiple datasets (different difficulty levels)
  2. For each seed, pick evolution strategy based on difficulty tier
  3. Prompt the model to evolve the problem
  4. Multi-round evolution (more rounds for competition seeds to amplify them)
  5. Save results as JSONL (append-safe for resume)

Usage:
  python generate.py \
    --model Qwen/QwQ-32B \
    --tp 4 \
    --seed_config seed_datasets.json \
    --output evolved_math.jsonl
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from evol_prompts import STRATEGIES, FOLLOWUP_STRATEGIES


# ============================================================
# Seed dataset registry
# ============================================================
SEED_REGISTRY = {
    # ---- Easy / General (大量种子，少轮进化) ----
    "scalequest": {
        "hf_id": "dyyyyyyyy/ScaleQuest-Math",
        "key": "query",
        "split": "train",
        "difficulty": "easy",
        "max_seeds": 15000,
        "evol_rounds": 1,
        "description": "1M general math problems (ScaleQuest, ICLR 2025)",
    },

    # ---- Medium (竞赛预备级) ----
    "numinamath": {
        "hf_id": "AI-MO/NuminaMath-CoT",
        "key": "problem",
        "split": "train",
        "difficulty": "medium",
        "max_seeds": 10000,
        "evol_rounds": 2,
        "description": "860K math problems with CoT (NuminaMath)",
    },
    "math": {
        "hf_id": "hendrycks/competition_math",
        "key": "problem",
        "split": "train",
        "difficulty": "medium",
        "max_seeds": -1,  # use all ~12.5K
        "evol_rounds": 2,
        "description": "12.5K competition math (MATH, Hendrycks)",
    },
    "omni_math": {
        "hf_id": "KbsdJames/Omni-MATH",
        "key": "problem",
        "split": "test",
        "difficulty": "hard",
        "max_seeds": -1,  # use all ~4.4K
        "evol_rounds": 3,
        "description": "4.4K Olympiad-level problems (Omni-MATH)",
    },
    "olympiadbench": {
        "hf_id": "Hothan/OlympiadBench",
        "key": "question",
        "split": "train",
        "difficulty": "hard",
        "max_seeds": 5000,
        "evol_rounds": 3,
        "description": "8.4K Olympiad math/physics (OlympiadBench)",
    },

    # ---- Competition (极少种子，多轮进化放大) ----
    "aime24": {
        "hf_id": "HuggingFaceH4/aime_2024",
        "key": "problem",
        "split": "train",
        "difficulty": "competition",
        "max_seeds": -1,  # all 30
        "evol_rounds": 5,
        "description": "30 AIME 2024 problems",
    },
    "aime25": {
        "hf_id": "MathArena/aime_2025",
        "key": "problem",
        "split": "train",
        "difficulty": "competition",
        "max_seeds": -1,
        "evol_rounds": 5,
        "description": "AIME 2025 problems",
    },
    "hmmt_feb25": {
        "hf_id": "MathArena/hmmt_feb_2025",
        "key": "problem",
        "split": "train",
        "difficulty": "competition",
        "max_seeds": -1,
        "evol_rounds": 5,
        "description": "HMMT February 2025 problems",
    },
    "hmmt_nov25": {
        "hf_id": "MathArena/hmmt_nov_2025",
        "key": "problem",
        "split": "train",
        "difficulty": "competition",
        "max_seeds": -1,
        "evol_rounds": 5,
        "description": "HMMT November 2025 problems",
    },
    "amo_bench": {
        "hf_id": "meituan-longcat/AMO-Bench",
        "key": "prompt",
        "split": "train",
        "difficulty": "competition",
        "max_seeds": -1,  # all 50
        "evol_rounds": 5,
        "description": "50 IMO+ level problems (AMO-Bench, Meituan)",
    },
}

# Strategy weights per difficulty tier
TIER_STRATEGY_WEIGHTS = {
    "easy": {
        "harder": 0.30, "rewrite": 0.20, "algebraize": 0.15,
        "apply": 0.15, "compose": 0.10, "competition": 0.10,
    },
    "medium": {
        "harder": 0.25, "rewrite": 0.15, "algebraize": 0.15,
        "apply": 0.15, "compose": 0.15, "competition": 0.15,
    },
    "hard": {
        "harder": 0.20, "rewrite": 0.15, "algebraize": 0.15,
        "apply": 0.10, "compose": 0.20, "competition": 0.20,
    },
    "competition": {
        "harder": 0.15, "rewrite": 0.20, "algebraize": 0.15,
        "apply": 0.10, "compose": 0.20, "competition": 0.20,
    },
}


def load_seeds_from_registry(dataset_names, local_files=None):
    """Load seeds from multiple datasets in the registry.

    Returns: list of (problem_text, source_name, difficulty)
    """
    from datasets import load_dataset

    all_seeds = []

    for name in dataset_names:
        if name not in SEED_REGISTRY:
            print(f"WARNING: Unknown dataset '{name}', skipping")
            continue

        info = SEED_REGISTRY[name]
        print(f"  Loading {name}: {info['description']}...")

        try:
            ds = load_dataset(info["hf_id"], split=info["split"])
        except Exception as e:
            print(f"  WARNING: Failed to load {info['hf_id']}: {e}")
            # Try alternative split names
            for alt_split in ["test", "train", "validation"]:
                if alt_split == info["split"]:
                    continue
                try:
                    ds = load_dataset(info["hf_id"], split=alt_split)
                    print(f"    -> Loaded with split='{alt_split}' instead")
                    break
                except Exception:
                    continue
            else:
                print(f"  SKIPPED: {name}")
                continue

        # Extract problems
        key = info["key"]
        count = 0
        for row in ds:
            if key in row and row[key]:
                text = str(row[key]).strip()
                if len(text) >= 10:  # skip trivially short
                    all_seeds.append({
                        "text": text,
                        "source": name,
                        "difficulty": info["difficulty"],
                        "evol_rounds": info["evol_rounds"],
                    })
                    count += 1
                    if info["max_seeds"] > 0 and count >= info["max_seeds"]:
                        break

        print(f"    -> Got {count} seeds (difficulty={info['difficulty']}, rounds={info['evol_rounds']})")

    # Load local JSONL files if specified
    if local_files:
        for fpath in local_files:
            if not os.path.exists(fpath):
                print(f"  WARNING: Local file not found: {fpath}")
                continue
            count = 0
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    text = obj.get("query") or obj.get("problem") or obj.get("question", "")
                    if len(text) >= 10:
                        all_seeds.append({
                            "text": text,
                            "source": f"local:{os.path.basename(fpath)}",
                            "difficulty": "medium",
                            "evol_rounds": 2,
                        })
                        count += 1
            print(f"  Loaded {count} seeds from {fpath}")

    return all_seeds


def pick_strategy(difficulty="medium", exclude=None):
    """Pick evolution strategy based on difficulty tier."""
    weights = TIER_STRATEGY_WEIGHTS.get(difficulty, TIER_STRATEGY_WEIGHTS["medium"])
    strategies = list(weights.keys())
    w = list(weights.values())

    if exclude:
        filtered = [(s, wt) for s, wt in zip(strategies, w) if s != exclude]
        strategies, w = zip(*filtered)
        strategies, w = list(strategies), list(w)

    total = sum(w)
    w = [x / total for x in w]
    return random.choices(strategies, weights=w, k=1)[0]


def build_prompts(seeds_text, strategies_list):
    """Build vLLM prompt list from seeds and strategies."""
    prompts = []
    for seed, strategy_name in zip(seeds_text, strategies_list):
        system_prompt, user_template = STRATEGIES[strategy_name]
        user_msg = user_template.format(problem=seed)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        prompts.append(messages)
    return prompts


def extract_problem(text):
    """Extract the evolved problem from model output."""
    text = text.strip()

    # For reasoning models (QwQ, R1), strip <think>...</think> blocks
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Remove common preambles
    prefixes_to_remove = [
        "Here is the new problem:",
        "Here is the rewritten problem:",
        "Here is the evolved problem:",
        "Here is the transformed problem:",
        "Here's the new problem:",
        "Here's the transformed problem:",
        "New problem:",
        "**Problem:**",
        "**New Problem:**",
        "**Problem Statement:**",
    ]
    for prefix in prefixes_to_remove:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    # Remove trailing solution if model accidentally included one
    solution_markers = [
        "\n**Solution", "\nSolution:", "\n**Answer", "\nAnswer:",
        "\n**Hint", "\nHint:", "\n---\n",
    ]
    for marker in solution_markers:
        idx = text.find(marker)
        if idx > 50:
            text = text[:idx].strip()

    return text


def get_existing_count(output_path):
    """Count existing lines for resume support."""
    if not os.path.exists(output_path):
        return 0
    count = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def generate_batch(llm, tokenizer, prompts, sampling_params):
    """Generate responses for a batch of chat prompts using vLLM."""
    formatted = []
    for msgs in prompts:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        formatted.append(text)

    outputs = llm.generate(formatted, sampling_params)
    return [output.outputs[0].text for output in outputs]


def main(args):
    random.seed(args.seed)

    # ---- Determine which datasets to load ----
    if args.datasets == "all":
        dataset_names = list(SEED_REGISTRY.keys())
    else:
        dataset_names = [d.strip() for d in args.datasets.split(",")]

    # ---- Load all seeds ----
    print("=" * 60)
    print("Loading seed problems from multiple datasets...")
    local_files = args.local_seeds.split(",") if args.local_seeds else None
    all_seeds = load_seeds_from_registry(dataset_names, local_files)

    # Shuffle
    random.shuffle(all_seeds)

    # ---- Print statistics ----
    from collections import Counter
    src_counts = Counter(s["source"] for s in all_seeds)
    diff_counts = Counter(s["difficulty"] for s in all_seeds)
    print(f"\nTotal seeds: {len(all_seeds)}")
    print("By source:")
    for src, cnt in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")
    print("By difficulty:")
    for diff, cnt in sorted(diff_counts.items()):
        print(f"  {diff}: {cnt}")

    # ---- Resume support ----
    existing = get_existing_count(args.output)
    if existing > 0:
        print(f"\nFound {existing} existing entries, resuming...")
        all_seeds = all_seeds[existing:]
        if not all_seeds:
            print("All seeds already processed!")
            return

    print(f"\nSeeds to process: {len(all_seeds)}")

    # ---- Init vLLM ----
    print("=" * 60)
    print(f"Loading model: {args.model} (TP={args.tp})")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=1.05,
    )

    # ---- Process in batches ----
    print("=" * 60)
    print("Starting evolution...")
    total_generated = 0
    batch_size = args.batch_size

    out_file = open(args.output, "a", encoding="utf-8")

    try:
        for batch_start in range(0, len(all_seeds), batch_size):
            batch = all_seeds[batch_start:batch_start + batch_size]
            t0 = time.time()

            # ---- Round 1 ----
            texts = [s["text"] for s in batch]
            diffs = [s["difficulty"] for s in batch]
            strategies_r1 = [pick_strategy(d) for d in diffs]
            prompts_r1 = build_prompts(texts, strategies_r1)
            results_r1 = generate_batch(llm, tokenizer, prompts_r1, sampling_params)

            # Collect round 1 results, grouped by max evol_rounds
            evolved = []  # (problem, strategy, seed_info)
            for seed_info, strategy, result in zip(batch, strategies_r1, results_r1):
                problem = extract_problem(result)
                if len(problem) < 20:
                    continue

                entry = {
                    "query": problem,
                    "source": seed_info["source"],
                    "difficulty": seed_info["difficulty"],
                    "strategy": strategy,
                    "round": 1,
                }
                out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total_generated += 1
                evolved.append((problem, strategy, seed_info))

            # ---- Round 2+ ----
            current = evolved
            max_round = max(s["evol_rounds"] for s in batch) if batch else 1

            for rnd in range(2, max_round + 1):
                if not current:
                    break

                # Only evolve seeds that want this many rounds
                to_evolve = [(p, s, info) for p, s, info in current if info["evol_rounds"] >= rnd]
                if not to_evolve:
                    break

                probs = [p for p, _, _ in to_evolve]
                prev_strats = [s for _, s, _ in to_evolve]
                infos = [info for _, _, info in to_evolve]
                strategies_rn = [
                    random.choice([s for s in FOLLOWUP_STRATEGIES if s != prev])
                    for prev in prev_strats
                ]
                prompts_rn = build_prompts(probs, strategies_rn)
                results_rn = generate_batch(llm, tokenizer, prompts_rn, sampling_params)

                next_round = []
                for _, strategy, result, info in zip(probs, strategies_rn, results_rn, infos):
                    problem = extract_problem(result)
                    if len(problem) < 20:
                        continue

                    entry = {
                        "query": problem,
                        "source": info["source"],
                        "difficulty": info["difficulty"],
                        "strategy": strategy,
                        "round": rnd,
                    }
                    out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    total_generated += 1
                    next_round.append((problem, strategy, info))

                current = next_round

            out_file.flush()
            elapsed = time.time() - t0
            progress = min(batch_start + batch_size, len(all_seeds))
            print(
                f"[{progress}/{len(all_seeds)}] "
                f"Generated {total_generated} total | "
                f"Batch: {elapsed:.1f}s | "
                f"Speed: {len(batch)/max(elapsed,0.1):.1f} seeds/s"
            )

    finally:
        out_file.close()

    print("=" * 60)
    print(f"Done! Total generated: {total_generated}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evol-Instruct Math Problem Generator")

    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct",
                        help="Model for evolution")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--enforce_eager", action="store_true", default=False)

    # Seed datasets
    parser.add_argument("--datasets", type=str, default="all",
                        help="Comma-separated dataset names from registry, or 'all'")
    parser.add_argument("--local_seeds", type=str, default=None,
                        help="Comma-separated local JSONL files as extra seeds")

    # Generation
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Max tokens for evolved problem output")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for vLLM generation")

    # Output
    parser.add_argument("--output", type=str, default="evolved_math_problems.jsonl")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
