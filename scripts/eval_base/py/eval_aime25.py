"""
Evaluate on AIME 2025 (opencompass/AIME2025).

Usage:
  python eval_aime25.py --model Qwen/Qwen3-1.7B --mode greedy
  python eval_aime25.py --model Qwen/Qwen3-8B --mode average --n_samples 16
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from math_eval_utils import extract_answer, strip_string, math_equal, MATH_FEW_SHOT_EXAMPLES

HF_REPO = "opencompass/AIME2025"
SPLIT = "test"
PROBLEM_KEY = "question"
ANSWER_KEY = "answer"
DATASET_NAME = "aime25"

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prompts(problems: list[str], tokenizer, num_shots: int = 4) -> list[str]:
    few_shot = MATH_FEW_SHOT_EXAMPLES[:num_shots]
    prompts = []
    for problem in problems:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for q, a in few_shot:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": problem})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if "<think>" not in prompt:
            prompt += "<think>\n"
        prompts.append(prompt)
    return prompts


def majority_vote(answers: list[str]) -> str:
    normalized = [strip_string(a) for a in answers]
    counter = Counter(normalized)
    return counter.most_common(1)[0][0] if counter else ""


def evaluate(args):
    ds1 = load_dataset(HF_REPO, "AIME2025-I", split=SPLIT)
    ds2 = load_dataset(HF_REPO, "AIME2025-II", split=SPLIT)
    ds = concatenate_datasets([ds1, ds2])
    problems = [row[PROBLEM_KEY] for row in ds]
    answers = [str(row[ANSWER_KEY]) for row in ds]
    print(f"Dataset: {DATASET_NAME} ({HF_REPO}), {len(problems)} problems (I={len(ds1)}, II={len(ds2)})")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = build_prompts(problems, tokenizer, num_shots=args.num_shots)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    if args.mode == "greedy":
        sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, n=1)
    else:
        sampling_params = SamplingParams(
            temperature=args.temperature, top_p=args.top_p,
            max_tokens=args.max_tokens, n=args.n_samples,
        )

    print(f"Mode: {args.mode}, n_samples: {sampling_params.n}, max_tokens: {args.max_tokens}")
    print("Generating responses...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start_time
    print(f"Generation completed in {elapsed:.1f}s")

    results = []
    correct_count = 0

    for i, output in enumerate(outputs):
        gt = answers[i]
        gt_stripped = strip_string(gt)

        if args.mode == "greedy":
            response = output.outputs[0].text
            predicted = extract_answer(response)
            is_correct = math_equal(predicted, gt_stripped)
            result = {"idx": i, "problem": problems[i][:200], "ground_truth": gt,
                      "predicted": predicted, "correct": is_correct}
        else:
            all_extracted = [extract_answer(s.text) for s in output.outputs]
            per_sample = [{"extracted": e, "correct": math_equal(e, gt_stripped)} for e in all_extracted]
            n_correct_samples = sum(1 for s in per_sample if s["correct"])

            if args.mode == "majority":
                voted = majority_vote(all_extracted)
                is_correct = math_equal(voted, gt_stripped)
                result = {"idx": i, "problem": problems[i][:200], "ground_truth": gt,
                          "voted_answer": voted, "correct": is_correct,
                          "n_correct_samples": n_correct_samples, "n_total_samples": len(per_sample)}
            else:
                sample_acc = n_correct_samples / len(per_sample) if per_sample else 0.0
                is_correct = n_correct_samples > 0
                result = {"idx": i, "problem": problems[i][:200], "ground_truth": gt,
                          "correct": is_correct, "n_correct_samples": n_correct_samples,
                          "n_total_samples": len(per_sample), "sample_accuracy": sample_acc}

        results.append(result)
        if is_correct:
            correct_count += 1

    if args.mode == "average":
        accuracy = sum(r["sample_accuracy"] for r in results) / len(results) if results else 0.0
    else:
        accuracy = correct_count / len(results) if results else 0.0

    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Dataset: {DATASET_NAME} ({len(results)} problems)")
    mode_str = args.mode
    if args.mode != "greedy":
        mode_str += f" (n={args.n_samples}, temp={args.temperature}, top_p={args.top_p})"
    print(f"Mode: {mode_str}")
    if args.mode == "average":
        print(f"Avg@{args.n_samples} Accuracy: {accuracy:.1%}")
    else:
        print(f"Accuracy: {correct_count}/{len(results)} = {accuracy:.1%}")
    print(f"{'='*60}")

    for r in results:
        status = "O" if r["correct"] else "X"
        if args.mode == "majority":
            print(f"  [{status}] #{r['idx']:2d}  gt={r['ground_truth']}  voted={r['voted_answer']}  "
                  f"({r['n_correct_samples']}/{r['n_total_samples']} samples correct)")
        elif args.mode == "average":
            print(f"  [{status}] #{r['idx']:2d}  gt={r['ground_truth']}  "
                  f"{r['n_correct_samples']}/{r['n_total_samples']} = {r['sample_accuracy']:.1%}")
        else:
            print(f"  [{status}] #{r['idx']:2d}  gt={r['ground_truth']}  pred={r['predicted']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.replace("/", "_")
    output_path = output_dir / f"{model_short}_{DATASET_NAME}_{args.mode}.json"

    summary = {
        "model": args.model, "dataset": DATASET_NAME, "hf_repo": HF_REPO,
        "mode": args.mode, "n_samples": args.n_samples if args.mode != "greedy" else 1,
        "max_tokens": args.max_tokens, "accuracy": accuracy,
        "correct": correct_count, "total": len(results),
        "elapsed_seconds": elapsed, "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    txt_dir = Path(__file__).resolve().parent.parent / "results"
    txt_dir.mkdir(parents=True, exist_ok=True)
    txt_path = txt_dir / f"{model_short}_{DATASET_NAME}_{args.mode}.txt"
    with open(txt_path, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {DATASET_NAME} ({HF_REPO})\n")
        f.write(f"Mode: {mode_str}\n")
        f.write(f"Accuracy: {correct_count}/{len(results)} = {accuracy:.1%}\n")
        f.write(f"Time: {elapsed:.1f}s\n")
        f.write(f"{'='*60}\n")
        for r in results:
            status = "O" if r["correct"] else "X"
            if args.mode == "majority":
                f.write(f"[{status}] #{r['idx']:2d}  gt={r['ground_truth']}  voted={r['voted_answer']}  "
                        f"({r['n_correct_samples']}/{r['n_total_samples']} samples correct)\n")
            elif args.mode == "average":
                f.write(f"[{status}] #{r['idx']:2d}  gt={r['ground_truth']}  "
                        f"{r['n_correct_samples']}/{r['n_total_samples']} = {r['sample_accuracy']:.1%}\n")
            else:
                f.write(f"[{status}] #{r['idx']:2d}  gt={r['ground_truth']}  pred={r['predicted']}\n")

    print(f"\nResults saved to {output_path}")
    print(f"Results saved to {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on AIME 2025")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, default="greedy", choices=["greedy", "majority", "average"])
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=38000)
    parser.add_argument("--max_model_len", type=int, default=40960)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--num_shots", type=int, default=4, help="Number of few-shot CoT examples (0 for zero-shot)")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    args = parser.parse_args()
    evaluate(args)
