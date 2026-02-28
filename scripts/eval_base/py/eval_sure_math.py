"""
Evaluate on SURE-Math test set (200 held-out curated problems).

The test set is a local JSONL file with fields: question, answer.

Usage:
  python eval_sure_math.py --model Qwen/Qwen3-1.7B --mode greedy --input ../sure_math_test.jsonl
  python eval_sure_math.py --model Qwen/Qwen3-8B --mode average --n_samples 16 --input ../sure_math_test.jsonl
"""

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from math_eval_utils import extract_answer, strip_string, math_equal, MATH_FEW_SHOT_EXAMPLES

DATASET_NAME = "sure_math"


def normalize_gt(answer: str) -> str:
    """Normalize ground truth answer through the same pipeline as model output.

    Both model output and ground truth go through identical preprocessing:
      1. Strip $...$ delimiters
      2. extract_answer(): handles \\boxed{}, "the answer is", last number fallback
      3. strip_string(): normalize LaTeX (fracs, sqrt, spacing, units, etc.)
    Then math_equal() handles numeric/symbolic/matrix comparison.
    """
    answer = answer.strip()
    # Remove $...$ wrapping
    if answer.startswith('$') and answer.endswith('$'):
        answer = answer[1:-1].strip()
    # Apply same extract_answer pipeline as model output
    extracted = extract_answer(answer, use_last_number=False)
    if not extracted:
        return strip_string(answer)
    return strip_string(extracted)

def _parse_interval(s: str):
    """Parse interval notation like (-inf, 4.5] or [1, 3) into (lo, lo_closed, hi, hi_closed)."""
    s = s.replace('\\infty', 'oo').replace('+oo', 'oo').replace('-oo', '-oo')
    m = re.match(r'^\s*([(\[])\s*(.+?)\s*,\s*(.+?)\s*([)\]])\s*$', s)
    if not m:
        return None
    lo_closed = m.group(1) == '['
    hi_closed = m.group(4) == ']'
    lo_str = m.group(2).strip()
    hi_str = m.group(3).strip()
    return lo_str, lo_closed, hi_str, hi_closed


def _parse_inequality(s: str):
    """Parse inequality like x≤4.5 or a<3 into interval form."""
    s = s.replace('\\le', '≤').replace('\\ge', '≥').replace('\\leq', '≤').replace('\\geq', '≥')
    s = s.replace('\\leqslant', '≤').replace('\\geqslant', '≥')
    # x ≤ a  →  (-oo, a]
    m = re.match(r'^[a-z]\s*≤\s*(.+)$', s)
    if m:
        return '-oo', False, m.group(1).strip(), True
    # x < a  →  (-oo, a)
    m = re.match(r'^[a-z]\s*<\s*(.+)$', s)
    if m:
        return '-oo', False, m.group(1).strip(), False
    # x ≥ a  →  [a, oo)
    m = re.match(r'^[a-z]\s*≥\s*(.+)$', s)
    if m:
        return m.group(1).strip(), True, 'oo', False
    # x > a  →  (a, oo)
    m = re.match(r'^[a-z]\s*>\s*(.+)$', s)
    if m:
        return m.group(1).strip(), False, 'oo', False
    # a ≤ x ≤ b  →  [a, b]
    m = re.match(r'^(.+?)\s*≤\s*[a-z]\s*≤\s*(.+)$', s)
    if m:
        return m.group(1).strip(), True, m.group(2).strip(), True
    # a < x < b  →  (a, b)
    m = re.match(r'^(.+?)\s*<\s*[a-z]\s*<\s*(.+)$', s)
    if m:
        return m.group(1).strip(), False, m.group(2).strip(), False
    # a ≤ x < b  →  [a, b)
    m = re.match(r'^(.+?)\s*≤\s*[a-z]\s*<\s*(.+)$', s)
    if m:
        return m.group(1).strip(), True, m.group(2).strip(), False
    # a < x ≤ b  →  (a, b]
    m = re.match(r'^(.+?)\s*<\s*[a-z]\s*≤\s*(.+)$', s)
    if m:
        return m.group(1).strip(), False, m.group(2).strip(), True
    return None


def _to_interval(s: str):
    """Try to parse string as interval (from interval notation or inequality)."""
    result = _parse_interval(s)
    if result:
        return result
    return _parse_inequality(s)


def _parse_chain(s: str):
    """Parse inequality chain like 'a < b < c' or 'c > b > a' into sorted form."""
    s = s.replace('\\le', '≤').replace('\\ge', '≥').replace('\\leq', '≤').replace('\\geq', '≥')
    s = s.replace('\\leqslant', '≤').replace('\\geqslant', '≥').replace('\\lt', '<').replace('\\gt', '>')
    # Split by < ≤ > ≥
    parts = re.split(r'\s*([<>≤≥])\s*', s.strip())
    if len(parts) < 3:
        return None
    # parts = [term, op, term, op, term, ...]
    terms = [parts[i].strip() for i in range(0, len(parts), 2)]
    ops = [parts[i] for i in range(1, len(parts), 2)]
    # Normalize direction: if first op is > or ≥, reverse everything
    if ops[0] in ('>', '≥'):
        terms = terms[::-1]
        ops = ops[::-1]
        op_map = {'<': '>', '>': '<', '≤': '≥', '≥': '≤'}
        ops = [op_map[o] for o in ops]
    return terms, ops


def _normalize_unicode_math(s: str) -> str:
    """Convert Unicode math symbols to LaTeX equivalents."""
    replacements = {
        '√': '\\sqrt',
        '×': '\\times',
        '÷': '\\div',
        '±': '\\pm',
        '∓': '\\mp',
        'π': '\\pi',
        '∞': '\\infty',
        '≤': '\\le',
        '≥': '\\ge',
        '≠': '\\ne',
        '∪': '\\cup',
        '∩': '\\cap',
        '∈': '\\in',
        '⊂': '\\subset',
        '⊆': '\\subseteq',
    }
    for uni, latex in replacements.items():
        s = s.replace(uni, latex)
    # Handle √expr → \sqrt{expr}: e.g. √2 → \sqrt{2}, √10 → \sqrt{10}
    s = re.sub(r'\\sqrt(\d+)', r'\\sqrt{\1}', s)
    return s


def _deep_normalize(s: str) -> str:
    """Comprehensive normalization: Unicode, option labels, broken LaTeX, spacing."""
    s = s.strip()
    # Remove $...$ wrapping
    if s.startswith('$') and s.endswith('$'):
        s = s[1:-1].strip()
    # Remove $$...$$ wrapping
    if s.startswith('$$') and s.endswith('$$'):
        s = s[2:-2].strip()
    # Remove leading option labels like "A.", "B.", "C.", "D."
    s = re.sub(r'^[A-D]\.\s*', '', s)
    # Unicode → LaTeX
    s = _normalize_unicode_math(s)
    # Fix sqrt(x) → \sqrt{x}
    s = re.sub(r'sqrt\(([^)]+)\)', r'\\sqrt{\1}', s)
    # Fix broken LaTeX: \text{rac} → \frac, bare form-feed+rac → \frac
    s = s.replace('\\text{rac}', '\\frac')
    s = s.replace('\x0crac', '\\frac')  # form feed + rac
    # Bare rac{ not preceded by \f → \frac{  (safety fallback)
    s = re.sub(r'(?<!\\f)(?<![a-z])rac\{', r'\\frac{', s)
    # Fix \wedgeq → \wedge q (missing space)
    s = s.replace('\\wedgeq', '\\wedge q')
    # \mathrm{e} → e, \text{e} → e (Euler's number)
    s = re.sub(r'\\mathrm\{e\}', 'e', s)
    s = re.sub(r'\\text\{e\}', 'e', s)
    # \frac{x}{1} → x (trivial denominator)
    s = re.sub(r'\\frac\{([^}]+)\}\{1\}', r'\1', s)
    # Escaped brackets: \[ → [, \] → ]
    s = s.replace('\\[', '[').replace('\\]', ']')
    # \left, \right removal
    s = s.replace('\\left', '').replace('\\right', '')
    # Normalize spacing
    s = re.sub(r'\s+', ' ', s).strip()
    return strip_string(s)


def enhanced_math_equal(pred: str, gt: str) -> bool:
    """math_equal with fallback for interval ↔ inequality, chain reversal, and Unicode normalization."""
    if math_equal(pred, gt):
        return True

    # Normalize both sides: Unicode → LaTeX, strip option labels, fix common formatting
    pred_clean = _deep_normalize(pred)
    gt_clean = _deep_normalize(gt)
    if math_equal(pred_clean, gt_clean):
        return True

    # Try interval comparison
    pred_iv = _to_interval(pred)
    gt_iv = _to_interval(gt)
    if pred_iv and gt_iv:
        p_lo, p_lc, p_hi, p_hc = pred_iv
        g_lo, g_lc, g_hi, g_hc = gt_iv
        if p_lc == g_lc and p_hc == g_hc:
            if math_equal(strip_string(p_lo), strip_string(g_lo)) and \
               math_equal(strip_string(p_hi), strip_string(g_hi)):
                return True

    # Try inequality chain reversal: c > a > b ↔ b < a < c
    pred_chain = _parse_chain(pred)
    gt_chain = _parse_chain(gt)
    if pred_chain and gt_chain:
        p_terms, p_ops = pred_chain
        g_terms, g_ops = gt_chain
        if len(p_terms) == len(g_terms) and p_ops == g_ops:
            if all(math_equal(strip_string(p), strip_string(g))
                   for p, g in zip(p_terms, g_terms)):
                return True

    # Numerical fallback: evaluate both as floats and compare
    try:
        from sympy import sympify, N
        from sympy.parsing.latex import parse_latex

        def _to_float(s):
            s = _deep_normalize(s)
            # Try sympy parse_latex first
            try:
                val = float(N(parse_latex(s)))
                if val == val:  # not NaN
                    return val
            except Exception:
                pass
            # Try sympify with common replacements
            s2 = s.replace('\\frac', '').replace('\\sqrt', 'sqrt')
            s2 = s2.replace('\\pi', 'pi').replace('\\infty', 'oo')
            s2 = s2.replace('{', '(').replace('}', ')')
            try:
                val = float(N(sympify(s2)))
                if val == val:
                    return val
            except Exception:
                pass
            return None

        pv = _to_float(pred)
        gv = _to_float(gt)
        if pv is not None and gv is not None:
            if abs(pv - gv) < 1e-6 * max(1, abs(gv)):
                return True
    except Exception:
        pass

    return False


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{} with latex format."


def load_test_set(path: str):
    """Load test set from JSONL or CSV file."""
    path = Path(path)
    problems, answers = [], []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                problems.append(row["question"])
                answers.append(str(row["answer"]))
    elif path.suffix == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                problems.append(row["question"])
                answers.append(str(row["answer"]))
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    return problems, answers


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
    problems, answers = load_test_set(args.input)
    print(f"Dataset: {DATASET_NAME} ({args.input}), {len(problems)} problems")

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
        gt_stripped = normalize_gt(gt)

        if args.mode == "greedy":
            response = output.outputs[0].text
            predicted = extract_answer(response)
            is_correct = enhanced_math_equal(predicted, gt_stripped)
            result = {"idx": i, "problem": problems[i][:200], "ground_truth": gt,
                      "predicted": predicted, "correct": is_correct}
        else:
            all_extracted = [extract_answer(s.text) for s in output.outputs]
            per_sample = [{"extracted": e, "correct": enhanced_math_equal(e, gt_stripped)} for e in all_extracted]
            n_correct_samples = sum(1 for s in per_sample if s["correct"])

            if args.mode == "majority":
                voted = majority_vote(all_extracted)
                is_correct = enhanced_math_equal(voted, gt_stripped)
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
            print(f"  [{status}] #{r['idx']:3d}  gt={r['ground_truth']}  voted={r['voted_answer']}  "
                  f"({r['n_correct_samples']}/{r['n_total_samples']} samples correct)")
        elif args.mode == "average":
            print(f"  [{status}] #{r['idx']:3d}  gt={r['ground_truth']}  "
                  f"{r['n_correct_samples']}/{r['n_total_samples']} = {r['sample_accuracy']:.1%}")
        else:
            print(f"  [{status}] #{r['idx']:3d}  gt={r['ground_truth']}  pred={r['predicted']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.replace("/", "_")
    output_path = output_dir / f"{model_short}_{DATASET_NAME}_{args.mode}.json"

    summary = {
        "model": args.model, "dataset": DATASET_NAME, "input": args.input,
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
        f.write(f"Dataset: {DATASET_NAME} ({args.input})\n")
        f.write(f"Mode: {mode_str}\n")
        f.write(f"Accuracy: {correct_count}/{len(results)} = {accuracy:.1%}\n")
        f.write(f"Time: {elapsed:.1f}s\n")
        f.write(f"{'='*60}\n")
        for r in results:
            status = "O" if r["correct"] else "X"
            if args.mode == "majority":
                f.write(f"[{status}] #{r['idx']:3d}  gt={r['ground_truth']}  voted={r['voted_answer']}  "
                        f"({r['n_correct_samples']}/{r['n_total_samples']} samples correct)\n")
            elif args.mode == "average":
                f.write(f"[{status}] #{r['idx']:3d}  gt={r['ground_truth']}  "
                        f"{r['n_correct_samples']}/{r['n_total_samples']} = {r['sample_accuracy']:.1%}\n")
            else:
                f.write(f"[{status}] #{r['idx']:3d}  gt={r['ground_truth']}  pred={r['predicted']}\n")

    print(f"\nResults saved to {output_path}")
    print(f"Results saved to {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on SURE-Math test set")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Path to SURE-Math test set (JSONL or CSV)")
    parser.add_argument("--mode", type=str, default="greedy", choices=["greedy", "majority", "average"])
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--num_shots", type=int, default=4, help="Number of few-shot CoT examples (0 for zero-shot)")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    args = parser.parse_args()
    evaluate(args)
