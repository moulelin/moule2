"""
Merge clean + augment logs and translate to English.

Translates question and answer separately — model returns plain text,
no JSON parsing needed.

Reads:
  - math_clean_log.jsonl   (644 entries)
  - math_augment_log.jsonl (281 entries)

Outputs:
  - math_merged_en.jsonl   (all entries with translated question/answer)

Usage:
  python translate_math.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --clean_log ../cleaned/math_clean_log.jsonl \
    --augment_log ../augmented/math_augment_log.jsonl \
    --output_dir ../merged \
    --tp 2 --batch_size 64
"""

import argparse
import json
import os
import re
import time

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


SYSTEM_PROMPT = """\
Translate the following math text from Chinese to English.

Rules:
1. Translate all Chinese text to natural, fluent English.
2. All mathematical expressions MUST be in standard LaTeX wrapped with $...$ delimiters. For example: $x^2+1$, $\\frac{1}{2}$, $\\{1, 2, 3\\}$, $\\sqrt{2}$.
3. Do NOT leave any LaTeX commands outside of $...$ delimiters.
4. Do NOT solve the problem or modify the mathematical content.
5. If the text is already in English, ensure the LaTeX formatting is correct and return it.
6. Output ONLY the translated text, nothing else."""


def has_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def normalize_latex_answer(answer: str) -> str:
    """Ensure answer is wrapped in $...$ if it contains LaTeX commands."""
    answer = answer.strip()
    if not answer:
        return answer

    # Already properly wrapped in $...$
    if answer.startswith('$') and answer.endswith('$'):
        return answer

    # Contains LaTeX commands → wrap in $...$
    latex_cmds = r'\\(frac|sqrt|left|right|text|cdot|pi|infty|cup|cap|complement|mathbb|leqslant|geqslant|neq|le|ge|lt|gt|pm|mp|times|div|circ|boxed|overline|underline|hat|bar|vec|sum|prod|int|lim|log|ln|sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan)'
    has_latex = bool(re.search(latex_cmds, answer))
    has_braces = '\\{' in answer or '\\}' in answer

    if has_latex or has_braces:
        return f'${answer}$'

    return answer


def clean_output(raw: str) -> str:
    """Remove thinking tags and strip."""
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    return raw.strip()


def main(args):
    # ---- Load and merge ----
    entries = []

    for path, source in [(args.clean_log, "clean"), (args.augment_log, "augment")]:
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                entries.append({
                    "question": row["question"],
                    "answer": row["answer"],
                    "source": source,
                })

    print(f"Total entries: {len(entries)} (clean + augment)")

    # Separate Chinese vs English-only
    needs_translate_q = []  # (entry_idx, text)
    needs_translate_a = []  # (entry_idx, text)

    for i, e in enumerate(entries):
        q_cn = has_chinese(e["question"])
        a_cn = has_chinese(e["answer"])
        if q_cn:
            needs_translate_q.append((i, e["question"]))
        if a_cn:
            needs_translate_a.append((i, e["answer"]))

    print(f"Questions needing translation: {len(needs_translate_q)}")
    print(f"Answers needing translation:   {len(needs_translate_a)}")
    total_prompts = len(needs_translate_q) + len(needs_translate_a)
    print(f"Total prompts: {total_prompts}")

    # ---- Build prompts ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    all_prompts = []
    prompt_map = []  # (entry_idx, field: "question"|"answer")

    for idx, text in needs_translate_q:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts.append(prompt)
        prompt_map.append((idx, "question"))

    for idx, text in needs_translate_a:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts.append(prompt)
        prompt_map.append((idx, "answer"))

    # ---- vLLM inference ----
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=args.max_tokens,
    )

    print(f"Generating translations for {len(all_prompts)} prompts...")
    start_time = time.time()

    all_outputs = []
    for batch_start in range(0, len(all_prompts), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(all_prompts))
        batch = all_prompts[batch_start:batch_end]
        print(f"  Batch {batch_start}-{batch_end} / {len(all_prompts)}")
        outputs = llm.generate(batch, sampling_params)
        all_outputs.extend(outputs)

    elapsed = time.time() - start_time
    print(f"Translation completed in {elapsed:.1f}s")

    # ---- Apply translations ----
    results = []
    for e in entries:
        results.append(dict(e))  # copy

    for j, (idx, field) in enumerate(prompt_map):
        translated = clean_output(all_outputs[j].outputs[0].text)
        if translated:
            results[idx][field] = translated

    # ---- Normalize answer to standard LaTeX ----
    for r in results:
        r["answer"] = normalize_latex_answer(r["answer"])

    # ---- Write output ----
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "math_merged_en.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nOutput saved to {output_path}")
    print(f"Total: {len(results)} entries")

    # ---- Verify ----
    remaining_cn = 0
    for r in results:
        if has_chinese(r["question"] + " " + r["answer"]):
            remaining_cn += 1
    print(f"Remaining Chinese entries: {remaining_cn}/{len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and translate math dataset to English")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--clean_log", type=str, required=True)
    parser.add_argument("--augment_log", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()
    main(args)
