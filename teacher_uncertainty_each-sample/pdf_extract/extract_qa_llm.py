"""
用30B模型从markdown文件中提取数学题目和答案，输出CSV。

流程:
  1. 按题号正则切分markdown为题目块
  2. 预过滤: 含图片的跳过
  3. 分批送入LLM提取 question + answer (具体值，不是选项字母)
  4. 后过滤: 多选题跳过、答案过长跳过
  5. 输出 CSV (question, answer)

Usage:
  python extract_qa_llm.py \
    --input output_math_marker.md \
    --model Qwen/Qwen2.5-32B-Instruct \
    --output extracted_qa.csv \
    --tp 2
"""

import argparse
import csv
import json
import re
import os
from vllm import LLM, SamplingParams


EXTRACT_SYSTEM_PROMPT = """\
你是一个数学题目提取专家。给定一段从PDF中OCR提取的数学文档（可能有乱码），请提取其中每道题目的 **题干** 和 **答案**。

要求：
1. 题干：只要问题描述本身，不要包含选项(A/B/C/D)，不要包含解析/详解
2. 答案：只要最终答案的**具体值**，不要写选项字母。例如：
   - 如果正确选项是 A. 4，答案写 "4" 而不是 "A"
   - 如果正确选项是 D. $(-\\infty, 2)$，答案写 "$(-\\infty, 2)$" 而不是 "D"
   - 填空题直接写具体值
   - 所有数学公式用LaTeX表示
3. 跳过以下题目（不要输出）：
   - 多选题（答案是多个选项的，如AB、ACD等）
   - 含图片/图形的题（题干中有"如图"、"如下图"且无法独立理解的）
   - 答案过于复杂的（如需要证明、画图、写过程的大题）
   - OCR严重乱码导致题目不完整的
4. 所有数学公式用LaTeX表示，用$...$包裹
5. 保留原文语言（中文）

输出严格的JSON数组，每个元素：
{"question": "题干文本", "answer": "答案"}

如果这一段没有可提取的题目，输出空数组 []
"""


def split_into_question_blocks(text: str) -> list[dict]:
    """按题号切分文档"""
    pattern = r"(?:^|\n)\s*-?\s*(\d{1,4})\s*[.．。·]\s*【"
    splits = list(re.finditer(pattern, text))

    blocks = []
    for i, m in enumerate(splits):
        qnum = int(m.group(1))
        start = m.start()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        raw = text[start:end].strip()
        blocks.append({"num": qnum, "raw": raw})
    return blocks


def has_image(raw: str) -> bool:
    """检查是否包含图片引用"""
    return bool(re.search(r"!\[.*?\]\(.*?\)", raw))


def group_blocks(blocks: list[dict], max_chars: int = 6000) -> list[list[dict]]:
    """将题目块分组，每组不超过max_chars字符"""
    groups = []
    current_group = []
    current_len = 0

    for b in blocks:
        blen = len(b["raw"])
        if current_group and current_len + blen > max_chars:
            groups.append(current_group)
            current_group = []
            current_len = 0
        current_group.append(b)
        current_len += blen

    if current_group:
        groups.append(current_group)
    return groups


def build_prompt(blocks: list[dict]) -> str:
    """构建用户prompt"""
    text_parts = []
    for b in blocks:
        text_parts.append(b["raw"])
    combined = "\n\n---\n\n".join(text_parts)
    return f"请从以下文档片段中提取数学题目和答案：\n\n{combined}"


def is_multi_select(answer: str) -> bool:
    """检查是否为多选题答案（LLM仍可能输出选项字母）"""
    clean = answer.strip().upper()
    if re.match(r"^[A-D]{2,4}$", clean):
        return True
    return False


def answer_too_complex(answer: str, max_len: int = 150) -> bool:
    """答案过长"""
    return len(answer) > max_len or "\n" in answer


def parse_llm_output(text: str) -> list[dict]:
    """从LLM输出中解析JSON数组"""
    text = text.strip()
    # 尝试找JSON数组
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # 尝试逐行找JSON对象
    results = []
    for m in re.finditer(r"\{[^{}]+\}", text):
        try:
            obj = json.loads(m.group())
            if "question" in obj and "answer" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue
    return results


def main(args):
    # Step 1: Read and split
    print(f"读取文件: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = split_into_question_blocks(text)
    print(f"正则切分得到 {len(blocks)} 个题目块")

    # Step 2: Pre-filter images
    filtered_blocks = []
    skipped_img = 0
    for b in blocks:
        if has_image(b["raw"]):
            skipped_img += 1
        else:
            filtered_blocks.append(b)
    print(f"预过滤: 跳过 {skipped_img} 道图片题, 剩余 {len(filtered_blocks)} 题")

    # Step 3: Group into batches
    groups = group_blocks(filtered_blocks, max_chars=args.chunk_size)
    print(f"分为 {len(groups)} 批次送入LLM")

    # Step 4: Build prompts
    prompts = []
    for g in groups:
        user_msg = build_prompt(g)
        prompt = (
            f"<|im_start|>system\n{EXTRACT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompts.append(prompt)

    # Step 5: LLM inference
    print(f"\n加载模型: {args.model} (TP={args.tp})")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    print(f"开始推理 ({len(prompts)} 批次)...")
    outputs = llm.generate(prompts, sampling_params)

    # Step 6: Parse and filter
    all_results = []
    parse_fail = 0
    skipped_multi = 0
    skipped_complex = 0
    skipped_empty = 0

    for i, output in enumerate(outputs):
        gen_text = output.outputs[0].text
        items = parse_llm_output(gen_text)

        if not items:
            parse_fail += 1
            if args.verbose:
                print(f"  批次{i}: 解析失败或无结果")
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()

            if not q or not a:
                skipped_empty += 1
                continue

            if is_multi_select(a):
                skipped_multi += 1
                if args.verbose:
                    print(f"  跳过多选: {q[:40]}... -> {a}")
                continue

            if answer_too_complex(a, args.max_answer_len):
                skipped_complex += 1
                if args.verbose:
                    print(f"  跳过复杂答案: {q[:40]}... -> {a[:50]}...")
                continue

            all_results.append({"question": q, "answer": a})

    # Step 7: Output CSV
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer"])
        for r in all_results:
            writer.writerow([r["question"], r["answer"]])

    print(f"\n===== 提取完成 =====")
    print(f"  正则切分: {len(blocks)} 题")
    print(f"  预过滤(图片): -{skipped_img}")
    print(f"  LLM批次: {len(prompts)} (解析失败: {parse_fail})")
    print(f"  后过滤(多选): -{skipped_multi}")
    print(f"  后过滤(复杂): -{skipped_complex}")
    print(f"  后过滤(空): -{skipped_empty}")
    print(f"  最终提取: {len(all_results)} 题")
    print(f"  输出文件: {args.output}")

    # Preview
    print(f"\n前5题预览:")
    for r in all_results[:5]:
        print(f"  Q: {r['question'][:80]}...")
        print(f"  A: {r['answer']}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用LLM从markdown提取数学QA")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="extracted_qa.csv")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--chunk_size", type=int, default=6000,
                        help="每批最大字符数")
    parser.add_argument("--max_answer_len", type=int, default=150)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
