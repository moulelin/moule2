"""
按讲次顺序合并 download/ 目录下所有 PDF 为一个文件。

Usage:
  python merge_pdfs.py
"""

import os
import re
from pypdf import PdfWriter

INPUT_DIR = os.path.join(os.path.dirname(__file__), "download")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "gaokao_merged.pdf")


def extract_lecture_num(filename):
    """从文件名中提取讲次编号，如 '第10讲  对数与对数函数.pdf' -> 10"""
    m = re.search(r"第(\d+)讲", filename)
    return int(m.group(1)) if m else 999


def main():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
    files.sort(key=extract_lecture_num)

    print(f"共 {len(files)} 个 PDF，按讲次顺序合并:")
    writer = PdfWriter()
    for f in files:
        path = os.path.join(INPUT_DIR, f)
        print(f"  {f}")
        writer.append(path)

    writer.write(OUTPUT_PATH)
    writer.close()

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\n合并完成: {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
