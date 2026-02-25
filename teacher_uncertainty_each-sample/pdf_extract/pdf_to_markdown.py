"""
使用 Marker 将 PDF 转换为 Markdown。

Usage:
  python pdf_to_markdown.py --input math.pdf --output_dir ./
"""

import argparse
import os
import subprocess
import sys


def main(args):
    if not os.path.isfile(args.input):
        print(f"错误: 文件不存在 {args.input}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    cmd = [
        "marker_single",
        args.input,
        "--output_dir", args.output_dir,
    ]

    if args.force_ocr:
        cmd.append("--force_ocr")

    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"\n转换完成! 输出目录: {args.output_dir}")
        # 列出输出文件
        for f in os.listdir(args.output_dir):
            if f.endswith(".md"):
                fpath = os.path.join(args.output_dir, f)
                lines = sum(1 for _ in open(fpath, encoding="utf-8"))
                print(f"  {f} ({lines} 行)")
    else:
        print(f"Marker 失败, 返回码: {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF转Markdown (Marker)")
    parser.add_argument("--input", type=str, required=True, help="PDF文件路径")
    parser.add_argument("--output_dir", type=str, default=".", help="输出目录")
    parser.add_argument("--force_ocr", action="store_true", help="强制使用OCR")
    args = parser.parse_args()
    main(args)
