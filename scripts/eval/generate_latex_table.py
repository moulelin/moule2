"""
Print evaluation results as a formatted table to terminal.
Reads JSON files from the results directory.
"""

import argparse
import json
from pathlib import Path


MODELS = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

DATASETS = ["aime24", "aime25", "hmmt25", "amo_bench"]

DATASET_DISPLAY = {
    "aime24": "AIME24",
    "aime25": "AIME25",
    "hmmt25": "HMMT25",
    "amo_bench": "AMO",
}

MODEL_DISPLAY = {
    "Qwen/Qwen3-1.7B": "Qwen3-1.7B",
    "Qwen/Qwen3-4B": "Qwen3-4B",
    "Qwen/Qwen3-8B": "Qwen3-8B",
}


def load_results(results_dir: Path) -> dict:
    results = {}
    for f in results_dir.glob("*.json"):
        with open(f) as fp:
            d = json.load(fp)
        model = d["model"]
        dataset = d["dataset"]
        mode = d["mode"]
        acc = d["accuracy"]
        results.setdefault(model, {}).setdefault(dataset, {})[mode] = acc
    return results


def fmt(acc):
    if acc is None:
        return "--"
    return f"{acc * 100:.1f}"


def print_table(results: dict):
    # Header
    header = f"{'Model':<15s}"
    for ds in DATASETS:
        header += f" | {DATASET_DISPLAY[ds]:^15s}"
    print(header)

    subheader = f"{'':15s}"
    for _ in DATASETS:
        subheader += f" | {'p@1':>6s}  {'avg@16':>6s}"
    print(subheader)
    print("-" * len(subheader))

    # Data rows
    for model in MODELS:
        name = MODEL_DISPLAY.get(model, model)
        row = f"{name:<15s}"
        for ds in DATASETS:
            g = results.get(model, {}).get(ds, {}).get("greedy")
            a = results.get(model, {}).get(ds, {}).get("average")
            row += f" | {fmt(g):>6s}  {fmt(a):>6s}"
        print(row)

    # LaTeX row format for easy copy-paste
    print("\n--- LaTeX rows (copy into baseline_results.tex) ---\n")
    for model in MODELS:
        name = MODEL_DISPLAY.get(model, model)
        parts = [name]
        for ds in DATASETS:
            g = results.get(model, {}).get(ds, {}).get("greedy")
            a = results.get(model, {}).get(ds, {}).get("average")
            parts.append(fmt(g))
            parts.append(fmt(a))
        print(" & ".join(parts) + r" \\")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    results = load_results(Path(args.results_dir))
    print_table(results)
