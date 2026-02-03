#!/usr/bin/env python3
"""
Evaluation Comparison Tool for Team Impossible
Compares evaluation results across baseline, post-SFT, and post-RL stages

Usage:
    python compare_evals.py --baseline /path/to/baseline --post-sft /path/to/sft [--post-rl /path/to/rl]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional


# Baseline results from S3 evaluation
BASELINE_RESULTS = {
    "gsm8k::olmes": 0.6785,
    "minerva_math_500::olmes": 0.4000,
    "codex_humaneval:3shot::olmo3": 0.4095,
    "codex_humaneval::starcoder_pass@1": 0.5860,
    "codex_humaneval::starcoder_pass@10": 0.8227,
    "codex_humanevalplus:temp0.8": 0.4268,
    "mbpp:3shot::olmo3": 0.3701,
    "mbpp::starcoder_pass@1": 0.4130,
    "mbpp::starcoder_pass@10": 0.6020,
    "mbppplus::none": 0.5423,
}


def load_results(results_dir: Path) -> Dict[str, float]:
    """Load results from OLMES output directory."""
    results = {}

    # Try to find metrics.json in subdirectories
    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            metrics_file = subdir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    data = json.load(f)
                    if "tasks" in data:
                        for task in data["tasks"]:
                            results[task["alias"]] = task["metrics"].get("primary_score", None)

    # Also check for top-level metrics.json
    top_metrics = results_dir / "metrics.json"
    if top_metrics.exists():
        with open(top_metrics) as f:
            data = json.load(f)
            if "tasks" in data:
                for task in data["tasks"]:
                    results[task["alias"]] = task["metrics"].get("primary_score", None)

    return results


def format_score(score: Optional[float], baseline: Optional[float] = None) -> str:
    """Format score with delta from baseline."""
    if score is None:
        return "N/A"

    score_str = f"{score * 100:.2f}%"

    if baseline is not None and baseline > 0:
        delta = (score - baseline) * 100
        sign = "+" if delta >= 0 else ""
        delta_str = f" ({sign}{delta:.2f})"
        score_str += delta_str

    return score_str


def compare_results(
    baseline: Dict[str, float],
    post_sft: Optional[Dict[str, float]] = None,
    post_rl: Optional[Dict[str, float]] = None,
) -> None:
    """Print comparison table."""
    # Collect all tasks
    all_tasks = set(baseline.keys())
    if post_sft:
        all_tasks.update(post_sft.keys())
    if post_rl:
        all_tasks.update(post_rl.keys())

    # Define categories
    code_tasks = [t for t in all_tasks if "humaneval" in t.lower() or "mbpp" in t.lower()]
    math_tasks = [t for t in all_tasks if "math" in t.lower() or "gsm" in t.lower()]
    other_tasks = [t for t in all_tasks if t not in code_tasks and t not in math_tasks]

    print("\n" + "=" * 80)
    print("EVALUATION COMPARISON - Team Impossible")
    print("=" * 80)

    header = f"{'Benchmark':<45} | {'Baseline':<15}"
    if post_sft:
        header += f" | {'Post-SFT':<18}"
    if post_rl:
        header += f" | {'Post-RL':<18}"
    print(header)
    print("-" * 80)

    def print_category(category_name: str, tasks: list):
        if not tasks:
            return
        print(f"\n{category_name}:")
        for task in sorted(tasks):
            base_score = baseline.get(task)
            row = f"  {task:<43} | {format_score(base_score):<15}"

            if post_sft:
                sft_score = post_sft.get(task)
                row += f" | {format_score(sft_score, base_score):<18}"

            if post_rl:
                rl_score = post_rl.get(task)
                row += f" | {format_score(rl_score, base_score):<18}"

            print(row)

    print_category("CODE BENCHMARKS", code_tasks)
    print_category("MATH BENCHMARKS", math_tasks)
    print_category("OTHER BENCHMARKS", other_tasks)

    print("\n" + "=" * 80)

    # Calculate and print averages
    def avg_score(results: Dict[str, float], tasks: list) -> float:
        scores = [results.get(t) for t in tasks if results.get(t) is not None]
        return sum(scores) / len(scores) if scores else 0

    print("\nAVERAGES:")
    print(f"  Code: Baseline={avg_score(baseline, code_tasks)*100:.2f}%", end="")
    if post_sft:
        print(f" | Post-SFT={avg_score(post_sft, code_tasks)*100:.2f}%", end="")
    if post_rl:
        print(f" | Post-RL={avg_score(post_rl, code_tasks)*100:.2f}%", end="")
    print()

    print(f"  Math: Baseline={avg_score(baseline, math_tasks)*100:.2f}%", end="")
    if post_sft:
        print(f" | Post-SFT={avg_score(post_sft, math_tasks)*100:.2f}%", end="")
    if post_rl:
        print(f" | Post-RL={avg_score(post_rl, math_tasks)*100:.2f}%", end="")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline results (or use built-in)",
    )
    parser.add_argument(
        "--post-sft",
        type=str,
        default=None,
        help="Path to post-SFT results",
    )
    parser.add_argument(
        "--post-rl",
        type=str,
        default=None,
        help="Path to post-RL results",
    )
    parser.add_argument(
        "--show-baseline-only",
        action="store_true",
        help="Just show baseline results",
    )

    args = parser.parse_args()

    # Load baseline
    if args.baseline:
        baseline = load_results(Path(args.baseline))
    else:
        baseline = BASELINE_RESULTS
        print("Using built-in baseline results from S3 evaluation")

    # Load post-SFT
    post_sft = None
    if args.post_sft:
        post_sft = load_results(Path(args.post_sft))

    # Load post-RL
    post_rl = None
    if args.post_rl:
        post_rl = load_results(Path(args.post_rl))

    # Compare
    compare_results(baseline, post_sft, post_rl)


if __name__ == "__main__":
    main()
