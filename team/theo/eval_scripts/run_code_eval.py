#!/usr/bin/env python3
"""
Code Evaluation Script for Team Impossible
Evaluates models on HumanEval+ and MBPP+ using EvalPlus

Usage:
    python run_code_eval.py --model /path/to/model --tp 8
    python run_code_eval.py --model s3://path/to/checkpoint --tp 8
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def sync_s3_checkpoint(s3_path: str, local_dir: str) -> str:
    """Sync S3 checkpoint to local directory."""
    local_path = Path(local_dir) / Path(s3_path).name
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"Syncing checkpoint from {s3_path} to {local_path}...")
    subprocess.run(
        ["aws", "s3", "sync", s3_path, str(local_path)],
        check=True
    )
    return str(local_path)


def run_evalplus(
    model_path: str,
    dataset: str,
    backend: str = "vllm",
    tp: int = 8,
    output_dir: str = None,
    greedy: bool = True,
    n_samples: int = None,
    temperature: float = None,
) -> dict:
    """Run EvalPlus evaluation."""

    cmd = [
        "evalplus.evaluate",
        "--model", model_path,
        "--dataset", dataset,
        "--backend", backend,
        "--tp", str(tp),
    ]

    if greedy:
        cmd.append("--greedy")
    elif n_samples and temperature:
        cmd.extend(["--n_samples", str(n_samples)])
        cmd.extend(["--temperature", str(temperature)])

    if output_dir:
        cmd.extend(["--output_dir", output_dir])

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None

    print(result.stdout)
    return {"stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser(description="Run code evaluation suite")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model (local or s3://...)"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=8,
        help="Tensor parallel size (default: 8)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["humaneval", "mbpp"],
        choices=["humaneval", "mbpp"],
        help="Datasets to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/claude/code/RL/team/theo/eval_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        default=True,
        help="Use greedy decoding (default: True)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples for pass@k (mutually exclusive with --greedy)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--local-checkpoint-dir",
        type=str,
        default="/home/claude/code/RL/team/theo/checkpoints",
        help="Local directory for S3 checkpoints"
    )

    args = parser.parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path(args.output_dir) / timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Handle S3 paths
    model_path = args.model
    if args.model.startswith("s3://"):
        model_path = sync_s3_checkpoint(args.model, args.local_checkpoint_dir)

    # Determine greedy or sampling
    greedy = args.greedy
    if args.n_samples or args.temperature:
        greedy = False

    # Run evaluations
    results = {}
    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset.upper()}")
        print(f"{'='*60}\n")

        dataset_output = run_output_dir / dataset
        dataset_output.mkdir(exist_ok=True)

        result = run_evalplus(
            model_path=model_path,
            dataset=dataset,
            backend="vllm",
            tp=args.tp,
            output_dir=str(dataset_output),
            greedy=greedy,
            n_samples=args.n_samples,
            temperature=args.temperature,
        )

        results[dataset] = result

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {run_output_dir}")

    return results


if __name__ == "__main__":
    main()
