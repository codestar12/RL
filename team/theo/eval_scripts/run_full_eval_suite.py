#!/usr/bin/env python3
"""
Full Evaluation Suite for Team Impossible
Evaluates models across code, math, and reasoning benchmarks

Benchmarks:
- Code: HumanEval+, MBPP+ (via EvalPlus)
- Math: MATH-500, AIME2024 (via NeMo RL)
- Reasoning: MMLU-Pro, GPQA-Diamond (via NeMo RL)

Usage:
    python run_full_eval_suite.py --model /path/to/model --stage baseline
    python run_full_eval_suite.py --model /path/to/model --stage post-sft
    python run_full_eval_suite.py --model /path/to/model --stage post-rl
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add nemo_rl to path
NEMO_RL_ROOT = Path("/home/claude/code/RL")
sys.path.insert(0, str(NEMO_RL_ROOT))


def run_command(cmd: list, env: dict = None) -> dict:
    """Run a command and capture output."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=merged_env)

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def sync_s3_checkpoint(s3_path: str, local_dir: str) -> str:
    """Sync S3 checkpoint to local directory."""
    checkpoint_name = s3_path.rstrip("/").split("/")[-1]
    local_path = Path(local_dir) / checkpoint_name
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"Syncing checkpoint from {s3_path} to {local_path}...")
    result = subprocess.run(
        ["aws", "s3", "sync", s3_path, str(local_path)],
        check=True
    )
    return str(local_path)


def run_evalplus_benchmark(model_path: str, dataset: str, output_dir: Path, tp: int = 8) -> dict:
    """Run EvalPlus code evaluation (HumanEval+ or MBPP+)."""
    dataset_output = output_dir / f"evalplus_{dataset}"
    dataset_output.mkdir(exist_ok=True)

    # Use evalplus CLI
    cmd = [
        "evalplus.evaluate",
        "--model", model_path,
        "--dataset", dataset,
        "--backend", "vllm",
        "--tp", str(tp),
        "--greedy",
    ]

    result = run_command(cmd)

    # Save results
    with open(dataset_output / "output.txt", "w") as f:
        f.write(f"STDOUT:\n{result['stdout']}\n\nSTDERR:\n{result['stderr']}")

    return {
        "benchmark": f"evalplus_{dataset}",
        "success": result["returncode"] == 0,
        "output": result["stdout"],
    }


def run_nemo_eval_benchmark(
    model_path: str,
    dataset_name: str,
    output_dir: Path,
    gpus: int = 1,
) -> dict:
    """Run NeMo RL evaluation (MATH, MMLU, GPQA, AIME)."""
    dataset_output = output_dir / f"nemo_{dataset_name}"
    dataset_output.mkdir(exist_ok=True)

    # Run via uv
    cmd = [
        "uv", "run", "python", str(NEMO_RL_ROOT / "examples" / "run_eval.py"),
        f"data.dataset_name={dataset_name}",
        f"generation.model_name={model_path}",
        f"eval.save_path={dataset_output}",
        f"cluster.gpus_per_node={gpus}",
        "cluster.num_nodes=1",
    ]

    result = run_command(cmd, env={"PYTHONPATH": str(NEMO_RL_ROOT)})

    # Save results
    with open(dataset_output / "output.txt", "w") as f:
        f.write(f"STDOUT:\n{result['stdout']}\n\nSTDERR:\n{result['stderr']}")

    return {
        "benchmark": f"nemo_{dataset_name}",
        "success": result["returncode"] == 0,
        "output": result["stdout"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation suite")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model (local or s3://...)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["baseline", "post-sft", "post-rl"],
        help="Evaluation stage for tracking"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=8,
        help="Tensor parallel size for vLLM"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/claude/code/RL/team/theo/eval_results",
        help="Output directory"
    )
    parser.add_argument(
        "--local-checkpoint-dir",
        type=str,
        default="/home/claude/code/RL/team/theo/checkpoints",
        help="Local directory for S3 checkpoints"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["humaneval", "mbpp", "math500", "mmlu_pro"],
        help="Benchmarks to run"
    )
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Only run code benchmarks (EvalPlus)"
    )
    parser.add_argument(
        "--math-only",
        action="store_true",
        help="Only run math/reasoning benchmarks (NeMo RL)"
    )

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path(args.output_dir) / f"{args.stage}_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Handle S3 paths
    model_path = args.model
    if args.model.startswith("s3://"):
        model_path = sync_s3_checkpoint(args.model, args.local_checkpoint_dir)

    # Determine which benchmarks to run
    code_benchmarks = ["humaneval", "mbpp"]
    nemo_benchmarks = {
        "math500": "math500",
        "math": "math",
        "aime2024": "aime2024",
        "aime2025": "aime2025",
        "mmlu": "mmlu",
        "mmlu_pro": "mmlu_pro",
        "gpqa": "gpqa",
        "gpqa_diamond": "gpqa_diamond",
    }

    results = {
        "model": args.model,
        "model_path": model_path,
        "stage": args.stage,
        "timestamp": timestamp,
        "benchmarks": {},
    }

    # Run code benchmarks (EvalPlus)
    if not args.math_only:
        for benchmark in args.benchmarks:
            if benchmark in code_benchmarks:
                print(f"\n{'='*60}")
                print(f"Running {benchmark.upper()} (EvalPlus)")
                print(f"{'='*60}\n")
                result = run_evalplus_benchmark(
                    model_path=model_path,
                    dataset=benchmark,
                    output_dir=run_output_dir,
                    tp=args.tp,
                )
                results["benchmarks"][benchmark] = result

    # Run NeMo RL benchmarks
    if not args.code_only:
        for benchmark in args.benchmarks:
            if benchmark in nemo_benchmarks:
                print(f"\n{'='*60}")
                print(f"Running {benchmark.upper()} (NeMo RL)")
                print(f"{'='*60}\n")
                result = run_nemo_eval_benchmark(
                    model_path=model_path,
                    dataset_name=nemo_benchmarks[benchmark],
                    output_dir=run_output_dir,
                    gpus=min(args.tp, 2),  # NeMo eval typically uses fewer GPUs
                )
                results["benchmarks"][benchmark] = result

    # Save combined results
    results_file = run_output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY - {args.stage.upper()}")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Timestamp: {timestamp}")
    print(f"Results: {results_file}")
    print()

    for benchmark, result in results["benchmarks"].items():
        status = "PASS" if result.get("success") else "FAIL"
        print(f"  {benchmark}: {status}")

    return results


if __name__ == "__main__":
    main()
