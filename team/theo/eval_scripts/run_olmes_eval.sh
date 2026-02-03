#!/bin/bash
# OLMES Evaluation Script for Team Impossible
# Replicates baseline evaluation for post-SFT and post-RL checkpoints
#
# Usage:
#   ./run_olmes_eval.sh <model_path> <stage> [output_dir]
#
# Examples:
#   ./run_olmes_eval.sh /path/to/model baseline
#   ./run_olmes_eval.sh s3://bucket/checkpoint post-sft
#   ./run_olmes_eval.sh /path/to/model post-rl /custom/output

set -e

# Source environment
source /home/claude/code/RL/team/.env

MODEL_PATH="${1:?Error: Model path required}"
STAGE="${2:?Error: Stage required (baseline/post-sft/post-rl)}"
OUTPUT_DIR="${3:-/home/claude/code/RL/team/theo/eval_results}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_OUTPUT="${OUTPUT_DIR}/${STAGE}_${TIMESTAMP}"

echo "=============================================="
echo "OLMES Evaluation Suite"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Stage: ${STAGE}"
echo "Output: ${RUN_OUTPUT}"
echo "=============================================="

# Create output directory
mkdir -p "${RUN_OUTPUT}"

# If S3 path, sync to local
LOCAL_MODEL="${MODEL_PATH}"
if [[ "${MODEL_PATH}" == s3://* ]]; then
    LOCAL_MODEL="${OUTPUT_DIR}/checkpoints/$(basename ${MODEL_PATH})"
    mkdir -p "${LOCAL_MODEL}"
    echo "Syncing model from S3..."
    aws s3 sync "${MODEL_PATH}" "${LOCAL_MODEL}"
fi

# Check if olmes is installed
if ! command -v olmes &> /dev/null; then
    echo "Installing OLMES..."
    pip install git+https://github.com/allenai/olmes.git
    pip install "vllm>=0.6.0"
fi

# Run OLMES evaluation with the same tasks as baseline
# Based on metrics.json from baseline evaluation

echo ""
echo "=== Running GSM8K (Math) ==="
olmes --model "${LOCAL_MODEL}" \
    --model-type vllm \
    --model-args '{"trust_remote_code": true, "max_length": 4096}' \
    --task gsm8k::olmes \
    --output-dir "${RUN_OUTPUT}/gsm8k" || echo "GSM8K evaluation failed"

echo ""
echo "=== Running MATH-500 ==="
olmes --model "${LOCAL_MODEL}" \
    --model-type vllm \
    --model-args '{"trust_remote_code": true, "max_length": 4096}' \
    --task minerva_math_500::olmes \
    --output-dir "${RUN_OUTPUT}/math500" || echo "MATH-500 evaluation failed"

echo ""
echo "=== Running HumanEval (Code) ==="
olmes --model "${LOCAL_MODEL}" \
    --model-type vllm \
    --model-args '{"trust_remote_code": true, "max_length": 4096}' \
    --task codex_humaneval::starcoder_pass@1 \
    --output-dir "${RUN_OUTPUT}/humaneval" || echo "HumanEval evaluation failed"

echo ""
echo "=== Running HumanEval+ (Code) ==="
olmes --model "${LOCAL_MODEL}" \
    --model-type vllm \
    --model-args '{"trust_remote_code": true, "max_length": 4096}' \
    --task 'codex_humanevalplus:temp0.8' \
    --output-dir "${RUN_OUTPUT}/humanevalplus" || echo "HumanEval+ evaluation failed"

echo ""
echo "=== Running MBPP (Code) ==="
olmes --model "${LOCAL_MODEL}" \
    --model-type vllm \
    --model-args '{"trust_remote_code": true, "max_length": 4096}' \
    --task mbpp::starcoder_pass@1 \
    --output-dir "${RUN_OUTPUT}/mbpp" || echo "MBPP evaluation failed"

echo ""
echo "=== Running MBPP+ (Code) ==="
olmes --model "${LOCAL_MODEL}" \
    --model-type vllm \
    --model-args '{"trust_remote_code": true, "max_length": 4096}' \
    --task mbppplus::none \
    --output-dir "${RUN_OUTPUT}/mbppplus" || echo "MBPP+ evaluation failed"

echo ""
echo "=============================================="
echo "Evaluation Complete"
echo "Results saved to: ${RUN_OUTPUT}"
echo "=============================================="

# Generate summary
echo ""
echo "Generating summary..."
python3 << 'EOF'
import json
import os
import sys
from pathlib import Path

output_dir = sys.argv[1] if len(sys.argv) > 1 else "${RUN_OUTPUT}"
output_dir = Path("${RUN_OUTPUT}")

results = {}
for task_dir in output_dir.iterdir():
    if task_dir.is_dir():
        metrics_file = task_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
                if "tasks" in data:
                    for task in data["tasks"]:
                        results[task["alias"]] = task["metrics"].get("primary_score", "N/A")

print("\n=== SUMMARY ===")
for task, score in sorted(results.items()):
    if isinstance(score, float):
        print(f"{task}: {score*100:.2f}%")
    else:
        print(f"{task}: {score}")
EOF

# Save summary to file
echo "Stage: ${STAGE}" > "${RUN_OUTPUT}/summary.txt"
echo "Model: ${MODEL_PATH}" >> "${RUN_OUTPUT}/summary.txt"
echo "Timestamp: ${TIMESTAMP}" >> "${RUN_OUTPUT}/summary.txt"
echo "" >> "${RUN_OUTPUT}/summary.txt"
echo "Results saved to ${RUN_OUTPUT}"
