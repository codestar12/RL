#!/bin/bash
# Setup script for EvalPlus code evaluation
# Run this once to set up the evaluation environment

set -e

# Source environment
source /home/claude/code/RL/team/.env

echo "=== Setting up EvalPlus for Code Evaluation ==="

# Create virtual environment for evalplus (isolated from main nemo_rl env)
EVAL_ENV_DIR="/home/claude/code/RL/team/theo/evalplus_env"

if [ ! -d "$EVAL_ENV_DIR" ]; then
    echo "Creating evalplus virtual environment..."
    python3 -m venv "$EVAL_ENV_DIR"
fi

# Activate evalplus environment
source "$EVAL_ENV_DIR/bin/activate"

# Install evalplus with vLLM support
echo "Installing EvalPlus with vLLM support..."
pip install --upgrade pip
pip install --upgrade "evalplus[vllm]"

# Install additional dependencies
pip install transformers accelerate torch

echo "=== EvalPlus setup complete ==="
echo "To activate: source $EVAL_ENV_DIR/bin/activate"
echo ""
echo "Available commands:"
echo "  evalplus.codegen  - Generate code samples"
echo "  evalplus.evaluate - Evaluate generated samples"
echo ""
echo "Example usage:"
echo "  evalplus.evaluate --model 'path/to/model' --dataset humaneval --backend vllm --greedy"
