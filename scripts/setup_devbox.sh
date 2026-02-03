#!/bin/bash
# Dev box setup script for Llama 3 8B post-training
# Run this after cloning your fork on the dev box

set -e

echo "=== NeMo RL Dev Box Setup ==="

# Check for GPUs
echo -e "\n[1/6] Checking GPU access..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "WARNING: nvidia-smi not found. Make sure NVIDIA drivers are installed."
fi

# Check for uv
echo -e "\n[2/6] Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# Install Claude Code
echo -e "\n[3/6] Installing Claude Code..."
if ! command -v claude &> /dev/null; then
    curl -fsSL https://claude.ai/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "Claude Code already installed: $(claude --version 2>/dev/null || echo 'unknown version')"
fi

# Create venv
echo -e "\n[4/6] Creating virtual environment..."
uv venv
echo "Virtual environment created at .venv/"

# Suggest environment variables
echo -e "\n[5/6] Environment variables to set (add to ~/.bashrc):"
cat << 'EOF'
# NeMo RL environment variables
export HF_HOME=/path/to/huggingface/cache
export HF_DATASETS_CACHE=/path/to/datasets/cache
export WANDB_API_KEY=your_wandb_key  # Optional, for logging

# If running multiple experiments, helpful to set:
export RAY_DEDUP_LOGS=0  # Show all worker logs for debugging
EOF

# Verify PyTorch can see GPUs
echo -e "\n[6/6] Verifying PyTorch GPU access..."
uv run python -c "
import torch
gpu_count = torch.cuda.device_count()
print(f'PyTorch sees {gpu_count} GPU(s)')
if gpu_count > 0:
    for i in range(gpu_count):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('WARNING: No GPUs detected by PyTorch')
"

echo -e "\n=== Setup Complete ==="
echo "Next steps:"
echo "  1. Set environment variables shown above"
echo "  2. Run: huggingface-cli login"
echo "  3. Run: wandb login"
echo "  4. Run: claude (to authenticate Claude Code)"
echo "  5. Copy your model checkpoint to this machine"
echo "  6. Run a test: uv run python examples/run_sft.py --help"
