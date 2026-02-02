# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeMo RL is NVIDIA's scalable, open-source post-training library for reinforcement learning on language models and vision-language models (VLMs). It supports training from single-GPU experiments to multi-thousand-GPU distributed deployments.

**Key Characteristics:**
- Python 3.12+, PyTorch-based
- Uses Ray for distributed coordination
- Uses `uv` as the package manager (not pip/conda)
- Supports DTensor/FSDP2 and Megatron Core training backends
- Supports vLLM, SGLang, and Megatron inference backends

## Common Commands

### Environment Setup
```bash
git clone --recursive https://github.com/NVIDIA-NeMo/RL.git nemo-rl
cd nemo-rl
uv venv
```

### Running Training
```bash
# Always use 'uv run' instead of activating venv
uv run python examples/run_grpo_math.py
uv run python examples/run_sft.py
uv run python examples/run_dpo.py
```

### Testing
```bash
# Prepare test assets (required for some tests)
uv run tests/unit/prepare_unit_test_assets.py

# Unit tests (require 2 GPUs for full suite)
uv run --group test bash tests/run_unit.sh              # Default tests
uv run --group test bash tests/run_unit.sh --hf-gated   # Include HF gated model tests
uv run --extra mcore --group test bash tests/run_unit.sh --mcore-only  # Megatron Core tests

# Run specific test file
uv run --group test pytest tests/unit/path/to/test_file.py

# Functional tests (may require multiple GPUs)
uv run bash tests/functional/sft.sh

# Static type checking
uv run --group test mypy examples/run_grpo_math.py
```

### Linting & Formatting
```bash
# Install pre-commit hooks
uv run --group dev pre-commit install

# Run pre-commit on all files
uv run --group dev pre-commit run --all-files

# Ruff is used for linting/formatting
uv run --group dev ruff check .
uv run --group dev ruff format .
```

### Docker Development
```bash
docker buildx build -t nemo-rl:latest -f Dockerfile .
docker run -it --gpus all -v /path/to/nemo-rl:/nemo-rl nemo-rl:latest
```

## Architecture Overview

### Single-Controller Architecture
The main process orchestrates distributed workers via Ray. Key flow:
1. Main controller loads config, initializes Ray, sets up tokenizer
2. Creates policy model, generation backend, and environment
3. Training loop: generate → evaluate → compute loss → backward pass

### Core Components

| Directory | Purpose |
|-----------|---------|
| `nemo_rl/algorithms/` | GRPO, DPO, SFT, Distillation, Reward Model implementations |
| `nemo_rl/models/policy/` | Policy model wrapper with DTensor/Megatron backends |
| `nemo_rl/models/generation/` | vLLM, SGLang, Megatron inference backends |
| `nemo_rl/data/` | Datasets, processors, collate functions, sequence packing |
| `nemo_rl/environments/` | Math, code, VLM environments; reward computation |
| `nemo_rl/distributed/` | RayVirtualCluster, RayWorkerGroup, NCCL collectives |
| `nemo_rl/utils/` | Logging (W&B/TensorBoard/MLflow), checkpointing, config |

### Configuration System

**YAML is the single source of truth for defaults - no defaults in code.**

- **Exemplar configs** (`examples/configs/*.yaml`): Base configurations with documentation
- **Recipe configs** (`examples/configs/recipes/llm/`, `recipes/vlm/`): Runnable snapshots
- **Naming convention**: `<algo>-<model>-<nodes>n<gpus>g-<strategy>[-modifiers][.vN].yaml`

Config sections: `policy`, `algorithm`, `loss_fn`, `cluster`, `data`, `env`, `logger`, `checkpointing`

CLI overrides use Hydra syntax: `uv run python examples/run_grpo_math.py policy.lr=1e-5`

### Key Interfaces

- `PolicyInterface`: Protocol for policy models (training + inference)
- `EnvironmentInterface`: Protocol for reward environments
- `GenerationInterface`: Protocol for text generation backends

## Development Guidelines

### Commits
- **Always sign-off commits**: `git commit --signoff -m "message"`
- NVIDIA copyright header required on Python files (except tests)

### Code Style
- Google Python Style Guide, 4 spaces indentation
- snake_case for files/functions, PascalCase for classes
- Google-style docstrings using Sphinx format
- Use TypedDict for config typing with `typing.NotRequired` for optional fields

### Configuration Rules
- **No code defaults** - all config defaults come from YAML files
- Be explicit; avoid arbitrary defaults in configs
- Recipes must pass minimize-check (pre-commit validates this)

### Testing Notes
- `@ray.remote` decorated functions need `# pragma: no cover` due to separate processes
- Unit tests may log metrics via the `tracker` fixture
- New key features require documentation updates
