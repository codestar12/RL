# Theo - Eval Engineer Notes

## Mission
Build evaluation pipeline to measure progress toward an empathic software engineering assistant.

## Priority Metrics
- Code generation quality
- Software engineering capabilities
- Reasoning ability
- Helpfulness/instruction following

## Resources
- Built-in evals in this repo
- OLMES repo (user preference)
- Modern code evaluation benchmarks

## Tasks
- [ ] Survey existing evals in nemo_rl repo
- [ ] Identify code-specific evaluation benchmarks
- [ ] Explore OLMES integration
- [ ] Design eval pipeline
- [ ] Create baseline measurements
- [ ] Document evaluation process

## Evaluation Targets
- Pre-SFT baseline
- Post-SFT checkpoint
- Post-RL checkpoint

## Progress Log
<!-- Add timestamped entries below -->

### 2026-02-03 10:25 - Initial Survey Complete

**Surveyed Existing Evaluation Infrastructure in NeMo RL:**

1. **Core Eval System** (`/home/claude/code/RL/nemo_rl/evals/eval.py`):
   - Supports pass@k and cons@k metrics
   - Uses vLLM for inference
   - Environment-based evaluation (step through generations, compute rewards)
   - Saves results to JSON

2. **Existing Eval Datasets** (`/home/claude/code/RL/nemo_rl/data/datasets/eval_datasets/`):
   - AIME (2024, 2025) - math competition
   - MATH - math benchmark
   - MMLU / MMLU-Pro - general knowledge
   - GPQA - graduate-level QA
   - Local dataset support (flexible format)
   - **NO code benchmarks currently built in**

3. **Code Environments** (`/home/claude/code/RL/nemo_rl/environments/`):
   - `code_environment.py` - Sandboxed code execution with safe builtins
   - `code_jaccard_environment.py` - Jaccard similarity for code responses
   - Both use Ray workers for parallel execution

4. **3rdparty Code Evaluation** (`/home/claude/code/RL/3rdparty/Gym-workspace/Gym/resources_servers/`):
   - `code_gen/` - LiveCodeBench integration with pass@k metrics
   - `swerl_gen/` - SWE-bench style patch generation evaluation
   - `mini_swe_agent/` - Mini software engineering agent (placeholder)
   - Has full testing_util.py for running code tests

**Key Benchmarks Identified (2025 Landscape):**

| Benchmark | Type | Description | Integration Priority |
|-----------|------|-------------|---------------------|
| HumanEval | Function-level | 164 Python problems | HIGH - foundational |
| MBPP | Function-level | 1000 basic Python problems | HIGH - foundational |
| BigCodeBench | Library-level | Diverse function calls | MEDIUM |
| LiveCodeBench | Competition-style | Real coding challenges | MEDIUM - already integrated |
| SWE-Bench | Issue resolution | GitHub issues/PRs | LOW - complex setup |
| HumanEval Pro/MBPP Pro | Self-invoking | Extended versions | LOW |

**OLMES Research:**
- OLMES = Open Language Model Evaluation Standard (Allen AI)
- Supports 20+ benchmarks, reproducible evaluations
- Includes code tasks: `olmo3:base:code`, `olmo3:base:code_fim`
- Installation: `uv sync --group gpu` for vLLM support
- Good for standardized evaluation but adds complexity

**Recommended Evaluation Strategy:**

1. **Foundational Code Evals** (Priority):
   - HumanEval (pass@1, pass@5)
   - MBPP (pass@1, pass@5)
   - Use bigcode-evaluation-harness

2. **Reasoning & General** (existing):
   - MATH / MATH-500
   - MMLU-Pro
   - GPQA-Diamond

3. **Advanced Code** (stretch):
   - LiveCodeBench (already in repo)
   - BigCodeBench

4. **Instruction Following**:
   - IFEval or similar

### 2026-02-03 10:45 - Evaluation Pipeline Design

**Selected Tool: EvalPlus** (https://github.com/evalplus/evalplus)

Why EvalPlus over bigcode-evaluation-harness:
- Built-in vLLM support (`pip install "evalplus[vllm]"`)
- HumanEval+ has 80x more tests than original
- MBPP+ has 35x more tests than original
- Docker support for safe code execution
- Active maintenance (v0.2.1+)

**Installation Command:**
```bash
pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"
```

**Evaluation Commands:**

1. **HumanEval+ with vLLM:**
```bash
evalplus.evaluate --model "/path/to/model" \
    --dataset humaneval \
    --backend vllm \
    --tp 8 \
    --greedy
```

2. **MBPP+ with vLLM:**
```bash
evalplus.evaluate --model "/path/to/model" \
    --dataset mbpp \
    --backend vllm \
    --tp 8 \
    --greedy
```

3. **Safe Docker Execution (recommended for untrusted code):**
```bash
# Generate code
evalplus.codegen --model "/path/to/model" \
    --dataset humaneval \
    --backend vllm \
    --greedy

# Execute in sandbox
docker run --rm --pull=always -v $(pwd)/evalplus_results:/app ganler/evalplus:latest \
    evalplus.evaluate --dataset humaneval \
    --samples /app/humaneval/model_name_vllm_temp_0.0.jsonl
```

**Source Checkpoint for Baseline:**
- HF format: `s3://datology-research/cody/torchtitan_ckpts/math_and_code/v2/8b/phase3_1T/hf/step-127120/`
- Need to sync to local for evaluation

**Full Evaluation Suite:**

| Benchmark | Tool | Metric | Priority |
|-----------|------|--------|----------|
| HumanEval+ | EvalPlus | pass@1 | HIGH |
| MBPP+ | EvalPlus | pass@1 | HIGH |
| MATH-500 | NeMo RL built-in | pass@k | HIGH |
| MMLU-Pro | NeMo RL built-in | accuracy | MEDIUM |
| GPQA-Diamond | NeMo RL built-in | accuracy | MEDIUM |
| LiveCodeBench | Gym integration | pass@k | LOW |

**Pipeline Stages:**
1. Pre-SFT baseline (source checkpoint)
2. Post-SFT checkpoint
3. Post-RL checkpoint

**Next Steps:**
- [ ] Set up EvalPlus environment
- [ ] Download baseline checkpoint from S3
- [ ] Run initial HumanEval+ baseline
- [ ] Create evaluation script for all benchmarks

### 2026-02-03 11:15 - Implementation Details

**Environment Setup:**
EvalPlus is NOT pre-installed in the NeMo RL venv. Need to either:
1. Install into existing venv: `pip install "evalplus[vllm]"`
2. Create separate eval venv (cleaner approach)

**Scripts Created:**
- `/home/claude/code/RL/team/theo/eval_scripts/setup_evalplus.sh` - Setup EvalPlus
- `/home/claude/code/RL/team/theo/eval_scripts/run_code_eval.py` - Run code evals
- `/home/claude/code/RL/team/theo/eval_scripts/run_full_eval_suite.py` - Full eval suite

**Verified S3 Checkpoint Access:**
```
s3://datology-research/cody/torchtitan_ckpts/math_and_code/v2/8b/phase3_1T/hf/step-127120/
  - config.json
  - generation_config.json
  - model-00001-of-00001.safetensors (16GB - single file)
  - tokenizer.json, tokenizer_config.json
  - evals/ subfolder exists (previous evals?)
```

**Alternative: Use LiveCodeBench Integration Already in Repo**
The `3rdparty/Gym-workspace/Gym/resources_servers/code_gen/` has:
- LiveCodeBench test execution framework
- pass@k computation utilities
- Full testing_util.py for sandboxed code execution

This could be adapted for HumanEval/MBPP with less setup overhead.

### 2026-02-03 11:30 - BASELINE EVALUATIONS FOUND!

**Pre-existing baseline evaluations discovered at:**
`s3://datology-research/cody/torchtitan_ckpts/math_and_code/v2/8b/phase3_1T/hf/step-127120/evals/datology_math_and_code_base/`

**BASELINE RESULTS (Pre-SFT Model):**

| Benchmark | Metric | Score | Notes |
|-----------|--------|-------|-------|
| GSM8K | exact_match | 67.85% | 8-shot, OLMES |
| MATH-500 | exact_match | 40.0% | 4-shot, Minerva style |
| HumanEval (3-shot) | pass@1 | 40.95% | Temperature 0.8 |
| HumanEval (starcoder) | pass@1 | 58.60% | Temperature 0.2 |
| HumanEval (starcoder) | pass@10 | 82.27% | Temperature 0.8 |
| HumanEval+ | pass@1 | 42.68% | Temperature 0.8 |
| MBPP (3-shot) | pass@1 | 37.01% | Temperature 0.8 |
| MBPP (starcoder) | pass@1 | 41.30% | Temperature 0.2 |
| MBPP (starcoder) | pass@10 | 60.20% | Temperature 0.8 |
| MBPP+ | pass@1 | 54.23% | Temperature 0.8 |

**Key Observations:**
1. Model already has solid code generation capabilities (58.6% HumanEval pass@1)
2. Math reasoning is decent (40% MATH-500, 67.8% GSM8K)
3. MBPP+ shows higher scores than HumanEval+ (54% vs 43%)
4. Temperature 0.2 gives better pass@1; 0.8 better for pass@k

**These are the metrics to beat after SFT and RL!**

**Evaluation Tool Used:** OLMES (Allen AI)
- vLLM backend
- max_length: 4096
- Comprehensive code execution for pass@k

**This confirms OLMES is the right tool for our evaluations.**
