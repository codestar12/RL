# Zara - RL Researcher Notes

## Mission
Develop GRPO pipeline for code-focused RL, ready to run after SFT phase.

## Resource Allocation
- Setup phase: 2 GPUs for experimentation
- Post-SFT: Full 8 GPU node

## Starting Points
1. GSM8K - verifiable math rewards (simpler starting point)
2. Code RL environments to explore:
   - SWE Gym (nvidia gym submodule in 3rdparty/)
   - nvidia/Nemotron-RL-coding-competition
   - Prime Intellect environments hub (stretch goal)

## Constraints
- No LLM judge API access
- Need self-hosted RM or verifiable rewards
- Focus on multi-turn RL where possible

## Tasks
- [x] Understand GRPO in NeMo RL
- [x] Explore NeMo Gym and code environments
- [x] Research code RL reward signals
- [x] Design post-SFT RL pipeline
- [x] Prepare production config for 8 GPU deployment
- [ ] Test GRPO on small model with GSM8K (2 GPU setup)
- [ ] Create code data processor
- [ ] Run end-to-end code GRPO test

## Created Configs

1. **2 GPU Test Config:** `/home/claude/code/RL/team/zara/configs/grpo_code_2gpu_test.yaml`
   - Model: Qwen2.5-Coder-1.5B-Instruct
   - 16 prompts, 8 generations per prompt
   - 100 steps test run
   - Uses math dataset initially to verify setup

2. **8 GPU Production Config:** `/home/claude/code/RL/team/zara/configs/grpo_code_8gpu_production.yaml`
   - Model: Qwen2.5-Coder-7B-Instruct (or SFT checkpoint)
   - 32 prompts, 16 generations per prompt
   - Async GRPO enabled
   - Full 1000 steps training

## Progress Log

### 2026-02-03 10:30 - Initial Deep Dive

**GRPO Algorithm Understanding:**
- Location: `/home/claude/code/RL/nemo_rl/algorithms/grpo.py`
- Main entry: `run_grpo_math.py` -> calls `grpo_train()` or `async_grpo_train()`
- Supports both synchronous and async GRPO training
- Key configs in `MasterConfig`: policy, loss_fn, env, data, grpo, logger, cluster, checkpointing

**Key GRPO Parameters:**
- `num_prompts_per_step`: Batch size for prompts (default 32)
- `num_generations_per_prompt`: Number of rollouts per prompt (default 16)
- `normalize_rewards`: True (per-prompt normalization)
- `use_leave_one_out_baseline`: True (better variance reduction)
- `max_rollout_turns`: 1 for single-turn, >1 for multi-turn

**Generation Backends Available:**
1. vLLM (primary, most tested)
2. SGLang (alternative)
3. Megatron (for large models)

**Code RL Environments Discovered:**

1. **CodeEnvironment** (`/home/claude/code/RL/nemo_rl/environments/code_environment.py`)
   - Sandboxed Python code execution
   - Uses `<code>...</code>` tags for code blocks
   - Returns execution results as observations
   - Supports multi-turn code generation
   - `terminate_on_evaluation`: Controls when episodes end

2. **NeMo Gym Integration** (`/home/claude/code/RL/nemo_rl/environments/nemo_gym.py`)
   - Wrapper for NeMo Gym environments
   - Async rollout collection
   - Rich resource servers available

3. **NeMo Gym Resource Servers** (in `/home/claude/code/RL/3rdparty/Gym-workspace/Gym/resources_servers/`):
   - `code_gen/`: Competitive coding with unit test verification
   - `mini_swe_agent/`: SWE-Gym integration for software dev
   - `swerl_gen/`: Patch generation with PASS_TO_PASS/FAIL_TO_PASS tests
   - `swerl_llm_judge/`: SWE with LLM judge

**Datasets Available:**
- `nvidia/Nemotron-RL-coding-competitive_coding` (16K examples)
  - Competitive programming problems
  - Unit tests for verification
  - Sources: CodeContests, Codeforces, CodeChef
- `SWE-Gym/SWE-Gym` for software engineering tasks

**Reward Model Support:**
- `RewardModelEnvironment` in `/home/claude/code/RL/nemo_rl/environments/reward_model_environment.py`
- Example: Skywork-Reward-V2-Qwen3-0.6B
- Bradley-Terry reward model type
- Config: `grpo_rm_1B.yaml`

### 2026-02-03 11:00 - Small Model Options for 2 GPU Testing

**Candidate Models for Initial Testing:**
1. `Qwen/Qwen2.5-1.5B` (base config default)
2. `Qwen/Qwen2.5-Math-1.5B-Instruct` (math-focused)
3. `Qwen/Qwen2.5-Coder-0.5B-Instruct` (code-focused, smallest)
4. `Qwen/Qwen2.5-Coder-1.5B-Instruct` (code-focused, balanced)

**Recommended 2 GPU Config:**
- Start with `grpo_math_1B.yaml` as base
- Use `Qwen/Qwen2.5-Coder-1.5B-Instruct` model
- Adjust `cluster.gpus_per_node: 2`
- Reduce sequence length for faster iteration

---

## Code RL Pipeline Design

### Phase 1: Verifiable Code Rewards (No LLM Judge Required)

**Strategy: Unit Test Execution as Reward Signal**

Rewards based on:
1. **Compilation/Syntax Check**: Binary (0/1)
2. **Unit Test Pass Rate**: 0.0 to 1.0 based on % tests passed
3. **Partial Credit**: Can give intermediate rewards for partial solutions

**Implementation Options:**

Option A: Use existing `CodeEnvironment`
```yaml
env:
  code:
    num_workers: 8
    terminate_on_evaluation: true  # Single-turn code completion
```

Option B: Integrate with NeMo Gym code_gen server
- More robust sandboxing
- LiveCodeBench execution backend
- Better for competitive coding

### Phase 2: Multi-Turn Code RL

**For SWE-style tasks:**
1. Use `mini_swe_agent` or `swerl_gen` servers
2. Docker-based sandbox execution
3. Reward = FAIL_TO_PASS test resolution

**Multi-turn trajectory:**
```
User: Bug description + relevant files
Assistant: Analyze + propose fix
Environment: Test results (pass/fail)
Assistant: Refine if needed
...
Final Reward: All tests pass = 1.0, else 0.0
```

### Phase 3: Self-Hosted Reward Model (Optional Enhancement)

If unit tests insufficient:
1. Train reward model on code quality
2. Use `RewardModelEnvironment`
3. Combine with test execution: `total_reward = 0.5 * test_reward + 0.5 * rm_reward`

---

## Proposed Experiment Plan

### Experiment 1: Baseline GRPO with Math (2 GPU)
**Goal:** Verify GRPO setup works
- Model: Qwen2.5-1.5B
- Dataset: OpenMathInstruct-2
- Expected: ~1 hour to see convergence signal

### Experiment 2: Code GRPO with Competitive Coding (2 GPU)
**Goal:** Test code-focused GRPO
- Model: Qwen2.5-Coder-1.5B-Instruct
- Dataset: nvidia/Nemotron-RL-coding-competitive_coding
- Environment: Code execution with unit tests
- Reward: % tests passed

### Experiment 3: Full-Scale Code RL (8 GPU, Post-SFT)
**Goal:** Production training
- Model: SFT checkpoint from Team Impossible
- Dataset: Nemotron-RL competitive coding + SWE-Gym
- Multi-turn enabled
- Async GRPO for throughput

---

## Config Templates

### 2 GPU Test Config (code_grpo_test.yaml)
```yaml
defaults: "grpo_math_1B.yaml"

policy:
  model_name: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
  tokenizer:
    name: ${policy.model_name}
  max_total_sequence_length: 2048
  train_global_batch_size: 256
  train_micro_batch_size: 2
  logprob_batch_size: 2
  generation:
    max_new_tokens: 1024
    vllm_cfg:
      max_model_len: 2048
      gpu_memory_utilization: 0.7

grpo:
  num_prompts_per_step: 16
  num_generations_per_prompt: 8
  max_rollout_turns: 1
  max_num_steps: 100

data:
  train:
    dataset_name: "nvidia/Nemotron-RL-coding-competitive_coding"
    split_validation_size: 0.05
  default:
    processor: "code_hf_data_processor"  # Need to verify/create
    env_name: "code"

env:
  code:
    num_workers: 4
    terminate_on_evaluation: true

cluster:
  gpus_per_node: 2
  num_nodes: 1
```

### 8 GPU Production Config (code_grpo_production.yaml)
```yaml
defaults: "grpo_math_1B.yaml"

policy:
  model_name: "/path/to/sft/checkpoint"  # After SFT phase
  max_total_sequence_length: 4096
  train_global_batch_size: 512
  train_micro_batch_size: 4
  logprob_batch_size: 4

grpo:
  num_prompts_per_step: 32
  num_generations_per_prompt: 16
  max_rollout_turns: 3  # Multi-turn for SWE tasks
  max_num_steps: 1000
  async_grpo:
    enabled: true
    max_trajectory_age_steps: 2

data:
  train:
    dataset_name: "nvidia/Nemotron-RL-coding-competitive_coding"
  default:
    processor: "code_hf_data_processor"
    env_name: "code"

env:
  code:
    num_workers: 16

cluster:
  gpus_per_node: 8
  num_nodes: 1
```

---

## Next Steps

1. **Immediate:** Create test config and run baseline math GRPO
2. **Today:** Verify code environment works with simple examples
3. **This Week:** Full code GRPO experiment on 2 GPUs
4. **After SFT:** Scale to 8 GPUs with SFT checkpoint

## Open Questions

1. Does `code_hf_data_processor` exist or do we need to create one?
   - **Answer:** No, need to create one. Can adapt `math_hf_data_processor`
   - The Nemotron-RL-coding dataset uses `responses_create_params.input[0].content` for problem
   - Unit tests in `verifier_metadata.unit_tests.{inputs, outputs}`

2. How to handle SWE-Gym's Docker/Singularity requirement?
   - **Answer:** Use NeMo Gym `mini_swe_agent` or `swerl_gen` servers
   - Requires Singularity images (may need pre-built images)

3. Best reward shaping for partial code solutions?
   - **Option A:** Binary (all tests pass = 1.0, else 0.0)
   - **Option B:** Fractional (% tests passed)
   - **Recommendation:** Start with Option B, tune reward_scaling config

4. Integration with Team Impossible's SFT checkpoint format?
   - NeMo RL supports loading from `checkpoint_dir/policy/weights`
   - Standard HF format should work directly

### 2026-02-03 11:30 - Data Format Analysis

**Nemotron-RL-coding-competitive_coding Format:**
```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "<problem>"}]
  },
  "verifier_metadata": {
    "unit_tests": {
      "inputs": ["test1_input", "test2_input"],
      "outputs": ["expected1", "expected2"]
    }
  },
  "hash_id": "...",
  "dataset": "taco",
  "source": "codeforces"
}
```

**Required Data Processor:**
Need to create `code_hf_data_processor` that:
1. Extracts problem from `responses_create_params.input[0].content`
2. Stores unit tests in `extra_env_info` for CodeEnvironment
3. Formats with appropriate code prompt template

---

## Final Pipeline Design

### Architecture

```
                    GRPO Training Loop
                           |
           +---------------+---------------+
           |                               |
     Policy Model                   Generation Backend
   (Qwen2.5-Coder)                      (vLLM)
           |                               |
           +---------------+---------------+
                           |
                      Rollouts
                           |
                   CodeEnvironment
                           |
                  +-----------------+
                  |  Unit Test      |
                  |  Execution      |
                  |  (Sandboxed)    |
                  +-----------------+
                           |
                    Reward Signal
                    (0.0 - 1.0)
```

### Reward Computation

```python
def compute_code_reward(code_output: str, unit_tests: dict) -> float:
    """
    Compute reward based on test execution results.

    Returns:
        float: Reward between 0.0 and 1.0
        - 0.0 if code doesn't compile or crashes
        - Fraction of tests passed otherwise
    """
    inputs = unit_tests["inputs"]
    outputs = unit_tests["outputs"]

    passed = 0
    for inp, expected_out in zip(inputs, outputs):
        try:
            actual = execute_code(code_output, inp)
            if actual.strip() == expected_out.strip():
                passed += 1
        except Exception:
            pass

    return passed / len(inputs) if inputs else 0.0
```

### Key Hyperparameters

| Parameter | 2 GPU Test | 8 GPU Production |
|-----------|------------|------------------|
| Model | Qwen2.5-Coder-1.5B | SFT Checkpoint |
| num_prompts_per_step | 16 | 32 |
| num_generations_per_prompt | 8 | 16 |
| train_global_batch_size | 128 | 512 |
| max_sequence_length | 2048 | 4096 |
| max_rollout_turns | 1 | 1-3 |
| learning_rate | 5e-6 | 5e-6 |

---

## Implementation Checklist

### Before Running Experiments

- [ ] Create `code_data_processor` in `nemo_rl/data/processors.py`
- [ ] Create `NemotronRLCodingDataset` in `nemo_rl/data/datasets/response_datasets/`
- [ ] Modify `CodeEnvironment` to compute rewards from unit tests
- [ ] Create 2-GPU test config
- [ ] Create 8-GPU production config
- [ ] Test data loading pipeline
- [ ] Verify CodeEnvironment works end-to-end

### Experiment Sequence

1. **Baseline Math GRPO** (1-2 hours)
   - Verify setup works
   - Check convergence on GSM8K

2. **Code GRPO Small Scale** (4-6 hours)
   - Use 2 GPUs
   - Small subset of competitive coding data
   - Validate reward computation

3. **Full Code GRPO** (After SFT, 8 GPUs)
   - Use SFT checkpoint as base
   - Full competitive coding dataset
   - Multi-turn if applicable

---

## References

- GRPO Paper: https://arxiv.org/abs/2402.03300
- NeMo Gym: https://docs.nvidia.com/nemo/gym/
- Nemotron-RL Coding Dataset: https://huggingface.co/datasets/nvidia/Nemotron-RL-coding-competitive_coding
- SWE-Gym: https://huggingface.co/datasets/SWE-Gym/SWE-Gym

---

### 2026-02-03 12:00 - Research Phase Complete

**STATUS: READY FOR IMPLEMENTATION**

Completed comprehensive research on GRPO for code RL:

1. **Algorithm Understanding**: Deep dive into GRPO implementation in NeMo RL
2. **Environment Options**: Identified CodeEnvironment and NeMo Gym code_gen
3. **Dataset**: nvidia/Nemotron-RL-coding-competitive_coding (16K examples with unit tests)
4. **Reward Strategy**: Unit test execution providing verifiable rewards (no LLM judge needed)
5. **Configs Created**: 2-GPU test and 8-GPU production YAML configs

**Next Action Items:**
1. Run baseline math GRPO to verify setup (2 GPUs)
2. Create code data processor for Nemotron-RL-coding dataset
3. Modify CodeEnvironment for unit test rewards
4. Test end-to-end code GRPO pipeline

**Blocking Issues:** None identified. Ready to proceed with implementation.

**Dependencies:**
- Waiting for SFT checkpoint from team for production training
- May need to coordinate with team on checkpoint format
