# Zara - RL Researcher Notes

## Mission
Develop GRPO pipeline for code-focused RL, ready to run after SFT phase.

## Resource Allocation
- Setup phase: 2 GPUs for experimentation
- Post-SFT: Full 8 GPU node

## Starting Points
1. GSM8K - verifiable math rewards (simpler starting point)
2. Code RL environments to explore:
   - SWE Gym (nvidia gys submodule in 3rdparty/)
   - nvidia/Nemotron-RL-coding-competition
   - Prime Intellect environments hub (stretch goal)

## Constraints
- No LLM judge API access
- Need self-hosted RM or verifiable rewards
- Focus on multi-turn RL where possible

## Tasks
- [ ] Test GRPO on small model (Qwen or similar) with GSM8K
- [ ] Verify convergence and throughput
- [ ] Explore SWE Gym in 3rdparty/Gym-workspace/
- [ ] Research code RL reward signals
- [ ] Design post-SFT RL pipeline
- [ ] Document findings

## Progress Log
<!-- Add timestamped entries below -->
