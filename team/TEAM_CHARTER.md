# Team Impossible - Charter

## Mission
Transform a Llama 8B base model into an empathic, intelligent software engineering assistant through post-training (SFT + RL).

## Critical Constraints
- **S3 Path**: `s3://datology-research/cody/team-impossible/` - ONLY allowed path
- **NO S3 DELETIONS** under any circumstances
- **Source Checkpoint (DCP)**: `s3://datology-research/cody/torchtitan_ckpts/math_and_code/v2/8b/phase3_1T/step-127120/`
- **Source Checkpoint (HF)**: `s3://datology-research/cody/torchtitan_ckpts/math_and_code/v2/8b/phase3_1T/hf/step-127120/`

## Resources
- 8x NVIDIA H100 80GB GPUs
- Credentials in `team/.env` (gitignored)

## Timeline
- Hours 0-8: Setup, data prep, performance tuning, RL testing (2 GPUs for Zara)
- Hours 8-20+: YOLO SFT run (all 8 GPUs)
- Post-SFT: RL training phase

## Team Members

### Mira - Data Architect
- Evaluate and blend modern SFT datasets
- Filter for non-multimodal, non-tool-use examples
- Build efficient data processing pipelines
- **Deliverable**: Downloaded, blended, training-ready dataset

### Kai - Performance Engineer
- Design high-MFU FP8 training config
- Quick hyperparameter testing
- Optimize throughput on 8 GPUs
- **Deliverable**: Battle-tested SFT config

### Atlas - Infrastructure Engineer
- Build S3 checkpoint streaming during training
- Study torchtitan S3 work for inspiration (one folder up: /home/claude/code/torchtitan)
- Design for multi-node readiness
- **Deliverable**: S3 checkpointing working

### Zara - RL Researcher
- Test GRPO on 2 GPUs during setup phase
- Start with GSM8K, explore code RL
- Focus on multi-turn RL
- **Deliverable**: RL pipeline ready for post-SFT

### Theo - Eval Engineer
- Set up evaluation pipeline
- Focus on code evaluation capabilities
- **Deliverable**: Eval suite ready

## Coordination
- Each agent maintains notes in their folder
- Escalate blockers to Nexus (coordinator)
- All work must be git-tracked
