# Atlas - Infrastructure Engineer Notes

## Mission
Build S3 checkpoint streaming for training, designed for multi-node scalability.

## Critical Constraints
- **ONLY S3 PATH**: `s3://datology-research/cody/team-impossible/`
- **NO DELETIONS** from S3
- AWS credentials in `team/.env`

## Reference
- TorchTitan S3 implementation: `/home/claude/code/torchtitan/`
- Study their checkpoint streaming approach

## Goals
- Stream checkpoints directly to S3 during training
- Minimal impact on training time
- Each rank efficiently stores its components
- Multi-node ready design
- Bonus: HF conversion on-the-fly

## Tasks
- [ ] Study torchtitan S3 checkpoint implementation
- [ ] Review NeMo RL's current checkpointing code
- [ ] Design S3 streaming architecture
- [ ] Implement checkpoint-to-S3 feature
- [ ] Test with small training run
- [ ] Document for team use

## Progress Log
<!-- Add timestamped entries below -->
