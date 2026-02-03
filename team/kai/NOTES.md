# Kai - Performance Engineer Notes

## Mission
Design and validate a high-MFU FP8 training configuration for SFT on 8x H100 80GB.

## Goals
- Maximize throughput (tokens/sec)
- FP8 training for efficiency
- Modern, high-quality setup
- Quick validation (limited HP tuning time)

## Hardware
- 8x NVIDIA H100 80GB HBM3
- Single node

## Tasks
- [ ] Review existing SFT configs in repo
- [ ] Identify FP8 training options
- [ ] Design optimal batch size / gradient accumulation
- [ ] Test config on small runs
- [ ] Validate MFU numbers
- [ ] Finalize YOLO config

## Key Metrics to Track
- MFU (Model FLOPs Utilization)
- Tokens per second
- Memory utilization
- Training stability

## Progress Log
<!-- Add timestamped entries below -->
