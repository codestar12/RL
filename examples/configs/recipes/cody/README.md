# Cody's Post-Training Configs

Custom configs for Llama 3 8B post-training experiments.

## Model Details

- **Base model**: Custom Llama 3 8B trained from scratch with torchtitan
- **Context length**: 4096 (no RoPE scaling)
- **Tokenizer**: Llama 3.1 Instruct tokenizer (copied to model dir)

## Setup Checklist

1. Copy instruct tokenizer files to your model checkpoint:
   ```bash
   cp /path/to/llama-3.1-8b-instruct/tokenizer.json /path/to/your/model/
   cp /path/to/llama-3.1-8b-instruct/tokenizer_config.json /path/to/your/model/
   cp /path/to/llama-3.1-8b-instruct/special_tokens_map.json /path/to/your/model/
   ```

2. Update your model's `config.json`:
   ```json
   "eos_token_id": [128001, 128008, 128009]
   ```

3. Update `policy.model_name` in the config files to point to your checkpoint.

## Configs

| Config | Description |
|--------|-------------|
| `sft-llama3-8b-1n8g.yaml` | Basic SFT with DTensor on 1 node, 8 GPUs |

## Running

```bash
uv run python examples/run_sft.py --config examples/configs/recipes/cody/sft-llama3-8b-1n8g.yaml
```
