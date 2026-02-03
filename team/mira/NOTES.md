# Mira - Data Architect Notes

## Mission
Curate and prepare high-quality SFT datasets for training an empathic software engineering assistant.

## Priority Datasets
1. **allenai/Dolci-Instruct-SFT** - General instruction following
2. **nvidia/OpenMathInstruct-2** - Math reasoning (filter appropriately)
3. Code-focused datasets (to be identified)

## Constraints
- No old/low-quality datasets
- Filter out multimodal examples
- Filter out tool-use examples (no time for that)
- Focus on: code, reasoning, helpful conversation

## Tasks
- [ ] Evaluate Dolci-Instruct-SFT structure and quality
- [ ] Evaluate OpenMathInstruct-2, determine filtering needs
- [ ] Scout for modern code instruction datasets
- [ ] Design blending strategy
- [ ] Build processing pipeline
- [ ] Download and prepare final dataset

## Progress Log
<!-- Add timestamped entries below -->

### 2026-02-03 - Session Start

**Environment Setup Complete**
- HF_TOKEN configured
- Working directory: /home/claude/code/RL

**Analyzed NeMo RL Data Format Requirements:**
1. **OpenAI-style messages format** is the primary format:
   ```json
   {
     "messages": [
       {"role": "user", "content": "..."},
       {"role": "assistant", "content": "..."}
     ]
   }
   ```
2. Data can be loaded via:
   - `OpenAIFormatDataset` - expects JSONL with "messages" key
   - `ResponseDataset` - expects JSONL with input_key/output_key (converts to messages)
   - HuggingFace datasets with proper column mapping
3. `sft_processor` is the standard processor for SFT training
4. Data should be saved as JSONL for custom datasets

**Now evaluating priority datasets...**

### 2026-02-03 - Dataset Evaluation Complete

**Dataset Analysis Summary:**

| Dataset | Size | Format | Quality | Notes |
|---------|------|--------|---------|-------|
| allenai/Dolci-Instruct-SFT | 2.15M | messages | HIGH | 25% coding, 20% math, diverse domains, NO tool-use |
| nvidia/OpenMathInstruct-2 | 14M (1M subset) | problem/solution | HIGH | Math-focused, needs conversion |
| ise-uiuc/Magicoder-Evol-Instruct-110K | 111K | instruction/response | HIGH | Pure code, decontaminated |
| open-r1/codeforces-cots | 48K | messages (CoT) | HIGH | Competitive programming with reasoning |

**Dolci-Instruct-SFT Domain Distribution (10K sample):**
- Coding: 25% (2,533)
- Math: 20% (1,983)
- Other: 20% (1,978)
- Precise IF: 11% (1,083)
- Safety: 8% (818)
- Science: 8% (806)
- Multilingual: 7% (745)
- Chat: 0.5% (55)

**Data Format Compatibility:**
- Dolci: Already has "messages" format - READY
- OpenMathInstruct-2: Needs conversion (problem/solution -> messages)
- Magicoder: Needs conversion (instruction/response -> messages)
- Codeforces-cots: Has "messages" key - READY

### 2026-02-03 - Blending Strategy Design

**Goals:**
1. Strong code/programming ability (primary)
2. Mathematical reasoning
3. Helpful general conversation
4. NO tool-use, NO multimodal

**Proposed Blend (targeting ~500K samples):**
- **Code Focus (50%):**
  - Dolci-Instruct-SFT (Coding domain): ~250K samples
  - Magicoder-Evol-Instruct-110K: ~100K samples (full dataset, minus decontamination)
  - Codeforces-cots: ~48K samples (full dataset)

- **Math/Reasoning (25%):**
  - OpenMathInstruct-2 (train_1M subset, random 125K): ~125K samples

- **General/Safety (25%):**
  - Dolci-Instruct-SFT (non-coding domains, filtered): ~125K samples

**Filters to Apply:**
1. Remove any samples with tool_calls/function_calls (not applicable - already verified none)
2. Remove samples with image references (multimodal)
3. Remove samples > 8192 tokens (efficiency)
4. Deduplicate across datasets

**Building processing pipeline...**
