#!/usr/bin/env python3
"""
SFT Dataset Preparation Script for Empathic Software Engineering Assistant
Author: Mira (Data Architect, Team Impossible)

This script downloads, filters, converts, and blends datasets into a single
training-ready JSONL file compatible with NeMo RL's OpenAI format.

Target blend (~500K samples):
- Code Focus (50%): Dolci coding, Magicoder, Codeforces
- Math/Reasoning (25%): OpenMathInstruct-2 subset
- General/Safety (25%): Dolci non-coding domains
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset
from tqdm import tqdm

# Configuration
OUTPUT_DIR = Path("/home/claude/code/RL/team/mira/data")
HF_CACHE = Path("/home/claude/code/RL/team/mira/cache")
SEED = 42
MAX_TOKENS_ESTIMATE = 8192  # Rough token limit (chars / 4)
MAX_CHARS = MAX_TOKENS_ESTIMATE * 4

random.seed(SEED)


@dataclass
class DatasetConfig:
    name: str
    hf_path: str
    split: str
    max_samples: int
    domain_filter: list[str] | None = None
    exclude_domains: list[str] | None = None


# Dataset configurations
DATASETS = [
    # Code-focused datasets
    DatasetConfig(
        name="dolci_coding",
        hf_path="allenai/Dolci-Instruct-SFT",
        split="train",
        max_samples=250_000,
        domain_filter=["Coding"],
    ),
    DatasetConfig(
        name="magicoder",
        hf_path="ise-uiuc/Magicoder-Evol-Instruct-110K",
        split="train",
        max_samples=100_000,
    ),
    DatasetConfig(
        name="codeforces_cots",
        hf_path="open-r1/codeforces-cots",
        split="train",
        max_samples=50_000,
    ),
    # Math/Reasoning
    DatasetConfig(
        name="openmathinstruct2",
        hf_path="nvidia/OpenMathInstruct-2",
        split="train_1M",  # Use 1M subset for efficiency
        max_samples=125_000,
    ),
    # General/Safety
    DatasetConfig(
        name="dolci_general",
        hf_path="allenai/Dolci-Instruct-SFT",
        split="train",
        max_samples=125_000,
        exclude_domains=["Coding"],  # Everything except coding
    ),
]


def is_multimodal(messages: list[dict]) -> bool:
    """Check if messages contain multimodal content."""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            # List content may contain images
            for item in content:
                if isinstance(item, dict) and item.get("type") in ["image", "image_url"]:
                    return True
        elif isinstance(content, str):
            # Check for image markers
            if "<image>" in content or "[image]" in content:
                return True
    return False


def has_tool_calls(messages: list[dict]) -> bool:
    """Check if messages contain tool/function calls."""
    for msg in messages:
        if msg.get("function_call") or msg.get("tool_calls"):
            return True
        if msg.get("role") == "function":
            return True
    return False


def estimate_length(messages: list[dict]) -> int:
    """Estimate total character length of messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    total += len(item["text"])
    return total


def convert_to_openai_format(sample: dict, source: str) -> dict | None:
    """Convert various formats to OpenAI messages format.

    Returns None if sample should be filtered out.
    """
    messages = None

    if source == "dolci_coding" or source == "dolci_general":
        # Dolci already has messages format
        raw_messages = sample.get("messages", [])
        if not raw_messages:
            return None

        # Clean up Dolci messages (remove function_calls/functions keys if None)
        messages = []
        for msg in raw_messages:
            clean_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
            messages.append(clean_msg)

    elif source == "magicoder":
        # instruction/response format
        instruction = sample.get("instruction", "")
        response = sample.get("response", "")
        if not instruction or not response:
            return None
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]

    elif source == "codeforces_cots":
        # Already has messages, but also has generation with CoT
        if "messages" in sample and sample["messages"]:
            messages = []
            for msg in sample["messages"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        else:
            # Fallback to prompt/generation
            prompt = sample.get("prompt", "")
            generation = sample.get("generation", "")
            if not prompt or not generation:
                return None
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": generation}
            ]

    elif source == "openmathinstruct2":
        # problem/generated_solution format
        problem = sample.get("problem", "")
        solution = sample.get("generated_solution", "")
        if not problem or not solution:
            return None
        messages = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution}
        ]

    if not messages:
        return None

    # Apply filters
    if is_multimodal(messages):
        return None

    if has_tool_calls(messages):
        return None

    # Check length
    if estimate_length(messages) > MAX_CHARS:
        return None

    # Ensure assistant is last message
    if messages[-1]["role"] != "assistant":
        return None

    return {"messages": messages, "source": source}


def process_dataset(config: DatasetConfig) -> Iterator[dict]:
    """Process a single dataset and yield converted samples."""
    print(f"\nProcessing {config.name} from {config.hf_path}...")

    os.environ["HF_HOME"] = str(HF_CACHE)

    try:
        ds = load_dataset(config.hf_path, split=config.split, streaming=True)
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return

    count = 0
    filtered_domain = 0
    filtered_other = 0

    for sample in tqdm(ds, desc=f"  {config.name}", total=config.max_samples):
        # Domain filtering for Dolci
        if config.domain_filter:
            domain = sample.get("domain", "")
            if domain not in config.domain_filter:
                filtered_domain += 1
                continue

        if config.exclude_domains:
            domain = sample.get("domain", "")
            if domain in config.exclude_domains:
                filtered_domain += 1
                continue

        # Convert to OpenAI format
        converted = convert_to_openai_format(sample, config.name)
        if converted is None:
            filtered_other += 1
            continue

        yield converted
        count += 1

        if count >= config.max_samples:
            break

    print(f"  Processed {count:,} samples (domain filtered: {filtered_domain:,}, other filtered: {filtered_other:,})")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SFT Dataset Preparation for Empathic Software Engineering Assistant")
    print("=" * 60)

    all_samples = []

    for config in DATASETS:
        samples = list(process_dataset(config))
        all_samples.extend(samples)
        print(f"  Total so far: {len(all_samples):,}")

    # Shuffle
    print(f"\nShuffling {len(all_samples):,} samples...")
    random.shuffle(all_samples)

    # Split into train/val (95/5)
    val_size = int(len(all_samples) * 0.05)
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]

    # Save
    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"

    print(f"\nSaving training data to {train_path}...")
    with open(train_path, "w") as f:
        for sample in tqdm(train_samples, desc="  Writing train"):
            f.write(json.dumps(sample) + "\n")

    print(f"Saving validation data to {val_path}...")
    with open(val_path, "w") as f:
        for sample in tqdm(val_samples, desc="  Writing val"):
            f.write(json.dumps(sample) + "\n")

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Training samples: {len(train_samples):,}")
    print(f"Validation samples: {len(val_samples):,}")
    print(f"Total samples: {len(all_samples):,}")
    print(f"\nOutput files:")
    print(f"  {train_path}")
    print(f"  {val_path}")

    # Source distribution
    print("\nSource distribution:")
    source_counts = {}
    for s in all_samples:
        src = s.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_samples) * 100
        print(f"  {src}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
