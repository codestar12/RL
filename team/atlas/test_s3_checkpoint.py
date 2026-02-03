#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test script for S3 checkpoint functionality.

This script verifies that S3 checkpointing works correctly by:
1. Testing S3 URI parsing
2. Testing S3 bucket connectivity
3. Saving a small test checkpoint to S3
4. Loading the checkpoint back from S3
5. Verifying data integrity

Usage:
    source team/.env  # Load AWS credentials
    python team/atlas/test_s3_checkpoint.py

Note: Uses the Team Impossible S3 path: s3://datology-research/cody/team-impossible/
"""

import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_parse_s3_uri():
    """Test S3 URI parsing."""
    from nemo_rl.utils.s3_checkpoint import parse_s3_uri

    print("\n=== Testing S3 URI Parsing ===")

    # Test valid URI
    bucket, prefix = parse_s3_uri("s3://my-bucket/path/to/checkpoint")
    assert bucket == "my-bucket", f"Expected 'my-bucket', got '{bucket}'"
    assert prefix == "path/to/checkpoint", f"Expected 'path/to/checkpoint', got '{prefix}'"
    print("  Valid URI parsing: PASS")

    # Test URI with no prefix
    bucket, prefix = parse_s3_uri("s3://my-bucket")
    assert bucket == "my-bucket"
    assert prefix == ""
    print("  URI without prefix: PASS")

    # Test invalid URI
    try:
        parse_s3_uri("http://not-s3/path")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  Invalid URI detection: PASS")

    print("S3 URI parsing tests: ALL PASS")


def test_s3_connectivity():
    """Test S3 bucket connectivity."""
    from nemo_rl.utils.s3_checkpoint import S3StorageWriter

    print("\n=== Testing S3 Connectivity ===")

    # Use the Team Impossible S3 path
    bucket = "datology-research"
    prefix = "cody/team-impossible/atlas-test"

    try:
        writer = S3StorageWriter(
            bucket=bucket,
            prefix=prefix,
        )
        print(f"  Connected to s3://{bucket}/{prefix}: PASS")
        return True
    except ValueError as e:
        print(f"  Connection failed: {e}")
        return False
    except ImportError as e:
        print(f"  boto3 not installed: {e}")
        return False


def test_checkpoint_save_load():
    """Test saving and loading a checkpoint to/from S3."""
    import torch

    from nemo_rl.utils.s3_checkpoint import (
        S3StorageReader,
        S3StorageWriter,
        list_s3_checkpoints,
        load_checkpoint_from_s3,
        save_checkpoint_to_s3,
    )

    print("\n=== Testing Checkpoint Save/Load ===")

    # Create test data
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    test_prefix = f"cody/team-impossible/atlas-test/checkpoint-{timestamp}"
    s3_uri = f"s3://datology-research/{test_prefix}"

    print(f"  Test URI: {s3_uri}")

    # Create a simple state dict
    state_dict = {
        "model_weights": torch.randn(100, 100),
        "optimizer_state": {"step": torch.tensor(42), "lr": torch.tensor(0.001)},
        "metadata": {"epoch": 5, "loss": 0.123},
    }

    print("  Saving checkpoint to S3...")
    start_time = time.time()
    save_checkpoint_to_s3(state_dict, s3_uri)
    save_time = time.time() - start_time
    print(f"  Save completed in {save_time:.2f}s: PASS")

    # Load the checkpoint back
    print("  Loading checkpoint from S3...")
    loaded_state = {
        "model_weights": torch.zeros(100, 100),
        "optimizer_state": {"step": torch.tensor(0), "lr": torch.tensor(0.0)},
        "metadata": {},
    }

    start_time = time.time()
    load_checkpoint_from_s3(loaded_state, s3_uri)
    load_time = time.time() - start_time
    print(f"  Load completed in {load_time:.2f}s")

    # Verify data integrity
    print("  Verifying data integrity...")
    assert torch.allclose(state_dict["model_weights"], loaded_state["model_weights"]), "Model weights mismatch"
    print("    Model weights: MATCH")

    assert torch.equal(state_dict["optimizer_state"]["step"], loaded_state["optimizer_state"]["step"]), "Step mismatch"
    assert torch.allclose(
        state_dict["optimizer_state"]["lr"], loaded_state["optimizer_state"]["lr"]
    ), "Learning rate mismatch"
    print("    Optimizer state: MATCH")

    print("  Data integrity verification: PASS")

    # List checkpoints
    print("\n  Listing available checkpoints...")
    checkpoints = list_s3_checkpoints("s3://datology-research/cody/team-impossible/atlas-test/")
    print(f"    Found {len(checkpoints)} checkpoint(s)")

    print("\nCheckpoint Save/Load tests: ALL PASS")
    return True


def test_distributed_checkpoint():
    """Test distributed checkpoint scenario (single rank simulation)."""
    import torch
    import torch.distributed.checkpoint as dcp

    from nemo_rl.utils.s3_checkpoint import S3StorageReader, S3StorageWriter

    print("\n=== Testing DCP Integration ===")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bucket = "datology-research"
    prefix = f"cody/team-impossible/atlas-test/dcp-test-{timestamp}"

    # Create test state
    state_dict = {
        "layer1": torch.randn(50, 50),
        "layer2": torch.randn(100, 100),
    }

    print(f"  Using prefix: {prefix}")

    # Save using DCP with S3 writer
    print("  Saving with DCP + S3StorageWriter...")
    writer = S3StorageWriter(bucket=bucket, prefix=prefix)
    dcp.save(state_dict, storage_writer=writer)
    print("  DCP save: PASS")

    # Load using DCP with S3 reader
    print("  Loading with DCP + S3StorageReader...")
    loaded_state = {
        "layer1": torch.zeros(50, 50),
        "layer2": torch.zeros(100, 100),
    }
    reader = S3StorageReader(bucket=bucket, prefix=prefix)
    dcp.load(loaded_state, storage_reader=reader)
    print("  DCP load: PASS")

    # Verify
    assert torch.allclose(state_dict["layer1"], loaded_state["layer1"])
    assert torch.allclose(state_dict["layer2"], loaded_state["layer2"])
    print("  Data verification: PASS")

    print("\nDCP Integration tests: ALL PASS")


def main():
    """Run all tests."""
    print("=" * 60)
    print("S3 Checkpoint Testing Suite")
    print("=" * 60)

    # Test URI parsing (no S3 connection needed)
    test_parse_s3_uri()

    # Test S3 connectivity
    if not test_s3_connectivity():
        print("\nSkipping remaining tests due to S3 connectivity failure.")
        print("Make sure AWS credentials are configured:")
        print("  export AWS_ACCESS_KEY_ID=...")
        print("  export AWS_SECRET_ACCESS_KEY=...")
        print("Or: source team/.env")
        return 1

    # Run checkpoint tests
    try:
        test_checkpoint_save_load()
        test_distributed_checkpoint()
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
