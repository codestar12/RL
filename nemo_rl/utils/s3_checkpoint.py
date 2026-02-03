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

"""S3 checkpoint streaming utilities for NeMo RL.

This module provides S3-based StorageWriter and StorageReader implementations
for torch.distributed.checkpoint (DCP), enabling direct checkpoint streaming
to S3 during distributed training.

Adapted from TorchTitan's S3 storage implementation.

Example usage:
    from nemo_rl.utils.s3_checkpoint import S3StorageWriter, S3StorageReader, parse_s3_uri

    # Save checkpoint to S3
    writer = S3StorageWriter(
        bucket="my-bucket",
        prefix="checkpoints/step-100",
        region_name="us-west-2",
    )
    dcp.save(state_dict, storage_writer=writer)

    # Load checkpoint from S3
    reader = S3StorageReader(
        bucket="my-bucket",
        prefix="checkpoints/step-100",
        region_name="us-west-2",
    )
    dcp.load(state_dict, storage_reader=reader)
"""

import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import Metadata, StorageReader, StorageWriter
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    SavePlan,
    SavePlanner,
    WriteItemType,
)
from torch.distributed.checkpoint.storage import WriteResult
from torch.distributed.checkpoint.utils import _DistWrapper
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.futures import Future

logger = logging.getLogger(__name__)


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Parse an S3 URI into (bucket, prefix).

    Args:
        uri: S3 URI in format s3://bucket/path/to/prefix

    Returns:
        Tuple of (bucket_name, prefix)

    Raises:
        ValueError: If URI is invalid or doesn't start with s3://

    Example:
        >>> parse_s3_uri("s3://my-bucket/some/path")
        ("my-bucket", "some/path")
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}. Must start with 's3://'")

    parsed = urlparse(uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    if not bucket:
        raise ValueError(f"Invalid S3 URI: {uri}. No bucket specified.")

    return bucket, prefix


class S3StorageWriter(StorageWriter):
    """StorageWriter implementation for AWS S3.

    This writer streams checkpoint data directly to S3 during distributed
    checkpointing. Each WriteItem is stored as a separate S3 object under
    a rank-specific prefix to avoid conflicts.

    Design: One S3 object per WriteItem.
    - S3 key format: {prefix}/rank{rank}/{fqn}.distcp
    - The coordinator rank writes the .metadata file

    Attributes:
        bucket: S3 bucket name
        prefix: S3 key prefix for all checkpoint objects
        region_name: AWS region for the S3 bucket
        storage_class: S3 storage class (default: STANDARD)
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        region_name: Optional[str] = None,
        aws_profile: Optional[str] = None,
        storage_class: str = "STANDARD",
        multipart_threshold: int = 100 * 1024 * 1024,
    ):
        """Initialize S3StorageWriter.

        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix (will be stripped of trailing slashes)
            region_name: AWS region name (optional)
            aws_profile: AWS profile name (optional)
            storage_class: S3 storage class (default: STANDARD)
            multipart_threshold: Size threshold for multipart uploads (bytes)

        Raises:
            ValueError: If bucket doesn't exist or is not accessible
        """
        super().__init__()
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.region_name = region_name
        self.aws_profile = aws_profile
        self.storage_class = storage_class
        self.multipart_threshold = multipart_threshold

        # Import boto3 lazily to avoid hard dependency
        try:
            import boto3
            from botocore.config import Config
            from botocore.exceptions import ClientError, NoCredentialsError
        except ImportError:
            raise ImportError(
                "S3 checkpointing requires boto3. Install with: pip install boto3"
            )

        session_kwargs: Dict[str, Any] = {}
        if aws_profile:
            session_kwargs["profile_name"] = aws_profile
        session = boto3.Session(**session_kwargs)

        config = Config(
            region_name=region_name,
            retries={"max_attempts": 5, "mode": "adaptive"},
            max_pool_connections=50,
        )

        self.s3_client = session.client("s3", config=config)

        # Verify bucket access
        try:
            self.s3_client.head_bucket(Bucket=bucket)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == "404":
                raise ValueError(f"Bucket '{bucket}' does not exist")
            elif code == "403":
                raise ValueError(f"No permission to access bucket '{bucket}'")
            else:
                raise
        except NoCredentialsError:
            raise ValueError(
                "No AWS credentials found. Please configure credentials via "
                "environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) "
                "or AWS profile."
            )

        self.is_coordinator: bool = False
        self.checkpoint_id: Optional[Union[str, os.PathLike]] = None

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        """Set up the storage writer with coordinator status."""
        self.is_coordinator = is_coordinator

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        """Prepare the local save plan (no modifications needed for S3)."""
        return plan

    def prepare_global_plan(self, global_plan: List[SavePlan]) -> List[SavePlan]:
        """Prepare the global save plan (no modifications needed for S3)."""
        return global_plan

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        """Reset the writer for a new checkpoint save."""
        self.checkpoint_id = checkpoint_id

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """Validate checkpoint ID (accepts any non-empty string)."""
        return bool(str(checkpoint_id))

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[List[WriteResult]]:
        """Write all WriteItems in the plan to S3.

        Each WriteItem is serialized and uploaded as a separate S3 object.
        Tensors are moved to CPU, trimmed, and serialized via torch.save().

        Args:
            plan: The save plan containing WriteItems
            planner: The save planner for data resolution

        Returns:
            Future containing list of WriteResult objects
        """
        from botocore.exceptions import BotoCoreError, ClientError

        results: List[WriteResult] = []
        rank = dist.get_rank() if dist.is_initialized() else 0

        for write_item in plan.items:
            # Resolve data from planner
            data = planner.resolve_data(write_item)

            # Unwrap DTensor wrapper if present
            if isinstance(data, _DistWrapper):
                data = data.unwrap()

            # Serialize to bytes
            if write_item.type == WriteItemType.BYTE_IO:
                if isinstance(data, io.BytesIO):
                    buf = data
                    buf.seek(0)
                    body = buf.getvalue()
                elif isinstance(data, (bytes, bytearray)):
                    body = bytes(data)
                else:
                    buf = io.BytesIO()
                    torch.save(data, buf)
                    buf.seek(0)
                    body = buf.getvalue()
            else:
                # TENSOR / SHARD: move to CPU and serialize
                if isinstance(data, torch.Tensor):
                    tensor = data.detach()
                    if tensor.device.type != "cpu":
                        tensor = tensor.to(device="cpu")
                    # Clone if storage doesn't match numel (sparse storage)
                    if tensor.untyped_storage().size() // tensor.element_size() != tensor.numel():
                        tensor = tensor.clone()
                    payload = tensor
                else:
                    payload = data

                buf = io.BytesIO()
                torch.save(payload, buf)
                buf.seek(0)
                body = buf.getvalue()

            size_in_bytes = len(body)

            # Construct S3 key with rank prefix to avoid conflicts
            fqn = write_item.index.fqn
            s3_key = f"{self.prefix}/rank{rank}/{fqn}.distcp"

            # Upload to S3
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket,
                    Key=s3_key,
                    Body=body,
                    StorageClass=self.storage_class,
                )
            except (ClientError, BotoCoreError) as e:
                logger.error(f"Failed to upload {s3_key} to S3: {e}")
                raise

            results.append(
                WriteResult(
                    index=write_item.index,
                    size_in_bytes=size_in_bytes,
                    storage_data=s3_key,
                )
            )

        fut: Future[List[WriteResult]] = Future()
        fut.set_result(results)
        return fut

    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        """Finalize the checkpoint save by writing metadata.

        Called once all ranks have finished write_data. The coordinator
        writes the .metadata file containing storage mappings.

        Args:
            metadata: DCP metadata object
            results: List of WriteResults from all ranks
        """
        import pickle

        from botocore.exceptions import BotoCoreError, ClientError

        storage_md: Dict[MetadataIndex, Any] = {}
        for wr_list in results:
            for wr in wr_list:
                storage_md[wr.index] = wr.storage_data

        metadata.storage_data = storage_md

        # Only coordinator writes the .metadata object
        if not self.is_coordinator:
            return

        buf = io.BytesIO()
        pickle.dump(metadata, buf)
        buf.seek(0)

        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}/.metadata",
                Body=buf.getvalue(),
                StorageClass=self.storage_class,
            )
            logger.info(f"Saved checkpoint metadata to s3://{self.bucket}/{self.prefix}/.metadata")
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to upload metadata to S3: {e}")
            raise


class S3StorageReader(StorageReader):
    """StorageReader implementation for AWS S3.

    This reader loads checkpoint data from S3 during distributed
    checkpoint restoration. It's the logical inverse of S3StorageWriter.

    Attributes:
        bucket: S3 bucket name
        prefix: S3 key prefix for checkpoint objects
        region_name: AWS region for the S3 bucket
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        region_name: Optional[str] = None,
        aws_profile: Optional[str] = None,
    ):
        """Initialize S3StorageReader.

        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix for checkpoint
            region_name: AWS region name (optional)
            aws_profile: AWS profile name (optional)
        """
        super().__init__()
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.region_name = region_name
        self.aws_profile = aws_profile

        # Import boto3 lazily
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise ImportError(
                "S3 checkpointing requires boto3. Install with: pip install boto3"
            )

        session_kwargs: Dict[str, Any] = {}
        if aws_profile:
            session_kwargs["profile_name"] = aws_profile
        session = boto3.Session(**session_kwargs)

        config = Config(
            region_name=region_name,
            retries={"max_attempts": 5, "mode": "adaptive"},
            max_pool_connections=50,
        )

        self.s3_client = session.client("s3", config=config)
        self.storage_data: Dict[MetadataIndex, Any] = {}
        self.checkpoint_id: Optional[Union[str, os.PathLike]] = None

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        """Reset the reader for a new checkpoint load."""
        self.checkpoint_id = checkpoint_id

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """Validate checkpoint ID (accepts any non-empty string)."""
        return bool(str(checkpoint_id))

    def read_metadata(self) -> Metadata:
        """Read the checkpoint metadata from S3.

        Returns:
            DCP Metadata object containing storage mappings

        Raises:
            RuntimeError: If metadata file is not found
        """
        import pickle

        from botocore.exceptions import BotoCoreError, ClientError

        key = f"{self.prefix}/.metadata"
        try:
            resp = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        except self.s3_client.exceptions.NoSuchKey:
            raise RuntimeError(
                f"Metadata file not found in S3: s3://{self.bucket}/{key}"
            )
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to load metadata from S3: {e}")
            raise

        buf = io.BytesIO(resp["Body"].read())
        buf.seek(0)
        metadata: Metadata = pickle.load(buf)
        return metadata

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        """Set up the reader with metadata storage mappings."""
        assert metadata.storage_data is not None
        self.storage_data = metadata.storage_data

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        """Prepare the local load plan (no modifications needed)."""
        return plan

    def prepare_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        """Prepare the global load plan (no modifications needed)."""
        return global_plan

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """Read all ReadItems in the plan from S3.

        For BYTE_IO items, calls planner.load_bytes().
        For TENSOR items, loads via torch.load() and copies to target tensor.

        Args:
            plan: The load plan containing ReadItems
            planner: The load planner for tensor resolution

        Returns:
            Future that resolves when loading is complete
        """
        from botocore.exceptions import BotoCoreError, ClientError

        for read_item in plan.items:
            storage_key = self.storage_data[read_item.storage_index]

            try:
                resp = self.s3_client.get_object(
                    Bucket=self.bucket,
                    Key=storage_key,
                )
            except self.s3_client.exceptions.NoSuchKey:
                raise RuntimeError(
                    f"Checkpoint file not found in S3: s3://{self.bucket}/{storage_key}"
                )
            except (ClientError, BotoCoreError) as e:
                logger.error(f"Failed to load {storage_key} from S3: {e}")
                raise

            raw = resp["Body"].read()

            if read_item.type == LoadItemType.BYTE_IO:
                buf = io.BytesIO(raw)
                buf.seek(0)
                planner.load_bytes(read_item, buf)
            else:
                buf = io.BytesIO(raw)
                buf.seek(0)
                tensor = torch.load(buf, map_location="cpu", weights_only=False)

                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError(
                        f"Expected Tensor for ReadItem {read_item.storage_index}, "
                        f"got {type(tensor)}"
                    )

                tensor = narrow_tensor_by_index(
                    tensor, read_item.storage_offsets, read_item.lengths
                )
                target_tensor = planner.resolve_tensor(read_item).detach()

                if target_tensor.size() != tensor.size():
                    raise RuntimeError(
                        f"Size mismatch for {read_item.storage_index}: "
                        f"target {target_tensor.size()} vs loaded {tensor.size()}"
                    )

                target_tensor.copy_(tensor)
                planner.commit_tensor(read_item, target_tensor)

            logger.debug(f"Successfully loaded {storage_key} from S3")

        fut: Future[None] = Future()
        fut.set_result(None)
        return fut


def save_checkpoint_to_s3(
    state_dict: Dict[str, Any],
    s3_uri: str,
    region_name: Optional[str] = None,
    aws_profile: Optional[str] = None,
    storage_class: str = "STANDARD",
) -> None:
    """High-level function to save a checkpoint directly to S3.

    Args:
        state_dict: State dictionary to save
        s3_uri: Full S3 URI (s3://bucket/prefix)
        region_name: AWS region name (optional)
        aws_profile: AWS profile name (optional)
        storage_class: S3 storage class (default: STANDARD)

    Example:
        >>> save_checkpoint_to_s3(
        ...     {"model": model_state, "optimizer": optim_state},
        ...     "s3://my-bucket/checkpoints/step-100",
        ... )
    """
    bucket, prefix = parse_s3_uri(s3_uri)
    writer = S3StorageWriter(
        bucket=bucket,
        prefix=prefix,
        region_name=region_name,
        aws_profile=aws_profile,
        storage_class=storage_class,
    )
    dcp.save(state_dict, storage_writer=writer)
    logger.info(f"Checkpoint saved to {s3_uri}")


def load_checkpoint_from_s3(
    state_dict: Dict[str, Any],
    s3_uri: str,
    region_name: Optional[str] = None,
    aws_profile: Optional[str] = None,
) -> None:
    """High-level function to load a checkpoint from S3.

    Args:
        state_dict: State dictionary to populate (will be modified in-place)
        s3_uri: Full S3 URI (s3://bucket/prefix)
        region_name: AWS region name (optional)
        aws_profile: AWS profile name (optional)

    Example:
        >>> state_dict = {"model": model_state, "optimizer": optim_state}
        >>> load_checkpoint_from_s3(state_dict, "s3://my-bucket/checkpoints/step-100")
    """
    bucket, prefix = parse_s3_uri(s3_uri)
    reader = S3StorageReader(
        bucket=bucket,
        prefix=prefix,
        region_name=region_name,
        aws_profile=aws_profile,
    )
    dcp.load(state_dict, storage_reader=reader)
    logger.info(f"Checkpoint loaded from {s3_uri}")


def list_s3_checkpoints(
    s3_uri: str,
    region_name: Optional[str] = None,
    aws_profile: Optional[str] = None,
) -> List[str]:
    """List available checkpoints under an S3 prefix.

    Looks for directories containing .metadata files.

    Args:
        s3_uri: S3 URI prefix to search (s3://bucket/prefix)
        region_name: AWS region name (optional)
        aws_profile: AWS profile name (optional)

    Returns:
        List of checkpoint step names (e.g., ["step-100", "step-200"])

    Example:
        >>> checkpoints = list_s3_checkpoints("s3://my-bucket/checkpoints/")
        >>> print(checkpoints)
        ['step-100', 'step-200', 'step-300']
    """
    import re

    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise ImportError(
            "S3 checkpointing requires boto3. Install with: pip install boto3"
        )

    bucket, prefix = parse_s3_uri(s3_uri)
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    session_kwargs: Dict[str, Any] = {}
    if aws_profile:
        session_kwargs["profile_name"] = aws_profile
    session = boto3.Session(**session_kwargs)

    config = Config(
        region_name=region_name,
        retries={"max_attempts": 5, "mode": "adaptive"},
    )
    s3_client = session.client("s3", config=config)

    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            Delimiter="/",
        )
    except Exception as e:
        logger.error(f"Failed to list S3 checkpoints: {e}")
        return []

    if "CommonPrefixes" not in response:
        return []

    # Find step directories with .metadata files
    checkpoints = []
    pattern = r"step-(\d+)"

    for prefix_info in response["CommonPrefixes"]:
        dir_prefix = prefix_info["Prefix"]
        dir_name = dir_prefix.rstrip("/").split("/")[-1]

        if re.match(pattern, dir_name):
            # Check if .metadata exists
            metadata_key = f"{dir_prefix}.metadata"
            try:
                s3_client.head_object(Bucket=bucket, Key=metadata_key)
                checkpoints.append(dir_name)
            except:
                pass  # No valid metadata, skip

    # Sort by step number
    checkpoints.sort(key=lambda x: int(re.search(r"step-(\d+)", x).group(1)))
    return checkpoints


def get_latest_s3_checkpoint(
    s3_uri: str,
    region_name: Optional[str] = None,
    aws_profile: Optional[str] = None,
) -> Optional[str]:
    """Get the latest checkpoint URI under an S3 prefix.

    Args:
        s3_uri: S3 URI prefix to search (s3://bucket/prefix)
        region_name: AWS region name (optional)
        aws_profile: AWS profile name (optional)

    Returns:
        Full S3 URI to the latest checkpoint, or None if no checkpoints found

    Example:
        >>> latest = get_latest_s3_checkpoint("s3://my-bucket/checkpoints/")
        >>> print(latest)
        's3://my-bucket/checkpoints/step-300'
    """
    checkpoints = list_s3_checkpoints(s3_uri, region_name, aws_profile)
    if not checkpoints:
        return None

    bucket, prefix = parse_s3_uri(s3_uri)
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    return f"s3://{bucket}/{prefix}{checkpoints[-1]}"
