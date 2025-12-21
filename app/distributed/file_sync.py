"""Distributed file synchronization for multi-rank model loading.

This module transfers model files from rank 0 to worker ranks using the
mlx.launch distributed backend (JACCL, ring, or MPI). It leverages the same
all_sum() broadcast pattern used for token coordination.

Key features:
- Two-phase transfer: metadata first, then weight files
- Sharded mode: only transfer files each rank needs (pipeline parallelism)
- Disk space validation before large transfers
- Chunked transfers to stay within MPI limits
- Backend-agnostic: works over JACCL/TB5 RDMA, ring/TCP, or MPI

The transfer uses all_sum() with zero-contribution pattern:
- Rank 0 contributes actual file data
- Workers contribute zeros
- Result: data + 0 + 0 + ... = data (everyone gets rank 0's data)
"""

import hashlib
import json
import os
import shutil
import socket
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
from loguru import logger

# Chunk size limits by backend
CHUNK_SIZE_MPI = 256 * 1024 * 1024      # 256 MB (MPI int32 limit safety margin)
CHUNK_SIZE_RING = 512 * 1024 * 1024     # 512 MB (balance blocking time vs overhead)
CHUNK_SIZE_JACCL = 1024 * 1024 * 1024   # 1 GB (TB5 RDMA handles large chunks well)

# Maximum filename length for manifest broadcast
MAX_FILENAME_LENGTH = 256

# Maximum files in manifest (should be plenty for any model)
MAX_MANIFEST_FILES = 100


def get_node_prefix(rank: int) -> str:
    """Get node/rank prefix for log messages.

    When running under mlx.launch, includes hostname for easier identification.

    Args:
        rank: Rank number

    Returns:
        Log prefix like "[misha/0]" or "[Rank 0]"
    """
    # Check if running under mlx.launch (MLX_RANK env var is set)
    if os.getenv("MLX_RANK") is not None:
        hostname = socket.gethostname().split(".")[0]  # Short hostname
        return f"[{hostname}/{rank}]"
    else:
        return f"[Rank {rank}]"


def detect_backend() -> str:
    """Detect which mlx.launch backend is in use.

    Uses heuristics based on environment variables set by mlx.launch.

    Returns:
        Backend name: "jaccl", "ring", "mpi", or "unknown"
    """
    # Check for explicit environment variable override
    backend = os.getenv("MLX_DISTRIBUTED_BACKEND", "").lower()
    if backend in ("jaccl", "ring", "mpi", "nccl"):
        return backend

    # Check for JACCL-specific environment variable
    # mlx.launch sets MLX_JACCL_COORDINATOR for jaccl backend
    if os.getenv("MLX_JACCL_COORDINATOR"):
        return "jaccl"

    # Check for MPI environment variables
    if any(key.startswith(("OMPI_", "PMI_", "MPI_")) for key in os.environ):
        return "mpi"

    # Check for ring backend (uses MLX_HOSTFILE without JACCL coordinator)
    if os.getenv("MLX_HOSTFILE"):
        return "ring"

    # Unable to detect
    return "unknown"


def get_chunk_size(backend: Optional[str] = None) -> int:
    """Get optimal chunk size based on backend.

    Args:
        backend: Backend name or None to auto-detect

    Returns:
        Chunk size in bytes
    """
    # Allow environment variable override
    override = os.getenv("MLX_FILE_SYNC_CHUNK_SIZE")
    if override:
        try:
            return int(override)
        except ValueError:
            logger.warning(
                f"Invalid MLX_FILE_SYNC_CHUNK_SIZE={override}, using default"
            )

    # Detect backend if not provided
    if backend is None:
        backend = detect_backend()

    # Select chunk size based on backend
    chunk_sizes = {
        "jaccl": CHUNK_SIZE_JACCL,   # 1 GB - TB5 RDMA handles it
        "ring": CHUNK_SIZE_RING,     # 512 MB - balance overhead vs blocking
        "mpi": CHUNK_SIZE_MPI,       # 256 MB - MPI int32 limit
        "nccl": CHUNK_SIZE_MPI,      # 256 MB - conservative
        "unknown": CHUNK_SIZE_MPI,   # 256 MB - conservative fallback
    }

    chunk_size = chunk_sizes.get(backend, CHUNK_SIZE_MPI)
    logger.debug(
        f"Using chunk size {chunk_size / 1024 / 1024:.0f}MB for backend={backend}"
    )
    return chunk_size


class FileSyncError(Exception):
    """Base exception for file sync errors."""
    pass


class DiskSpaceError(FileSyncError):
    """Raised when destination has insufficient disk space."""
    pass


class TransferError(FileSyncError):
    """Raised when file transfer fails."""
    pass


def get_cache_path(model_id: str) -> Path:
    """Get local cache path matching HuggingFace structure.

    Args:
        model_id: HuggingFace model ID (e.g., "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

    Returns:
        Path to local cache directory for this model
    """
    # Handle both "org/name" format and local paths
    if "/" in model_id and not os.path.exists(model_id):
        # HuggingFace format: mlx-community/Qwen2.5-0.5B-Instruct-4bit
        # → ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit
        parts = model_id.split("/")
        if len(parts) == 2:
            org, name = parts
            return Path.home() / ".cache/huggingface/hub" / f"models--{org}--{name}" / "snapshots" / "main"

    # Local path or already resolved
    return Path(model_id)


def check_disk_space(cache_path: Path, required_bytes: int) -> tuple[bool, int]:
    """Check if destination disk has sufficient space.

    Args:
        cache_path: Path where files will be written
        required_bytes: Total bytes needed for transfer

    Returns:
        Tuple of (has_enough_space, available_bytes)
    """
    # Ensure parent directory exists for disk_usage check
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    usage = shutil.disk_usage(cache_path.parent)
    # Require 10% buffer for safety
    has_space = usage.free >= required_bytes * 1.1
    return has_space, usage.free


def get_model_files(model_path: Path) -> list[tuple[str, int]]:
    """Get list of model files with sizes.

    Args:
        model_path: Path to model directory

    Returns:
        List of (filename, size_bytes) tuples
    """
    files = []

    # Patterns for model files
    patterns = [
        "*.json",
        "*.safetensors",
        "*.model",  # tokenizer
        "*.txt",    # vocab
        "*.py",     # custom code
        "*.jinja",  # chat templates
    ]

    for pattern in patterns:
        for filepath in model_path.glob(pattern):
            if filepath.is_file():
                files.append((filepath.name, filepath.stat().st_size))

    return sorted(files)


def broadcast_manifest(
    files: Optional[list[tuple[str, int]]],
    group: mx.distributed.Group,
) -> list[tuple[str, int]]:
    """Broadcast file manifest from rank 0 to all ranks.

    Rank 0 provides the file list; workers receive it via all_sum.

    Args:
        files: List of (filename, size) tuples (rank 0 only, None for workers)
        group: MLX distributed group

    Returns:
        File manifest on all ranks
    """
    rank = group.rank()

    # Broadcast file count first
    if rank == 0:
        count = mx.array([len(files)], dtype=mx.int32)
    else:
        count = mx.zeros((1,), dtype=mx.int32)

    count_result = mx.distributed.all_sum(count, group=group)
    mx.eval(count_result)
    file_count = int(count_result[0].item())

    if file_count == 0:
        return []

    logger.debug(f"{get_node_prefix(rank)} Manifest has {file_count} files")

    # Broadcast each file entry: name (as bytes) + size
    # Pack as: [name_length, size, name_bytes...]
    manifest = []

    for i in range(file_count):
        # Broadcast name length and file size
        if rank == 0:
            name = files[i][0].encode('utf-8')
            name_len = len(name)
            file_size = files[i][1]
            header = mx.array([name_len, file_size], dtype=mx.int64)
        else:
            header = mx.zeros((2,), dtype=mx.int64)

        header_result = mx.distributed.all_sum(header, group=group)
        mx.eval(header_result)
        name_len = int(header_result[0].item())
        file_size = int(header_result[1].item())

        # Broadcast filename bytes
        if rank == 0:
            name_bytes = files[i][0].encode('utf-8')
            # Pad to fixed length for all_sum
            padded = np.zeros(MAX_FILENAME_LENGTH, dtype=np.uint8)
            padded[:len(name_bytes)] = list(name_bytes)
            name_array = mx.array(padded)
        else:
            name_array = mx.zeros((MAX_FILENAME_LENGTH,), dtype=mx.uint8)

        name_result = mx.distributed.all_sum(name_array, group=group)
        mx.eval(name_result)

        # Decode filename
        name_bytes = bytes(name_result[:name_len].tolist())
        filename = name_bytes.decode('utf-8')

        manifest.append((filename, file_size))

    return manifest


def transfer_file(
    src_path: Optional[Path],
    dst_path: Path,
    file_size: int,
    group: mx.distributed.Group,
    chunk_size: int,
) -> None:
    """Transfer a single file from rank 0 to all workers.

    Uses chunked all_sum transfers to handle large files.

    Args:
        src_path: Source file path (rank 0 only, None for workers)
        dst_path: Destination file path
        file_size: Expected file size in bytes
        group: MLX distributed group
        chunk_size: Size of each transfer chunk in bytes
    """
    rank = group.rank()

    # Calculate chunks
    num_chunks = (file_size + chunk_size - 1) // chunk_size

    logger.debug(
        f"{get_node_prefix(rank)} Transferring {dst_path.name}: "
        f"{file_size / 1e6:.1f}MB in {num_chunks} chunks "
        f"({chunk_size / 1e6:.0f}MB each)"
    )

    # Open files
    if rank == 0:
        src_file = open(src_path, 'rb')

    # Workers prepare destination
    if rank != 0:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_file = open(dst_path, 'wb')

    try:
        bytes_transferred = 0

        for chunk_idx in range(num_chunks):
            # Calculate chunk size (last chunk may be smaller)
            remaining = file_size - bytes_transferred
            this_chunk_size = min(chunk_size, remaining)

            # Read and broadcast chunk
            if rank == 0:
                data = src_file.read(this_chunk_size)
                # Pad to chunk size for consistent all_sum
                if len(data) < chunk_size:
                    data = data + b'\x00' * (chunk_size - len(data))
                chunk = mx.array(np.frombuffer(data, dtype=np.uint8))
            else:
                chunk = mx.zeros((chunk_size,), dtype=mx.uint8)

            # Transfer via all_sum
            result = mx.distributed.all_sum(chunk, group=group)
            mx.eval(result)

            # Workers write chunk
            if rank != 0:
                chunk_data = bytes(result[:this_chunk_size].tolist())
                dst_file.write(chunk_data)

            bytes_transferred += this_chunk_size

            if (chunk_idx + 1) % 10 == 0 or chunk_idx == num_chunks - 1:
                pct = bytes_transferred / file_size * 100
                logger.debug(f"{get_node_prefix(rank)} {dst_path.name}: {pct:.0f}%")

    finally:
        if rank == 0:
            src_file.close()
        if rank != 0:
            dst_file.close()

    # Verify file size on workers
    if rank != 0:
        actual_size = dst_path.stat().st_size
        if actual_size != file_size:
            raise TransferError(
                f"Size mismatch for {dst_path.name}: "
                f"expected {file_size}, got {actual_size}"
            )


def get_metadata_files(manifest: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Extract metadata files from manifest (small files needed by all ranks)."""
    metadata_patterns = (
        ".json",      # config.json, tokenizer.json, etc.
        ".txt",       # vocab files
        ".model",     # sentencepiece models
        ".jinja",     # chat templates
        ".py",        # custom code
    )
    return [(name, size) for name, size in manifest
            if any(name.endswith(ext) for ext in metadata_patterns)]


def get_weight_files(manifest: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Extract weight files from manifest (large safetensor files)."""
    return [(name, size) for name, size in manifest if name.endswith(".safetensors")]


def compute_pipeline_files(
    index_path: Path,
    config_path: Path,
    rank: int,
    world_size: int,
) -> set[str]:
    """Compute which weight files a rank needs for pipeline parallelism.

    Uses the same layer assignment logic as mlx_lm's pipeline() method.
    """
    # Read config to get layer count
    with open(config_path, "r") as f:
        config = json.load(f)

    num_layers = config.get("num_hidden_layers", 0)
    if num_layers == 0:
        # Can't determine layers, fall back to all files
        return None

    # Read weight index
    if not index_path.exists():
        # No index file, can't do sharded loading
        return None

    with open(index_path, "r") as f:
        weight_index = json.load(f).get("weight_map", {})

    if not weight_index:
        return None

    # Compute layer assignment (same as PipelineMixin)
    # Layers are distributed evenly across ranks
    layers_per_rank = num_layers // world_size
    start_layer = rank * layers_per_rank
    end_layer = start_layer + layers_per_rank if rank < world_size - 1 else num_layers

    # Find files containing parameters for our layers
    needed_files = set()
    for param_name, file_name in weight_index.items():
        # Check if this parameter belongs to our layers
        # Format: model.layers.{N}.* or similar
        if ".layers." in param_name:
            try:
                # Extract layer number
                parts = param_name.split(".layers.")
                if len(parts) >= 2:
                    layer_num = int(parts[1].split(".")[0])
                    if start_layer <= layer_num < end_layer:
                        needed_files.add(file_name)
            except (ValueError, IndexError):
                # Can't parse layer number, include file to be safe
                needed_files.add(file_name)
        else:
            # Non-layer parameter (embeddings, lm_head, etc.)
            # These are typically needed by rank 0 and last rank
            if rank == 0 or rank == world_size - 1:
                needed_files.add(file_name)

    return needed_files


def resolve_worker_path(model_path: str, worker_model_path: Optional[str]) -> Path:
    """Resolve the model path for worker ranks.

    Args:
        model_path: Original model path from rank 0
        worker_model_path: Override path, "hf-cache", or None

    Returns:
        Path where workers should store/load the model
    """
    if worker_model_path is None:
        # Default: same path as rank 0
        return Path(model_path)
    elif worker_model_path == "hf-cache":
        # Use HuggingFace cache structure
        return get_cache_path(model_path)
    else:
        # Explicit path override
        return Path(worker_model_path)


def sync_model_to_workers(
    model_path: str,
    group: mx.distributed.Group,
    mode: str = "full",
    worker_model_path: Optional[str] = None,
) -> Path:
    """Ensure model files are available on all ranks.

    Transfers model files from rank 0 to workers via the distributed backend.
    Supports full transfer (all files) or sharded transfer (only needed files).

    Args:
        model_path: HuggingFace model ID or local path
        group: MLX distributed group
        mode: Transfer mode - "full", "sharded", or "none"
        worker_model_path: Override path for workers, "hf-cache", or None

    Returns:
        Local path to model directory (usable by all ranks)

    Raises:
        DiskSpaceError: If destination lacks sufficient space
        TransferError: If file transfer fails
        FileNotFoundError: If model not found on rank 0
    """
    rank = group.rank()
    size = group.size()

    if size == 1:
        # Single rank, no sync needed
        return Path(model_path) if os.path.exists(model_path) else get_cache_path(model_path)

    # Determine destination path for workers
    if rank == 0:
        dst_path = Path(model_path)  # Rank 0 uses original path
    else:
        dst_path = resolve_worker_path(model_path, worker_model_path)

    if mode == "none":
        # Assume files are pre-staged at expected path
        if rank != 0 and not dst_path.exists():
            raise FileNotFoundError(
                f"{get_node_prefix(rank)} Model not found at {dst_path} "
                f"and --file-sync=none was specified. "
                f"Use --file-sync=full to sync via distributed backend, or "
                f"--worker-model-path to specify alternate location."
            )
        return dst_path

    logger.info(f"{get_node_prefix(rank)} Starting model sync (mode={mode})")

    # Detect backend and select optimal chunk size
    backend = detect_backend()
    chunk_size = get_chunk_size(backend)
    logger.info(
        f"{get_node_prefix(rank)} Using backend={backend}, "
        f"chunk_size={chunk_size / 1024 / 1024:.0f}MB"
    )

    # Rank 0: resolve model path (may download from HF)
    if rank == 0:
        # Try to resolve the model path
        src_path = Path(model_path)
        if not src_path.exists():
            # Try HuggingFace cache
            from huggingface_hub import snapshot_download
            try:
                src_path = Path(snapshot_download(model_path))
                logger.info(f"[Rank 0] Model resolved to {src_path}")
            except Exception as e:
                raise FileNotFoundError(
                    f"Model not found locally or on HuggingFace: {model_path}"
                ) from e

        files = get_model_files(src_path)
        total_size = sum(size for _, size in files)
        logger.info(
            f"[Rank 0] Found {len(files)} files, "
            f"total {total_size / 1e9:.2f}GB"
        )
    else:
        src_path = None
        files = None

    # Broadcast manifest to all ranks
    manifest = broadcast_manifest(files, group)

    # Separate metadata and weight files
    metadata_files = get_metadata_files(manifest)
    weight_files = get_weight_files(manifest)

    # Create destination directory for workers
    if rank != 0:
        dst_path.mkdir(parents=True, exist_ok=True)

    # Phase 1: Transfer metadata files (small, needed by all ranks)
    logger.info(
        f"{get_node_prefix(rank)} Phase 1: Transferring {len(metadata_files)} metadata files"
    )
    for filename, file_size in metadata_files:
        if rank == 0:
            file_src = src_path / filename
        else:
            file_src = None
        file_dst = dst_path / filename
        transfer_file(file_src, file_dst, file_size, group, chunk_size)

    # Phase 2: Determine which weight files this rank needs
    if mode == "sharded":
        # Try to compute needed files for pipeline parallelism
        index_path = dst_path / "model.safetensors.index.json"
        config_path = dst_path / "config.json"

        needed_file_names = compute_pipeline_files(index_path, config_path, rank, size)

        if needed_file_names is not None:
            # Filter weight files to only those needed by this rank
            weight_files_to_transfer = [
                (name, fsize) for name, fsize in weight_files
                if name in needed_file_names
            ]
            logger.info(
                f"{get_node_prefix(rank)} Sharded mode: need {len(weight_files_to_transfer)} "
                f"of {len(weight_files)} weight files"
            )
        else:
            # Fallback to full transfer if we can't determine needed files
            logger.warning(
                f"{get_node_prefix(rank)} Cannot determine pipeline sharding, "
                f"falling back to full transfer"
            )
            weight_files_to_transfer = weight_files
    else:
        # Full mode: transfer all weight files
        weight_files_to_transfer = weight_files

    # Calculate required disk space for this rank's files
    metadata_size = sum(fsize for _, fsize in metadata_files)
    weight_size = sum(fsize for _, fsize in weight_files_to_transfer)
    required_size = metadata_size + weight_size

    # Check disk space on workers before large transfers
    if rank != 0:
        has_space, free_bytes = check_disk_space(dst_path, required_size)
        if not has_space:
            raise DiskSpaceError(
                f"{get_node_prefix(rank)} Insufficient disk space: "
                f"need {required_size / 1e9:.1f}GB, "
                f"only {free_bytes / 1e9:.1f}GB available at {dst_path.parent}"
            )
        logger.info(
            f"{get_node_prefix(rank)} Disk check OK: "
            f"{free_bytes / 1e9:.1f}GB free, need {required_size / 1e9:.1f}GB"
        )

    # Phase 2: Transfer weight files
    logger.info(
        f"{get_node_prefix(rank)} Phase 2: Transferring {len(weight_files_to_transfer)} "
        f"weight files ({weight_size / 1e9:.2f}GB)"
    )

    files_to_transfer = weight_files_to_transfer

    # Transfer each file
    for filename, file_size in files_to_transfer:
        if rank == 0:
            file_src = src_path / filename
        else:
            file_src = None

        file_dst = dst_path / filename

        # Skip if file already exists with correct size
        if rank != 0 and file_dst.exists():
            if file_dst.stat().st_size == file_size:
                logger.debug(f"{get_node_prefix(rank)} Skipping {filename} (already exists)")
                # Still need to participate in all_sum for rank 0's transfer
                # Actually, we need to skip on all ranks or none
                # For now, always transfer - could optimize later
                pass

        transfer_file(file_src, file_dst, file_size, group, chunk_size)

    total_transferred = metadata_size + weight_size
    logger.info(
        f"{get_node_prefix(rank)} Model sync complete: "
        f"{len(metadata_files) + len(files_to_transfer)} files, "
        f"{total_transferred / 1e9:.2f}GB"
    )

    # Return the appropriate path
    if rank == 0:
        return src_path
    else:
        return dst_path
