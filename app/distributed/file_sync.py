"""Distributed file synchronization for multi-rank model loading.

This module transfers model files from rank 0 to worker ranks using the
mlx.launch distributed backend (JACCL, ring, or MPI).

Key features:
- Two-phase transfer: metadata first, then weight files
- Sharded mode: only transfer files each rank needs (pipeline parallelism)
- Disk space validation before large transfers
- Chunked transfers to stay within MPI limits
- Backend-agnostic: works over JACCL/TB5 RDMA, ring/TCP, or MPI

JACCL Coordination Patterns
===========================
JACCL requires explicit coordination for send/recv due to timing asymmetry.
This module uses both patterns depending on the transfer mode:

1. **all_sum() broadcast pattern** (full disk mode)
   Rank 0 contributes data, workers contribute zeros → result = data.
   All ranks participate synchronously - no additional coordination needed.
   Used by: broadcast_manifest(), transfer_file(), broadcast_file_bytes()

2. **OOB-coordinated send/recv** (memory mode, sharded disk mode)
   Uses PyTorch TCPStore for receiver-initiated rendezvous (see oob.py).
   Receiver signals ready → sender waits → safe to send.
   Used by: oob_send_file_bytes(), oob_recv_file_bytes(),
            make_distributed_weight_loader(), transfer_files_targeted()

Ring and MPI backends don't need OOB coordination - Ring has implicit sync
for neighbor operations, MPI has built-in rendezvous. See oob.py for details.

Transfer modes:
- Memory mode (--file-sync=memory): OOB-coordinated send/recv for targeted
  point-to-point transfers. Best for large models with pipeline parallelism.

- Sharded disk mode (--file-sync=sharded): OOB-coordinated send/recv for
  targeted disk transfers. Each rank only receives files it needs.
  Optimal for pipeline parallelism where different ranks need different files.

- Full disk mode (--file-sync=full): all_sum() broadcast pattern.
  All ranks receive all files. Optimal for tensor parallelism where
  all ranks need all weights.
"""

import hashlib
import json
import os
import shutil
import socket
import struct
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import mlx.core as mx
import numpy as np
from loguru import logger

from .helpers import broadcast_value, synced_all_sum_sync


# Safetensors dtype string -> (numpy dtype for raw bytes, mlx dtype, element size)
# For types numpy doesn't support natively (BF16), we use uint of same size
SAFETENSOR_DTYPE_MAP = {
    "F64": (np.float64, mx.float64, 8),
    "F32": (np.float32, mx.float32, 4),
    "F16": (np.float16, mx.float16, 2),
    "BF16": (np.uint16, mx.bfloat16, 2),  # numpy lacks bfloat16, use uint16
    "I64": (np.int64, mx.int64, 8),
    "I32": (np.int32, mx.int32, 4),
    "I16": (np.int16, mx.int16, 2),
    "I8": (np.int8, mx.int8, 1),
    "U64": (np.uint64, mx.uint64, 8),
    "U32": (np.uint32, mx.uint32, 4),
    "U16": (np.uint16, mx.uint16, 2),
    "U8": (np.uint8, mx.uint8, 1),
    "BOOL": (np.bool_, mx.bool_, 1),
}


def parse_safetensors(data: bytes, log_file=None) -> Dict[str, mx.array]:
    """Parse safetensors format from bytes, returning MLX arrays.

    Zero-copy optimized: uses np.frombuffer with offset to read directly
    from the source buffer without intermediate copies. This handles BF16
    which safetensors.numpy doesn't support.

    Args:
        data: Raw safetensors file bytes
        log_file: Optional file handle for timing logs

    Returns:
        Dictionary mapping tensor names to mx.array
    """
    t_start = time.perf_counter()

    # Parse header size (first 8 bytes, little-endian uint64)
    header_size = struct.unpack("<Q", data[:8])[0]

    # Parse JSON header
    header_json = data[8:8 + header_size].decode("utf-8")
    header = json.loads(header_json)
    t_header = time.perf_counter()

    # Data section starts after header
    data_offset = 8 + header_size

    tensors = {}
    tensor_count = 0
    for name, info in header.items():
        # Skip metadata entry
        if name == "__metadata__":
            continue

        dtype_str = info["dtype"]
        shape = info["shape"]
        start, end = info["data_offsets"]

        if dtype_str not in SAFETENSOR_DTYPE_MAP:
            raise ValueError(f"Unsupported safetensors dtype: {dtype_str}")

        np_dtype, mx_dtype, elem_size = SAFETENSOR_DTYPE_MAP[dtype_str]

        # Calculate element count
        byte_count = end - start
        elem_count = byte_count // elem_size

        # Zero-copy view into original buffer using offset
        np_array = np.frombuffer(
            data, dtype=np_dtype, count=elem_count, offset=data_offset + start
        ).reshape(shape)

        # Single copy to MLX unified memory
        if dtype_str == "BF16":
            mx_array = mx.array(np_array, dtype=mx.uint16)
            tensors[name] = mx_array.view(mx.bfloat16)
        else:
            tensors[name] = mx.array(np_array, dtype=mx_dtype)

        tensor_count += 1

    # Evaluate all tensors immediately to prevent lazy graph accumulation.
    # Without this, MLX builds up computation graphs across all files,
    # causing OOM when finally evaluated.
    mx.eval(tensors)

    t_end = time.perf_counter()

    if log_file:
        log_file.write(
            f"  parse: header={1000*(t_header-t_start):.1f}ms, "
            f"tensors={1000*(t_end-t_header):.1f}ms ({tensor_count} tensors), "
            f"total={1000*(t_end-t_start):.1f}ms\n"
        )
        log_file.flush()

    return tensors

# Chunk size limits by backend
CHUNK_SIZE_MPI = 256 * 1024 * 1024      # 256 MB (MPI int32 limit safety margin)
CHUNK_SIZE_RING = 512 * 1024 * 1024     # 512 MB (balance blocking time vs overhead)
CHUNK_SIZE_JACCL = 1024 * 1024 * 1024   # 1 GB (TB5 RDMA handles large chunks well)

# Maximum filename length for manifest broadcast
MAX_FILENAME_LENGTH = 256

# Maximum files in manifest (should be plenty for any model)
MAX_MANIFEST_FILES = 100

# Disk space buffer percentage (large models use large disks)
DISK_SPACE_BUFFER_PERCENT = 2


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


class MemoryError(FileSyncError):
    """Raised when system has insufficient memory for streaming."""
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
    buffer_multiplier = 1 + DISK_SPACE_BUFFER_PERCENT * 0.01
    has_space = usage.free >= required_bytes * buffer_multiplier
    return has_space, usage.free


def get_available_memory() -> int:
    """Get available system memory in bytes.

    On macOS, calculates total RAM minus wired pages. Wired pages are truly
    locked in memory; everything else (inactive, purgeable, file cache, etc.)
    can be reclaimed under memory pressure. Using free+inactive significantly
    underestimates available memory.

    Returns:
        Available memory in bytes
    """
    import subprocess
    import platform

    if platform.system() == "Darwin":
        # macOS: total RAM minus wired pages gives usable memory
        try:
            # Get total physical memory via sysctl
            total_result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            total_ram = int(total_result.stdout.strip())

            # Get wired pages from vm_stat (truly locked, can't be reclaimed)
            vm_result = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
                check=True,
            )
            page_size = 16384  # Default for Apple Silicon
            wired_pages = 0
            for line in vm_result.stdout.split("\n"):
                if "page size of" in line:
                    page_size = int(line.split()[-2])
                elif "Pages wired down:" in line:
                    wired_pages = int(line.split()[-1].rstrip("."))

            # Available = total - wired (everything else can be reclaimed)
            wired_bytes = wired_pages * page_size
            return total_ram - wired_bytes
        except Exception:
            pass

    # Fallback: try psutil if available
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass

    # Last resort: return 0 (will skip check)
    return 0


def check_memory_for_streaming(
    model_bytes: int,
    max_file_bytes: int,
    rank: int,
) -> tuple[bool, int, int]:
    """Check if system has enough memory for streaming weight loading.

    For memory-based loading, we need:
    - Space for the model weights
    - Plus one file buffer (held during transfer, freed after parsing)

    Args:
        model_bytes: Total bytes of model weights to load
        max_file_bytes: Size of the largest single file (buffer requirement)
        rank: Current rank (for logging)

    Returns:
        Tuple of (has_enough_memory, available_bytes, required_bytes)
    """
    available = get_available_memory()

    if available == 0:
        # Couldn't determine memory, skip check
        logger.warning(
            f"{get_node_prefix(rank)} Could not determine available memory, skipping check"
        )
        return True, 0, 0

    # Required: model weights + one file buffer + small margin
    required = model_bytes + max_file_bytes
    buffer_multiplier = 1 + DISK_SPACE_BUFFER_PERCENT * 0.01
    required_with_buffer = int(required * buffer_multiplier)

    has_memory = available >= required_with_buffer

    return has_memory, available, required_with_buffer


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

    count_result = synced_all_sum_sync(count, group, "manifest_count")
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

        header_result = synced_all_sum_sync(header, group, f"manifest_header_{i}")
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

        name_result = synced_all_sum_sync(name_array, group, f"manifest_name_{i}")

        # Decode filename
        name_bytes = np.array(name_result[:name_len], copy=False).tobytes()
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

    # Get OOB for termination checking
    from .oob import get_oob
    oob = get_oob()

    try:
        bytes_transferred = 0

        for chunk_idx in range(num_chunks):
            # Periodic termination check (every 10 chunks) to allow graceful abort
            if chunk_idx > 0 and chunk_idx % 10 == 0:
                if oob is not None and oob.is_any_peer_terminating():
                    raise TransferError(
                        f"Peer terminated during transfer of {dst_path.name} "
                        f"(chunk {chunk_idx}/{num_chunks})"
                    )

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
                chunk_data = np.array(result[:this_chunk_size], copy=False).tobytes()
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

    # Compute layer assignment matching PipelineMixin's REVERSE order:
    # rank=0 gets the LAST layers, rank=(world_size-1) gets FIRST layers.
    # See mlx_lm/models/pipeline.py:
    #   "Split layers in reverse so rank=0 gets the last layers"
    layers_per_rank = num_layers // world_size
    extra = num_layers - layers_per_rank * world_size

    # Match PipelineMixin's exact formula for handling uneven splits
    if rank < extra:
        layers_per_rank_this = layers_per_rank + 1
    else:
        layers_per_rank_this = layers_per_rank

    # PipelineMixin: start_idx = (pipeline_size - pipeline_rank - 1) * layers_per_rank
    # For weight loading, we need the actual layer indices this rank handles
    start_layer = (world_size - rank - 1) * layers_per_rank
    if rank < extra:
        # Adjust for extra layers distributed to lower ranks
        start_layer += min(extra, world_size - rank - 1)
    end_layer = start_layer + layers_per_rank_this

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
            # Non-layer parameters - assign based on pipeline position:
            # - Embeddings process input → needed by rank with FIRST layers
            #   In PipelineMixin's reverse order, that's rank=(world_size-1)
            # - lm_head produces output → needed by rank with LAST layers
            #   In PipelineMixin's reverse order, that's rank=0
            param_lower = param_name.lower()
            if "embed" in param_lower:
                # Embeddings → first layers → rank (world_size - 1)
                if rank == world_size - 1:
                    needed_files.add(file_name)
            else:
                # lm_head, model.norm, etc. → last layers → rank 0
                if rank == 0:
                    needed_files.add(file_name)

    return needed_files


def compute_all_rank_files(
    index_path: Path,
    config_path: Path,
    world_size: int,
) -> dict[int, set[str]]:
    """Compute which weight files each rank needs for pipeline parallelism.

    Called by rank 0 to determine the full transfer schedule.

    Args:
        index_path: Path to model.safetensors.index.json
        config_path: Path to config.json
        world_size: Number of ranks

    Returns:
        Dictionary mapping rank -> set of filenames needed by that rank.
        Returns None if sharding cannot be determined.
    """
    rank_files = {}
    for rank in range(world_size):
        files = compute_pipeline_files(index_path, config_path, rank, world_size)
        if files is None:
            return None
        rank_files[rank] = files
    return rank_files


def broadcast_rank_assignments(
    rank_files: Optional[dict[int, set[str]]],
    group: mx.distributed.Group,
) -> dict[str, set[int]]:
    """Broadcast file-to-rank assignments from rank 0 to all workers.

    Converts rank->files mapping to file->ranks mapping and broadcasts via all_sum.

    Args:
        rank_files: Dictionary from compute_all_rank_files() (rank 0 only)
        group: MLX distributed group

    Returns:
        Dictionary mapping filename -> set of ranks that need it
    """
    rank = group.rank()
    world_size = group.size()

    # Rank 0 serializes and broadcasts the assignments
    if rank == 0:
        # Convert rank->files to file->ranks for easier lookup during transfer
        file_to_ranks: dict[str, set[int]] = {}
        if rank_files:
            for r, files in rank_files.items():
                for f in files:
                    if f not in file_to_ranks:
                        file_to_ranks[f] = set()
                    file_to_ranks[f].add(r)

        # Serialize to JSON (convert sets to lists for JSON)
        assignments_json = json.dumps({
            f: list(ranks) for f, ranks in file_to_ranks.items()
        })
        assignments_bytes = assignments_json.encode("utf-8")
        size_array = mx.array([len(assignments_bytes)], dtype=mx.int64)
    else:
        size_array = mx.zeros((1,), dtype=mx.int64)

    # Broadcast size
    size_result = synced_all_sum_sync(size_array, group, "assignments_size")
    data_size = int(size_result[0].item())

    if data_size == 0:
        return {}

    # Broadcast data (pad to fixed size for all_sum)
    if rank == 0:
        padded = np.zeros(data_size, dtype=np.uint8)
        padded[:len(assignments_bytes)] = list(assignments_bytes)
        data_array = mx.array(padded)
    else:
        data_array = mx.zeros((data_size,), dtype=mx.uint8)

    data_result = synced_all_sum_sync(data_array, group, "assignments_data")

    # Deserialize
    data_bytes = np.array(data_result, copy=False).tobytes()
    assignments_json = data_bytes.decode("utf-8")
    raw_assignments = json.loads(assignments_json)

    # Convert lists back to sets
    return {f: set(ranks) for f, ranks in raw_assignments.items()}


def transfer_files_targeted(
    src_path: Optional[Path],
    dst_path: Path,
    weight_files: list[tuple[str, int]],
    file_assignments: dict[str, set[int]],
    group: mx.distributed.Group,
    chunk_size: int,
) -> int:
    """Transfer weight files to specific ranks using OOB-coordinated send/recv.

    Unlike transfer_file() which broadcasts to all ranks via all_sum, this
    function only sends each file to the ranks that need it. This is optimal
    for pipeline parallelism where different ranks need different files.

    Requires OOB coordinator to be initialized (see oob.py).

    Args:
        src_path: Source directory (rank 0 only)
        dst_path: Destination directory
        weight_files: List of (filename, size) tuples to transfer
        file_assignments: Dict mapping filename -> set of ranks that need it
        group: MLX distributed group
        chunk_size: Size of each transfer chunk

    Returns:
        Total bytes transferred to this rank
    """
    from .oob import get_oob

    oob = get_oob()
    if oob is None:
        raise RuntimeError(
            "OOB coordinator not initialized. Sharded disk mode requires OOB. "
            "See oob.py for details."
        )

    rank = group.rank()
    bytes_received = 0

    # Process files in deterministic order (all ranks use same order)
    for filename, file_size in sorted(weight_files):
        needed_by = file_assignments.get(filename, set())

        if not needed_by:
            # No rank needs this file, skip
            continue

        transfer_base_id = f"sharded_disk_{filename}"

        if rank == 0:
            # Rank 0 reads the file
            file_bytes = (src_path / filename).read_bytes()

            # Write locally if rank 0 needs it
            if 0 in needed_by:
                file_dst = dst_path / filename
                file_dst.parent.mkdir(parents=True, exist_ok=True)
                file_dst.write_bytes(file_bytes)
                bytes_received += len(file_bytes)
                logger.debug(f"{get_node_prefix(rank)} Wrote {filename} locally")

            # Send to workers that need it (in sorted order for determinism)
            for dst_rank in sorted(needed_by - {0}):
                transfer_id = f"{transfer_base_id}_to_{dst_rank}"
                logger.debug(
                    f"{get_node_prefix(rank)} Sending {filename} to rank {dst_rank}"
                )
                oob_send_file_bytes(
                    file_bytes, group, dst_rank, transfer_id, chunk_size
                )
        else:
            # Workers receive if they need this file
            if rank in needed_by:
                transfer_id = f"{transfer_base_id}_to_{rank}"
                logger.debug(
                    f"{get_node_prefix(rank)} Receiving {filename} from rank 0"
                )
                file_bytes = oob_recv_file_bytes(
                    group, src_rank=0, transfer_id=transfer_id, chunk_size=chunk_size
                )

                # Write to disk
                file_dst = dst_path / filename
                file_dst.parent.mkdir(parents=True, exist_ok=True)
                file_dst.write_bytes(file_bytes)
                bytes_received += len(file_bytes)
                logger.debug(
                    f"{get_node_prefix(rank)} Wrote {filename} ({len(file_bytes)} bytes)"
                )

    return bytes_received


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

    # Phase 2: Transfer weight files
    # Different strategies for sharded vs full mode:
    # - sharded: OOB-coordinated send/recv (only transfers files to ranks that need them)
    # - full: all_sum broadcast (all ranks receive all files)
    metadata_size = sum(fsize for _, fsize in metadata_files)

    if mode == "sharded":
        # Sharded mode uses OOB-coordinated send/recv for targeted transfers.
        # This is optimal for pipeline parallelism where different ranks need
        # different files. See transfer_files_targeted() for details.
        index_path = dst_path / "model.safetensors.index.json"
        config_path = dst_path / "config.json"

        # Rank 0 computes file assignments for ALL ranks
        if rank == 0:
            rank_files = compute_all_rank_files(index_path, config_path, size)
            if rank_files is None:
                logger.warning(
                    f"{get_node_prefix(rank)} Cannot determine pipeline sharding, "
                    f"falling back to full transfer"
                )
        else:
            rank_files = None

        # Broadcast assignments to all ranks (small data, uses all_sum)
        file_assignments = broadcast_rank_assignments(rank_files, group)

        if not file_assignments:
            # Fallback to full mode if sharding couldn't be determined
            logger.warning(
                f"{get_node_prefix(rank)} Sharded mode fallback: using full transfer"
            )
            mode = "full"  # Fall through to full mode below
        else:
            # Calculate which files this rank needs (for disk space check)
            my_files = [
                (name, fsize) for name, fsize in weight_files
                if rank in file_assignments.get(name, set())
            ]
            weight_size = sum(fsize for _, fsize in my_files)
            required_size = metadata_size + weight_size

            # Check disk space before transfer
            if rank != 0:
                has_space, free_bytes = check_disk_space(dst_path, required_size)
                if not has_space:
                    buffer_multiplier = 1 + DISK_SPACE_BUFFER_PERCENT * 0.01
                    buffered_size = required_size * buffer_multiplier
                    raise DiskSpaceError(
                        f"{get_node_prefix(rank)} Insufficient disk space: "
                        f"need {buffered_size / 1e9:.1f}GB (inc. {DISK_SPACE_BUFFER_PERCENT}% buffer), "
                        f"only {free_bytes / 1e9:.1f}GB available at {dst_path.parent}"
                    )
                logger.info(
                    f"{get_node_prefix(rank)} Disk check OK: "
                    f"{free_bytes / 1e9:.1f}GB free, need {required_size / 1e9:.1f}GB"
                )

            logger.info(
                f"{get_node_prefix(rank)} Phase 2: Sharded transfer - "
                f"this rank needs {len(my_files)} of {len(weight_files)} weight files "
                f"({weight_size / 1e9:.2f}GB)"
            )

            # Transfer using OOB-coordinated send/recv
            bytes_transferred = transfer_files_targeted(
                src_path, dst_path, weight_files, file_assignments, group, chunk_size
            )

            total_transferred = metadata_size + bytes_transferred
            logger.info(
                f"{get_node_prefix(rank)} Model sync complete: "
                f"{len(metadata_files)} metadata + {len(my_files)} weight files, "
                f"{total_transferred / 1e9:.2f}GB"
            )

    # Full mode: transfer all weight files to all ranks via all_sum broadcast
    if mode == "full":
        weight_files_to_transfer = weight_files
        weight_size = sum(fsize for _, fsize in weight_files_to_transfer)
        required_size = metadata_size + weight_size

        # Check disk space on workers before large transfers
        if rank != 0:
            has_space, free_bytes = check_disk_space(dst_path, required_size)
            if not has_space:
                buffer_multiplier = 1 + DISK_SPACE_BUFFER_PERCENT * 0.01
                buffered_size = required_size * buffer_multiplier
                raise DiskSpaceError(
                    f"{get_node_prefix(rank)} Insufficient disk space: "
                    f"need {buffered_size / 1e9:.1f}GB (inc. {DISK_SPACE_BUFFER_PERCENT}% buffer), "
                    f"only {free_bytes / 1e9:.1f}GB available at {dst_path.parent}"
                )
            logger.info(
                f"{get_node_prefix(rank)} Disk check OK: "
                f"{free_bytes / 1e9:.1f}GB free, need {required_size / 1e9:.1f}GB"
            )

        logger.info(
            f"{get_node_prefix(rank)} Phase 2: Full transfer - "
            f"{len(weight_files_to_transfer)} weight files ({weight_size / 1e9:.2f}GB)"
        )

        # Transfer each file via all_sum broadcast
        for filename, file_size in weight_files_to_transfer:
            if rank == 0:
                file_src = src_path / filename
            else:
                file_src = None
            file_dst = dst_path / filename
            transfer_file(file_src, file_dst, file_size, group, chunk_size)

        total_transferred = metadata_size + weight_size
        logger.info(
            f"{get_node_prefix(rank)} Model sync complete: "
            f"{len(metadata_files) + len(weight_files_to_transfer)} files, "
            f"{total_transferred / 1e9:.2f}GB"
        )

    # Return the appropriate path
    if rank == 0:
        return src_path
    else:
        return dst_path


def sync_metadata_to_workers(
    model_path: str,
    group: mx.distributed.Group,
    worker_model_path: Optional[str] = None,
) -> Path:
    """Sync only metadata files (configs, tokenizer) for memory-based weight loading.

    This is used in memory streaming mode where weight files are broadcast via
    all_sum but config files need to be on disk for mlx-lm's model loading.

    Args:
        model_path: HuggingFace model ID or local path
        group: MLX distributed group
        worker_model_path: Override path for workers, "hf-cache", or None

    Returns:
        Local path to model directory (usable by all ranks)

    Raises:
        FileNotFoundError: If model not found on rank 0
    """
    rank = group.rank()
    size = group.size()

    if size == 1:
        # Single rank, no sync needed
        return Path(model_path) if os.path.exists(model_path) else get_cache_path(model_path)

    # Determine destination path for workers
    if rank == 0:
        dst_path = Path(model_path)
    else:
        dst_path = resolve_worker_path(model_path, worker_model_path)

    logger.info(f"{get_node_prefix(rank)} Syncing metadata files for memory mode")

    # Detect backend and select chunk size
    backend = detect_backend()
    chunk_size = get_chunk_size(backend)

    # Rank 0: resolve model path
    if rank == 0:
        src_path = Path(model_path)
        if not src_path.exists():
            from huggingface_hub import snapshot_download
            try:
                src_path = Path(snapshot_download(model_path))
                logger.info(f"[Rank 0] Model resolved to {src_path}")
            except Exception as e:
                raise FileNotFoundError(
                    f"Model not found locally or on HuggingFace: {model_path}"
                ) from e

        files = get_model_files(src_path)
    else:
        src_path = None
        files = None

    # Broadcast manifest to all ranks
    manifest = broadcast_manifest(files, group)

    # Only sync metadata files (small configs, tokenizer)
    metadata_files = get_metadata_files(manifest)

    # Create destination directory for workers
    if rank != 0:
        dst_path.mkdir(parents=True, exist_ok=True)

    # Transfer metadata files
    logger.info(
        f"{get_node_prefix(rank)} Transferring {len(metadata_files)} metadata files"
    )
    for filename, file_size in metadata_files:
        if rank == 0:
            file_src = src_path / filename
        else:
            file_src = None
        file_dst = dst_path / filename
        transfer_file(file_src, file_dst, file_size, group, chunk_size)

    metadata_size = sum(fsize for _, fsize in metadata_files)
    logger.info(
        f"{get_node_prefix(rank)} Metadata sync complete: "
        f"{len(metadata_files)} files, {metadata_size / 1e6:.1f}MB"
    )

    # Return the appropriate path
    if rank == 0:
        return src_path
    else:
        return dst_path


def broadcast_file_bytes(
    file_path: str,
    group: mx.distributed.Group,
    chunk_size: Optional[int] = None,
    log_file=None,
) -> bytearray:
    """Broadcast file bytes from rank 0 to all ranks via all_sum.

    Optimized to minimize intermediate copies:
    - Uses numpy views with offset instead of byte slicing
    - Pre-allocates output buffer (single allocation)
    - Reuses padded chunk buffer on rank 0
    - Returns bytearray directly (no final copy)

    Args:
        file_path: Path to file (only read on rank 0)
        group: MLX distributed group
        chunk_size: Override chunk size (auto-detected if None)
        log_file: Optional file handle for timing logs

    Returns:
        File bytes as bytearray on all ranks
    """
    rank = group.rank()
    t_start = time.perf_counter()

    if chunk_size is None:
        chunk_size = get_chunk_size()

    # Rank 0 reads file and broadcasts size
    if rank == 0:
        file_bytes = Path(file_path).read_bytes()
        file_size = len(file_bytes)
        size_array = mx.array([file_size], dtype=mx.int64)
    else:
        file_bytes = None
        size_array = mx.zeros((1,), dtype=mx.int64)

    t_read = time.perf_counter()

    # Broadcast file size
    size_result = synced_all_sum_sync(size_array, group, "file_size")
    file_size = int(size_result[0].item())
    del size_array, size_result

    if file_size == 0:
        return bytearray()

    # Calculate chunks
    num_chunks = (file_size + chunk_size - 1) // chunk_size

    # OPTIMIZATION: Rank 0 already has the file data. Since all_sum result = data + zeros = data,
    # Rank 0 doesn't need an output buffer - it can return its original file_bytes.
    # This halves Rank 0's memory usage per file.
    if rank != 0:
        # Only non-zero ranks need an output buffer
        output = bytearray(file_size)

    bytes_transferred = 0

    # Pre-allocate reusable padded buffer for rank 0 (avoids repeated allocation)
    if rank == 0:
        padded_buffer = np.zeros(chunk_size, dtype=np.uint8)

    t_alloc = time.perf_counter()
    chunk_times = []

    # Get OOB for termination checking
    from .oob import get_oob
    oob = get_oob()

    for chunk_idx in range(num_chunks):
        # Periodic termination check (every 10 chunks) to allow graceful abort
        if chunk_idx > 0 and chunk_idx % 10 == 0:
            if oob is not None and oob.is_any_peer_terminating():
                raise TransferError(
                    f"Peer terminated during broadcast of {file_path} "
                    f"(chunk {chunk_idx}/{num_chunks})"
                )

        t_chunk_start = time.perf_counter()
        remaining = file_size - bytes_transferred
        this_chunk_size = min(chunk_size, remaining)

        if rank == 0:
            # Zero-copy view into source bytes, copy into pre-allocated padded buffer
            src_view = np.frombuffer(
                file_bytes, dtype=np.uint8, count=this_chunk_size, offset=bytes_transferred
            )
            padded_buffer[:this_chunk_size] = src_view
            # Zero padding for remainder (only needed if chunk is partial)
            if this_chunk_size < chunk_size:
                padded_buffer[this_chunk_size:] = 0
            chunk = mx.array(padded_buffer)
        else:
            chunk = mx.zeros((chunk_size,), dtype=mx.uint8)

        t_prep = time.perf_counter()

        result = mx.distributed.all_sum(chunk, group=group)
        mx.eval(result)

        t_allsum = time.perf_counter()

        # Only non-zero ranks need to copy the result - rank 0 already has the data
        if rank != 0:
            np.copyto(
                np.frombuffer(output, dtype=np.uint8, count=this_chunk_size, offset=bytes_transferred),
                np.array(result[:this_chunk_size], copy=False)
            )
        bytes_transferred += this_chunk_size

        # Free MLX arrays immediately
        del chunk, result

        t_chunk_end = time.perf_counter()
        chunk_times.append({
            'prep': t_prep - t_chunk_start,
            'allsum': t_allsum - t_prep,
            'copy': t_chunk_end - t_allsum,
        })

    mx.clear_cache()

    t_end = time.perf_counter()

    if log_file:
        file_size_mb = file_size / 1e6
        total_ms = 1000 * (t_end - t_start)
        read_ms = 1000 * (t_read - t_start)
        alloc_ms = 1000 * (t_alloc - t_read)
        transfer_ms = 1000 * (t_end - t_alloc)
        throughput = file_size_mb / (t_end - t_start) if (t_end - t_start) > 0 else 0

        # Aggregate chunk timing
        total_prep = sum(c['prep'] for c in chunk_times) * 1000
        total_allsum = sum(c['allsum'] for c in chunk_times) * 1000
        total_copy = sum(c['copy'] for c in chunk_times) * 1000

        log_file.write(
            f"  broadcast: {file_size_mb:.1f}MB in {total_ms:.0f}ms ({throughput:.0f}MB/s)\n"
            f"    read={read_ms:.0f}ms, alloc={alloc_ms:.0f}ms, transfer={transfer_ms:.0f}ms\n"
            f"    chunks({num_chunks}): prep={total_prep:.0f}ms, allsum={total_allsum:.0f}ms, copy={total_copy:.0f}ms\n"
        )
        log_file.flush()

    # Return appropriate buffer - rank 0 uses original file_bytes (no copy), others use output
    # parse_safetensors accepts both bytes and bytearray via np.frombuffer
    if rank == 0:
        return file_bytes  # Return original bytes directly, no copy
    else:
        return output


def _get_send_function():
    """Get the appropriate send function, bypassing JACCL patch if active.

    File transfers have their own OOB coordination (signal_ready/wait_complete),
    so they must bypass the JACCL patch's barrier coordination to avoid deadlock.
    The patch is designed for model inference (lazy tensor loading), not raw byte transfers.
    """
    from . import jaccl_patch
    if jaccl_patch.is_patched() and jaccl_patch._original_send is not None:
        return jaccl_patch._original_send
    return mx.distributed.send


def send_file_bytes(
    data: bytes,
    group: mx.distributed.Group,
    dst_rank: int,
    chunk_size: Optional[int] = None,
    log_file=None,
) -> None:
    """Send bytes to a specific destination rank via send().

    Only called on rank 0. Uses chunked transfer for large data.

    NOTE: Uses the original (unpatched) send when JACCL patch is active.
    File transfers have their own OOB coordination and don't need the patch's
    barrier. Using the patched version would cause double coordination deadlock.

    Args:
        data: Bytes to send
        group: MLX distributed group
        dst_rank: Destination rank (must be != 0)
        chunk_size: Override chunk size (auto-detected if None)
        log_file: Optional file handle for timing logs
    """
    t_start = time.perf_counter()

    if chunk_size is None:
        chunk_size = get_chunk_size()

    # Get the appropriate send function (original if patch is active)
    send_fn = _get_send_function()

    file_bytes = data
    file_size = len(file_bytes)

    # Send file size first.
    # IMPORTANT: Eval the RESULT of send(), not the input array.
    # send() returns a dependency-tracked array; eval triggers the actual send.
    size_array = mx.array([file_size], dtype=mx.int64)
    sent_size = send_fn(size_array, dst_rank, group=group)
    mx.eval(sent_size)
    del size_array, sent_size

    if file_size == 0:
        return

    # Calculate chunks
    num_chunks = (file_size + chunk_size - 1) // chunk_size

    # Pre-allocate reusable padded buffer
    padded_buffer = np.zeros(chunk_size, dtype=np.uint8)
    bytes_sent = 0

    t_alloc = time.perf_counter()

    for chunk_idx in range(num_chunks):
        remaining = file_size - bytes_sent
        this_chunk_size = min(chunk_size, remaining)

        # Zero-copy view into source bytes
        src_view = np.frombuffer(
            file_bytes, dtype=np.uint8, count=this_chunk_size, offset=bytes_sent
        )
        padded_buffer[:this_chunk_size] = src_view
        if this_chunk_size < chunk_size:
            padded_buffer[this_chunk_size:] = 0

        chunk = mx.array(padded_buffer)
        # Eval the send result to trigger the actual send
        sent = send_fn(chunk, dst_rank, group=group)
        mx.eval(sent)

        bytes_sent += this_chunk_size
        del chunk, sent

    t_end = time.perf_counter()

    if log_file:
        file_size_mb = file_size / 1e6
        total_ms = 1000 * (t_end - t_start)
        send_ms = 1000 * (t_end - t_alloc)
        throughput = file_size_mb / (t_end - t_start) if (t_end - t_start) > 0 else 0

        log_file.write(
            f"  send: {file_size_mb:.1f}MB in {total_ms:.0f}ms ({throughput:.0f}MB/s)\n"
            f"    transfer={send_ms:.0f}ms ({num_chunks} chunks)\n"
        )
        log_file.flush()

    mx.clear_cache()


def _get_recv_like_function():
    """Get the appropriate recv_like function, bypassing JACCL patch if active.

    File transfers have their own OOB coordination (signal_ready/wait_complete),
    so they must bypass the JACCL patch's barrier coordination to avoid deadlock.
    The patch is designed for model inference (lazy tensor loading), not raw byte transfers.
    """
    from . import jaccl_patch
    if jaccl_patch.is_patched() and jaccl_patch._original_recv_like is not None:
        return jaccl_patch._original_recv_like
    return mx.distributed.recv_like


def recv_file_bytes(
    group: mx.distributed.Group,
    src_rank: int,
    chunk_size: Optional[int] = None,
    log_file=None,
) -> bytearray:
    """Receive file bytes from a source rank via recv_like().

    Only called on non-zero ranks. Uses chunked transfer for large files.

    NOTE: Uses the original (unpatched) recv_like when JACCL patch is active.
    File transfers have their own OOB coordination and don't need the patch's
    barrier. Using the patched version would cause double coordination deadlock.

    Args:
        group: MLX distributed group
        src_rank: Source rank (must be 0)
        chunk_size: Override chunk size (auto-detected if None)
        log_file: Optional file handle for timing logs

    Returns:
        File bytes as bytearray
    """
    t_start = time.perf_counter()

    if chunk_size is None:
        chunk_size = get_chunk_size()

    # Get the appropriate recv_like function (original if patch is active)
    recv_like_fn = _get_recv_like_function()

    # Receive file size first
    size_template = mx.zeros((1,), dtype=mx.int64)
    size_array = recv_like_fn(size_template, src_rank, group=group)
    mx.eval(size_array)
    file_size = int(size_array[0].item())
    del size_template, size_array

    if file_size == 0:
        return bytearray()

    # Calculate chunks
    num_chunks = (file_size + chunk_size - 1) // chunk_size

    # Pre-allocate output buffer
    output = bytearray(file_size)
    bytes_received = 0

    t_alloc = time.perf_counter()

    # Template for receiving chunks
    chunk_template = mx.zeros((chunk_size,), dtype=mx.uint8)

    for chunk_idx in range(num_chunks):
        remaining = file_size - bytes_received
        this_chunk_size = min(chunk_size, remaining)

        result = recv_like_fn(chunk_template, src_rank, group=group)
        mx.eval(result)

        # Copy into pre-allocated output buffer
        np.copyto(
            np.frombuffer(output, dtype=np.uint8, count=this_chunk_size, offset=bytes_received),
            np.array(result[:this_chunk_size], copy=False)
        )
        bytes_received += this_chunk_size
        del result

    t_end = time.perf_counter()

    if log_file:
        file_size_mb = file_size / 1e6
        total_ms = 1000 * (t_end - t_start)
        recv_ms = 1000 * (t_end - t_alloc)
        throughput = file_size_mb / (t_end - t_start) if (t_end - t_start) > 0 else 0

        log_file.write(
            f"  recv: {file_size_mb:.1f}MB in {total_ms:.0f}ms ({throughput:.0f}MB/s)\n"
            f"    recv={recv_ms:.0f}ms ({num_chunks} chunks)\n"
        )
        log_file.flush()

    del chunk_template
    mx.clear_cache()

    return output


async def oob_send_file_bytes_async(
    data: bytes,
    group: mx.distributed.Group,
    dst_rank: int,
    transfer_id: str,
    chunk_size: Optional[int] = None,
    log_file=None,
) -> None:
    """Async version: Send bytes with OOB receiver-initiated rendezvous.

    Use this from async contexts (e.g., main.py startup).
    """
    from .oob import get_oob

    oob = get_oob()
    if oob is None:
        raise RuntimeError(
            "OOB coordinator not initialized. "
            "Call init_oob() or set MLX_OOB_HOST before using oob_send_file_bytes."
        )

    if oob.is_any_peer_terminating():
        raise RuntimeError("Peer terminated before send could complete")

    # Wait for receiver to signal ready
    logger.info(f"[Rank {oob.rank}] oob_send: waiting for receiver ready signal")
    await oob.wait_ready(transfer_id, dst_rank)
    logger.info(f"[Rank {oob.rank}] oob_send: receiver ready, sending data")

    # Now safe to send - receiver has posted recv
    send_file_bytes(data, group, dst_rank, chunk_size, log_file)
    logger.info(f"[Rank {oob.rank}] oob_send: send complete, signaling completion")

    # Signal completion
    await oob.signal_complete(transfer_id)


def oob_send_file_bytes(
    data: bytes,
    group: mx.distributed.Group,
    dst_rank: int,
    transfer_id: str,
    chunk_size: Optional[int] = None,
    log_file=None,
) -> None:
    """Send bytes with OOB receiver-initiated rendezvous.

    Implements the safe JACCL send pattern (see oob.py):
    1. Wait for receiver to signal ready (via TCPStore)
    2. Send data chunks (receiver has already posted recv)
    3. Signal completion

    Args:
        data: Bytes to send
        group: MLX distributed group
        dst_rank: Destination rank
        transfer_id: Unique identifier for this transfer (for OOB coordination)
        chunk_size: Override chunk size (auto-detected if None)
        log_file: Optional file handle for timing logs
    """
    from .oob import get_oob, oob_wait_ready_sync, oob_signal_complete_sync

    oob = get_oob()
    if oob is None:
        raise RuntimeError(
            "OOB coordinator not initialized. "
            "Call init_oob() or set MLX_OOB_HOST before using oob_send_file_bytes."
        )

    # Check for peer termination before starting transfer
    if oob.is_any_peer_terminating():
        raise RuntimeError("Peer terminated before send could complete")

    # Wait for receiver to signal ready
    oob_wait_ready_sync(transfer_id, dst_rank)

    # Now safe to send - receiver has posted recv
    send_file_bytes(data, group, dst_rank, chunk_size, log_file)

    # Signal completion
    oob_signal_complete_sync(transfer_id)


async def oob_recv_file_bytes_async(
    group: mx.distributed.Group,
    src_rank: int,
    transfer_id: str,
    chunk_size: Optional[int] = None,
    log_file=None,
) -> bytearray:
    """Async version: Receive bytes with OOB receiver-initiated rendezvous.

    Use this from async contexts (e.g., main.py startup).
    """
    from .oob import get_oob

    oob = get_oob()
    if oob is None:
        raise RuntimeError(
            "OOB coordinator not initialized. "
            "Call init_oob() or set MLX_OOB_HOST before using oob_recv_file_bytes."
        )

    if oob.is_any_peer_terminating():
        raise RuntimeError("Peer terminated before recv could complete")

    # Signal we're ready to receive
    logger.info(f"[Rank {oob.rank}] oob_recv: signaling ready")
    await oob.signal_ready(transfer_id)
    logger.info(f"[Rank {oob.rank}] oob_recv: ready signaled, receiving data")

    # Receive the data
    result = recv_file_bytes(group, src_rank, chunk_size, log_file)
    logger.info(f"[Rank {oob.rank}] oob_recv: receive complete, waiting for sender completion")

    # Wait for sender to confirm completion
    await oob.wait_complete(transfer_id, src_rank)

    return result


def oob_recv_file_bytes(
    group: mx.distributed.Group,
    src_rank: int,
    transfer_id: str,
    chunk_size: Optional[int] = None,
    log_file=None,
) -> bytearray:
    """Receive bytes with OOB receiver-initiated rendezvous.

    Implements the safe JACCL recv pattern (see oob.py):
    1. Signal ready (via TCPStore) - tells sender we've posted recv
    2. Receive data chunks
    3. Wait for sender's completion signal

    Args:
        group: MLX distributed group
        src_rank: Source rank
        transfer_id: Unique identifier for this transfer (for OOB coordination)
        chunk_size: Override chunk size (auto-detected if None)
        log_file: Optional file handle for timing logs

    Returns:
        File bytes as bytearray
    """
    from .oob import get_oob, oob_signal_ready_sync, oob_wait_complete_sync

    oob = get_oob()
    if oob is None:
        raise RuntimeError(
            "OOB coordinator not initialized. "
            "Call init_oob() or set MLX_OOB_HOST before using oob_recv_file_bytes."
        )

    # Check for peer termination before starting transfer
    if oob.is_any_peer_terminating():
        raise RuntimeError("Peer terminated before recv could complete")

    # Signal we're ready to receive
    oob_signal_ready_sync(transfer_id)

    # Receive the data
    result = recv_file_bytes(group, src_rank, chunk_size, log_file)

    # Wait for sender to confirm completion
    oob_wait_complete_sync(transfer_id, src_rank)

    return result


async def make_distributed_weight_loader(
    group: mx.distributed.Group,
    model_path: str = None,
    distributed_mode: str = None,
) -> Callable[[str], Dict[str, Any]]:
    """Create a weight loader using OOB-coordinated send/recv for JACCL transfers.

    This loader implements point-to-point weight file transfers using the
    OOB (out-of-band) coordination layer. See oob.py for the full API.

    The OOB pattern (receiver-initiated rendezvous) solves JACCL's timing
    asymmetry issues that cause SIGBUS with raw send/recv:
    1. Receiver signals ready via TCPStore
    2. Sender waits for ready signal
    3. Safe to send - receiver has posted recv

    For pipeline parallelism, only sends files to ranks that need them.
    For tensor parallelism, broadcasts all files to all ranks.

    Requires OOB coordinator to be initialized via init_oob() before calling.

    Note: This is an async function because it performs OOB coordination during
    setup. The returned weight loader closure is synchronous for compatibility
    with mlx-lm's load_model.

    Args:
        group: MLX distributed group
        model_path: Path to model directory (for computing needed files)
        distributed_mode: "pipeline" or "tensor" parallelism mode

    Returns:
        A callable that accepts a file path and returns a weight dictionary

    Example:
        >>> group = mx.distributed.init()
        >>> await init_oob(rank, world_size, "coordinator_host", 29400)
        >>> loader = await make_distributed_weight_loader(group, model_path, "pipeline")
        >>> model, tokenizer = load(model_path, weight_loader=loader)
    """
    from .oob import get_oob, oob_barrier_sync

    rank = group.rank()
    world_size = group.size()
    backend = detect_backend()
    chunk_size = get_chunk_size(backend)

    # Get OOB coordinator for synchronization
    oob = get_oob()
    if oob is None:
        raise RuntimeError(
            "OOB coordinator not initialized. Call init_oob() before "
            "make_distributed_weight_loader(). See oob.py for details."
        )

    # For pipeline parallelism, compute which files each rank needs
    # We need to know BOTH rank's needs to route files correctly
    rank0_files = None
    rank1_files = None

    # File list for load_model (set on the returned loader function)
    file_order = None

    if distributed_mode == "pipeline" and model_path:
        # Use OOB-coordinated send/recv for manifest exchange.
        # OOB provides receiver-initiated rendezvous to avoid JACCL timing issues.

        if rank == 0:
            src_path = Path(model_path)
            if not src_path.exists():
                from huggingface_hub import snapshot_download
                src_path = Path(snapshot_download(model_path, local_files_only=True))

            index_path = src_path / "model.safetensors.index.json"
            config_path = src_path / "config.json"

            # Read full file list from index
            with open(index_path, "r") as f:
                weight_index = json.load(f)["weight_map"]
            all_weight_files = sorted(set(weight_index.values()))

            # Compute needed files for ALL ranks
            all_rank_files = {}
            if config_path.exists():
                for r in range(world_size):
                    files = compute_pipeline_files(index_path, config_path, r, world_size)
                    all_rank_files[r] = list(files) if files else None

            # Build manifest with file order and rank assignments
            manifest = {
                "file_order": all_weight_files,
                "rank_files": all_rank_files,
            }

            # Serialize manifest
            manifest_json = json.dumps(manifest).encode("utf-8")

            logger.info(
                f"{get_node_prefix(rank)} Sending manifest ({len(manifest_json)} bytes) "
                f"to {world_size - 1} workers"
            )

            # Send manifest to each worker via OOB-coordinated send
            for dst_rank in range(1, world_size):
                transfer_id = f"manifest_to_rank{dst_rank}"
                await oob_send_file_bytes_async(manifest_json, group, dst_rank, transfer_id,
                                                chunk_size=chunk_size)

            # Extract for local use
            file_order = manifest.get("file_order", [])
            all_rank_files_dict = manifest.get("rank_files", {})
        else:
            # Workers receive manifest via OOB-coordinated recv
            transfer_id = f"manifest_to_rank{rank}"
            manifest_bytes = await oob_recv_file_bytes_async(group, src_rank=0, transfer_id=transfer_id,
                                                              chunk_size=chunk_size)
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            del manifest_bytes

            # Extract file order and rank assignments
            file_order = manifest.get("file_order", [])
            all_rank_files_dict = manifest.get("rank_files", {})

        # Extract file sets (JSON keys are strings when deserialized)
        rank0_files = set(all_rank_files_dict.get(0, all_rank_files_dict.get("0", [])) or [])
        rank1_files = set(all_rank_files_dict.get(1, all_rank_files_dict.get("1", [])) or [])

        logger.info(
            f"{get_node_prefix(rank)} Pipeline mode: {len(file_order)} total files, "
            f"rank0 needs {len(rank0_files)}, rank1 needs {len(rank1_files)}"
        )

        # MANIFEST CHECKPOINT: Use OOB barrier to ensure all ranks received manifest
        await oob.barrier("manifest_exchange")
        logger.info(f"{get_node_prefix(rank)} Manifest checkpoint OK: all ranks in sync")

    logger.info(
        f"{get_node_prefix(rank)} Created distributed weight loader "
        f"(backend={backend}, chunk_size={chunk_size // 1024 // 1024}MB, mode=OOB send/recv)"
    )

    # Debug/timing log file for weight loading (bypasses all output redirection)
    _debug_log = open(f"/tmp/weight_loader_rank{rank}.log", "w")
    _file_counter = [0]  # Use list for closure mutability

    def distributed_loader(file_path: str) -> Dict[str, Any]:
        """Load weights via OOB-coordinated send/recv."""
        t_file_start = time.perf_counter()
        _file_counter[0] += 1
        file_num = _file_counter[0]

        file_name = Path(file_path).name

        # OOB barrier: verify all ranks are in sync before each file transfer.
        # Lighter than all_sum - uses TCPStore key exchange, no data transfer.
        oob_barrier_sync(f"file_{file_num}")

        # Check if any peer is terminating before starting RDMA transfer
        if oob.is_any_peer_terminating():
            logger.warning(f"[Rank {rank}] Peer terminating, aborting weight transfer")
            raise RuntimeError("Peer terminated during weight transfer")

        # Determine who needs this file
        r0_needs = rank0_files is None or file_name in rank0_files
        r1_needs = rank1_files is None or file_name in rank1_files
        i_need = (rank == 0 and r0_needs) or (rank == 1 and r1_needs)

        _debug_log.write(f"\n=== FILE {file_num}: {file_name} (barrier OK) ===\n")
        _debug_log.write(f"rank0_needs={r0_needs}, rank1_needs={r1_needs}, i_need={i_need}\n")
        _debug_log.flush()

        # Transfer ID for this file (unique per file)
        transfer_id = f"weight_file_{file_num}"

        # Determine transfer pattern based on who needs the file:
        # - Neither needs: skip entirely (both ranks)
        # - Only rank 0 needs: rank 0 reads locally, rank 1 does nothing
        # - Only rank 1 needs: rank 0 reads and sends, rank 1 receives
        # - Both need: rank 0 reads locally and sends, rank 1 receives

        if not r0_needs and not r1_needs:
            # No rank needs this file - skip entirely (both ranks must agree)
            _debug_log.write(f"  SKIP: {file_name} (no rank needs it)\n")
            _debug_log.flush()
            t_file_end = time.perf_counter()
            _debug_log.write(f"  TOTAL: {1000*(t_file_end-t_file_start):.0f}ms (skipped)\n")
            _debug_log.flush()
            return {}

        file_bytes = None

        if rank == 0:
            # Rank 0 always reads the file (it's the source)
            file_bytes = Path(file_path).read_bytes()
            _debug_log.write(f"  READ: {len(file_bytes)} bytes from {file_path}\n")
            _debug_log.flush()

            if r1_needs:
                # Send to rank 1 (OOB-coordinated: waits for receiver ready)
                _debug_log.write(f"  SEND: to rank 1 via {transfer_id}\n")
                _debug_log.flush()
                oob_send_file_bytes(file_bytes, group, dst_rank=1, transfer_id=transfer_id,
                                    chunk_size=chunk_size, log_file=_debug_log)

            if not r0_needs:
                # Rank 0 sent to rank 1 but doesn't need it itself
                del file_bytes
                _debug_log.write(f"  DISCARD: {file_name} (sent to rank 1, not needed locally)\n")
                _debug_log.flush()
                t_file_end = time.perf_counter()
                _debug_log.write(f"  TOTAL: {1000*(t_file_end-t_file_start):.0f}ms (sent only)\n")
                _debug_log.flush()
                return {}
        else:
            # Rank 1 (or other workers)
            if r1_needs:
                # Receive from rank 0 (OOB-coordinated: signals ready first)
                _debug_log.write(f"  RECV: from rank 0 via {transfer_id}\n")
                _debug_log.flush()
                file_bytes = oob_recv_file_bytes(group, src_rank=0, transfer_id=transfer_id,
                                                  chunk_size=chunk_size, log_file=_debug_log)
                _debug_log.write(f"  RECEIVED: {len(file_bytes)} bytes\n")
                _debug_log.flush()
            else:
                # Rank 1 doesn't need this file
                _debug_log.write(f"  SKIP: {file_name} (not needed by rank {rank})\n")
                _debug_log.flush()
                t_file_end = time.perf_counter()
                _debug_log.write(f"  TOTAL: {1000*(t_file_end-t_file_start):.0f}ms (not needed)\n")
                _debug_log.flush()
                return {}

        _debug_log.write(f"  KEEP: {file_name}\n")
        _debug_log.flush()

        # Parse if we have bytes
        if file_bytes is None:
            t_file_end = time.perf_counter()
            _debug_log.write(f"  TOTAL: {1000*(t_file_end-t_file_start):.0f}ms (no data)\n")
            _debug_log.flush()
            return {}

        file_size_mb = len(file_bytes) / 1e6
        t_after_transfer = time.perf_counter()

        # Parse safetensors from bytes directly to mx.array (handles BF16)
        weights = parse_safetensors(file_bytes, log_file=_debug_log)

        t_after_parse = time.perf_counter()

        # Free raw bytes immediately
        del file_bytes

        t_file_end = time.perf_counter()

        # Log timing summary
        transfer_ms = 1000 * (t_after_transfer - t_file_start)
        parse_ms = 1000 * (t_after_parse - t_after_transfer)
        total_ms = 1000 * (t_file_end - t_file_start)
        throughput = file_size_mb / (t_file_end - t_file_start) if (t_file_end - t_file_start) > 0 else 0

        _debug_log.write(
            f"  TOTAL: {total_ms:.0f}ms (transfer={transfer_ms:.0f}ms, parse={parse_ms:.0f}ms) "
            f"@ {throughput:.0f}MB/s, {len(weights)} tensors\n"
        )
        _debug_log.flush()

        logger.debug(
            f"{get_node_prefix(rank)} Loaded {file_name}: {len(weights)} tensors in {total_ms:.0f}ms"
        )

        return weights

    # Store file order on the loader function for retrieval by mlx_lm
    distributed_loader.file_order = file_order

    return distributed_loader


def get_weight_loader_file_list(weight_loader: Callable) -> Optional[list]:
    """Get the file list from a distributed weight loader.

    The file list is stored as an attribute on the loader function by
    make_distributed_weight_loader. This allows mlx_lm to use the same
    file order that was exchanged via the manifest.

    Args:
        weight_loader: The weight loader function

    Returns:
        List of weight file names, or None if not available
    """
    return getattr(weight_loader, "file_order", None)


async def validate_memory_for_streaming(
    model_path: str,
    group: mx.distributed.Group,
) -> None:
    """Validate all ranks have enough memory for streaming weight loading.

    This should be called before creating the distributed weight loader
    to fail early if any rank lacks sufficient memory.

    Uses OOB-coordinated send/recv to broadcast model size information
    from rank 0 to workers.

    Args:
        model_path: Path to model directory (rank 0 only)
        group: MLX distributed group

    Raises:
        MemoryError: If any rank has insufficient memory
        RuntimeError: If OOB coordinator is not initialized
    """
    from .oob import get_oob

    rank = group.rank()
    world_size = group.size()
    chunk_size = get_chunk_size()

    logger.info(f"{get_node_prefix(rank)} validate_memory_for_streaming: entering")

    # Get OOB coordinator for synchronization
    oob = get_oob()
    if oob is None:
        raise RuntimeError(
            "OOB coordinator not initialized. Call init_oob() before "
            "validate_memory_for_streaming(). See oob.py for details."
        )

    logger.info(f"{get_node_prefix(rank)} validate_memory_for_streaming: OOB coordinator OK")

    # Rank 0 reads manifest and sends sizes to workers
    if rank == 0:
        src_path = Path(model_path)
        if not src_path.exists():
            # Try HuggingFace cache
            from huggingface_hub import snapshot_download
            src_path = Path(snapshot_download(model_path, local_files_only=True))

        files = get_model_files(src_path)
        weight_files = get_weight_files(files)
        total_size = sum(size for _, size in weight_files)
        max_file_size = max(size for _, size in weight_files) if weight_files else 0

        logger.info(
            f"{get_node_prefix(rank)} Model: {len(weight_files)} weight files, "
            f"total {total_size / 1e9:.1f}GB, max file {max_file_size / 1e9:.1f}GB"
        )

        # Send size info to each worker via OOB-coordinated send
        sizes_json = json.dumps({"total_size": total_size, "max_file_size": max_file_size})
        sizes_bytes = sizes_json.encode("utf-8")
        for dst_rank in range(1, world_size):
            transfer_id = f"memory_validation_to_rank{dst_rank}"
            await oob_send_file_bytes_async(sizes_bytes, group, dst_rank, transfer_id,
                                            chunk_size=chunk_size)
    else:
        # Workers receive size info via OOB-coordinated recv
        transfer_id = f"memory_validation_to_rank{rank}"
        logger.info(f"{get_node_prefix(rank)} validate_memory_for_streaming: receiving sizes from rank 0")
        sizes_bytes = await oob_recv_file_bytes_async(group, src_rank=0, transfer_id=transfer_id,
                                                       chunk_size=chunk_size)
        sizes_data = json.loads(sizes_bytes.decode("utf-8"))
        total_size = sizes_data["total_size"]
        max_file_size = sizes_data["max_file_size"]

    # Each rank checks their memory
    has_memory, available, required = check_memory_for_streaming(
        total_size, max_file_size, rank
    )

    if not has_memory:
        raise MemoryError(
            f"{get_node_prefix(rank)} Insufficient memory for streaming: "
            f"need {required / 1e9:.1f}GB (model + buffer), "
            f"only {available / 1e9:.1f}GB available. "
            f"Use --file-sync=full or --file-sync=sharded to use disk instead."
        )

    if available > 0:
        logger.info(
            f"{get_node_prefix(rank)} Memory check OK: "
            f"{available / 1e9:.1f}GB available, need {required / 1e9:.1f}GB"
        )

    # Barrier to ensure all ranks completed memory check before proceeding
    await oob.barrier("memory_validation")
