"""Distributed inference coordination for mlx-openai-server.

This module implements the coordination protocol for multi-rank
inference over mlx.launch backends (JACCL, ring, or MPI). The
coordinator (rank 0) serves HTTP and broadcasts tokens to workers.
Workers participate in forward passes via mx.distributed.all_sum().

Also provides distributed file synchronization for transferring
model files to worker ranks before inference begins.
"""

from .coordinator import (
    DistributedCoordinator,
    run_worker_loop,
    MAX_PROMPT_LENGTH,
    PARAM_COUNT,
)
from .file_sync import (
    sync_model_to_workers,
    check_disk_space,
    get_cache_path,
    make_distributed_weight_loader,
    validate_memory_for_streaming,
    FileSyncError,
    DiskSpaceError,
    MemoryError,
    TransferError,
)

__all__ = [
    # Coordination
    "DistributedCoordinator",
    "run_worker_loop",
    "MAX_PROMPT_LENGTH",
    "PARAM_COUNT",
    # File sync
    "sync_model_to_workers",
    "check_disk_space",
    "get_cache_path",
    "make_distributed_weight_loader",
    "validate_memory_for_streaming",
    "FileSyncError",
    "DiskSpaceError",
    "MemoryError",
    "TransferError",
]
