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
    sync_metadata_to_workers,
    check_disk_space,
    get_cache_path,
    make_distributed_weight_loader,
    validate_memory_for_streaming,
    oob_send_file_bytes,
    oob_recv_file_bytes,
    FileSyncError,
    DiskSpaceError,
    MemoryError,
    TransferError,
)
from .oob import (
    OOBCoordinator,
    init_oob,
    get_oob,
    oob_barrier,
)
from .hostfile import (
    HostConfig,
    load_hostfile,
    setup_jaccl_env,
    get_oob_host_from_hostfile,
)

__all__ = [
    # Coordination
    "DistributedCoordinator",
    "run_worker_loop",
    "MAX_PROMPT_LENGTH",
    "PARAM_COUNT",
    # File sync
    "sync_model_to_workers",
    "sync_metadata_to_workers",
    "check_disk_space",
    "get_cache_path",
    "make_distributed_weight_loader",
    "validate_memory_for_streaming",
    "oob_send_file_bytes",
    "oob_recv_file_bytes",
    "FileSyncError",
    "DiskSpaceError",
    "MemoryError",
    "TransferError",
    # Out-of-band coordination
    "OOBCoordinator",
    "init_oob",
    "get_oob",
    "oob_barrier",
    # Hostfile (direct execution without mlx.launch)
    "HostConfig",
    "load_hostfile",
    "setup_jaccl_env",
    "get_oob_host_from_hostfile",
]
