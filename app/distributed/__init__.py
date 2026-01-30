"""Distributed inference coordination for mlx-openai-server.

This module implements the coordination protocol for multi-rank
inference over mlx.launch backends (JACCL, ring, or MPI). The
coordinator (rank 0) serves HTTP and broadcasts tokens to workers.
Workers participate in forward passes via mx.distributed operations.

JACCL Coordination Patterns
===========================
JACCL (Thunderbolt 5 RDMA) lacks MPI's built-in rendezvous protocol,
which causes SIGBUS crashes when send/recv have asymmetric timing.
We use two patterns to handle this:

1. **all_sum() broadcast pattern** (coordinator.py)
   For synchronous operations where all ranks participate together.
   Rank 0 contributes data, workers contribute zeros → result = data.
   Used for: token broadcast, parameter sync, synchronous file transfers.

2. **OOB-coordinated send/recv** (oob.py, file_sync.py)
   For point-to-point transfers with asymmetric timing.
   Uses PyTorch TCPStore for receiver-initiated rendezvous.
   Used for: memory-mode weight streaming, targeted file transfers.

See oob.py docstring for full details on the OOB coordination layer.
Ring backend doesn't need OOB (neighbor-only, implicit sync).
MPI has built-in rendezvous and doesn't need OOB.

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
from .helpers import (
    PeerTerminatedError,
    synced_all_sum,
    broadcast_value,
    safe_collective,
)
from .testing import (
    MockOOBCoordinator,
    MockDistributedGroup,
    mock_all_sum,
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
    # Synchronization helpers
    "PeerTerminatedError",
    "synced_all_sum",
    "broadcast_value",
    "safe_collective",
    # Testing infrastructure
    "MockOOBCoordinator",
    "MockDistributedGroup",
    "mock_all_sum",
    # Hostfile (direct execution without mlx.launch)
    "HostConfig",
    "load_hostfile",
    "setup_jaccl_env",
    "get_oob_host_from_hostfile",
]
