"""Distributed inference coordination for mlx-openai-server.

This module implements the coordination protocol for multi-rank
inference over JACCL/Thunderbolt 5 RDMA. The coordinator (rank 0)
serves HTTP and broadcasts tokens to workers. Workers participate
in forward passes via mx.distributed.all_sum().
"""

from .coordinator import (
    DistributedCoordinator,
    run_worker_loop,
    MAX_PROMPT_LENGTH,
    PARAM_COUNT,
)

__all__ = [
    "DistributedCoordinator",
    "run_worker_loop",
    "MAX_PROMPT_LENGTH",
    "PARAM_COUNT",
]
