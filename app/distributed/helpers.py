"""Async synchronization helpers for JACCL collective operations.

This module provides async helper functions that encapsulate the OOB barrier +
all_sum pattern required for reliable JACCL operations. The key insight is that
JACCL requires ranks to enter collective operations simultaneously - without OOB
synchronization, RDMA operations can receive data from subsequent operations.

Design Note: Why No CPU Stream Parameter
-----------------------------------------
JACCL internally forces all collective operations to CPU stream regardless of
the `stream` argument (see `communication_stream()` in jaccl.cpp:777-779).
This is an architectural choice because JACCL uses CPU-driven RDMA polling.

Our OOB barriers solve the DIFFERENT problem of timing synchronization -
ensuring ranks enter collective operations together. We don't specify
`stream=mx.cpu` because:
1. It's redundant for JACCL (forced internally)
2. It would suggest we're solving the timeout problem (we're not)
3. Our value-add is OOB synchronization, not stream management

Usage Examples
--------------
Warmup synchronization:
    warmup = await synced_all_sum(warmup_input, group, "warmup", oob=current_oob)

Broadcast from rank 0:
    count = await broadcast_value(mx.array([len(files)]), rank, group, "file_count", oob)

Termination-safe collective:
    try:
        length = await safe_collective(
            lambda: synced_all_sum(...), "token_length", oob
        )
    except PeerTerminatedError:
        return
"""

from typing import TYPE_CHECKING, Callable, Optional, TypeVar

import mlx.core as mx

# Import exceptions from oob module (single source of truth)
from .oob import AbortError, PeerTerminatedError, PeerTimeoutError

if TYPE_CHECKING:
    from .oob import OOBCoordinator

# Re-export exceptions for backward compatibility
__all__ = [
    "PeerTerminatedError",
    "PeerTimeoutError",
    "AbortError",
    "synced_all_sum",
    "broadcast_value",
    "safe_collective",
    "synced_all_sum_sync",
    "broadcast_value_sync",
]

R = TypeVar("R")


async def synced_all_sum(
    data: mx.array,
    group: mx.distributed.Group,
    barrier_name: str,
    oob: Optional["OOBCoordinator"] = None,
) -> mx.array:
    """Perform synchronized all_sum with optional OOB coordination.

    This is the fundamental building block for JACCL collective operations.
    The OOB barrier ensures all ranks enter the all_sum simultaneously,
    preventing the timing bug where RDMA operations receive data from
    subsequent operations.

    Args:
        data: The array to sum across all ranks
        group: The MLX distributed group for the operation
        barrier_name: Name for the OOB barrier (for debugging/logging)
        oob: Optional OOB coordinator for synchronization. If None, no
             barrier is performed (suitable for single-rank or non-JACCL).

    Returns:
        The result of all_sum, evaluated to ensure completion

    Note:
        CPU stream is NOT specified because JACCL forces CPU stream internally
        regardless of what stream is requested (see jaccl.cpp:777-779).
    """
    if oob is not None:
        await oob.barrier(barrier_name)

    result = mx.distributed.all_sum(data, group=group)
    mx.eval(result)
    return result


async def broadcast_value(
    value: mx.array,
    rank: int,
    group: mx.distributed.Group,
    barrier_name: str,
    oob: Optional["OOBCoordinator"] = None,
    source_rank: int = 0,
) -> mx.array:
    """Broadcast a value from source_rank to all ranks via all_sum.

    This pattern uses all_sum where the source rank contributes the value
    and all other ranks contribute zeros. The result is the broadcast value
    on all ranks.

    Args:
        value: The array to broadcast (only used if rank == source_rank)
        rank: This process's rank
        group: The MLX distributed group for the operation
        barrier_name: Name for the OOB barrier (for debugging/logging)
        oob: Optional OOB coordinator for synchronization
        source_rank: The rank that contributes the value (default: 0)

    Returns:
        The broadcast value on all ranks

    Example:
        # Rank 0 broadcasts file count to all ranks
        count = await broadcast_value(
            mx.array([len(files)]),
            rank=my_rank,
            group=group,
            barrier_name="file_count",
            oob=oob,
        )
    """
    if rank == source_rank:
        contribution = value
    else:
        contribution = mx.zeros_like(value)

    return await synced_all_sum(contribution, group, barrier_name, oob)


async def safe_collective(
    collective_fn: Callable[[], R],
    barrier_name: str,
    oob: Optional["OOBCoordinator"] = None,
    raise_on_termination: bool = True,
) -> Optional[R]:
    """Execute a collective operation with termination safety.

    Checks if any peer has signaled termination before and after executing
    the collective. This prevents RDMA operations against dead peers, which
    can cause hangs or crashes.

    Args:
        collective_fn: A callable that performs the collective operation.
                       Can be sync (for mx.eval patterns) or return awaitable.
        barrier_name: Name for logging (used if termination detected)
        oob: Optional OOB coordinator for termination checking
        raise_on_termination: If True (default), raises PeerTerminatedError
                              when termination is detected. If False, returns None.

    Returns:
        The result of collective_fn, or None if termination detected and
        raise_on_termination is False.

    Raises:
        PeerTerminatedError: If a peer has signaled termination and
                             raise_on_termination is True.

    Example:
        try:
            result = await safe_collective(
                lambda: mx.distributed.all_sum(data, group=group),
                "my_operation",
                oob,
            )
            mx.eval(result)
        except PeerTerminatedError:
            logger.info("Peer terminated, shutting down")
            return
    """
    # Check for termination before the collective
    if oob is not None and oob.is_any_peer_terminating():
        if raise_on_termination:
            raise PeerTerminatedError(
                f"Peer terminating detected before collective '{barrier_name}'"
            )
        return None

    # Execute the collective operation
    result = collective_fn()

    # Check for termination after the collective
    if oob is not None and oob.is_any_peer_terminating():
        if raise_on_termination:
            raise PeerTerminatedError(
                f"Peer terminating detected after collective '{barrier_name}'"
            )
        return None

    return result


# Synchronous wrappers for non-async code paths (e.g., model forward pass)


def synced_all_sum_sync(
    data: mx.array,
    group: mx.distributed.Group,
    barrier_name: str,
    oob: Optional["OOBCoordinator"] = None,
    timeout: Optional[float] = None,
) -> mx.array:
    """Synchronous version of synced_all_sum for non-async code paths.

    Uses oob_barrier_sync internally, which creates a new event loop if needed.
    Prefer the async version where possible.

    Args:
        data: The array to sum across all ranks
        group: The MLX distributed group for the operation
        barrier_name: Name for the OOB barrier (for debugging/logging)
        oob: Optional OOB coordinator for synchronization
        timeout: Optional timeout override for the barrier

    Returns:
        The result of all_sum, evaluated to ensure completion
    """
    if oob is not None:
        from .oob import oob_barrier_sync

        oob_barrier_sync(barrier_name, timeout)

    result = mx.distributed.all_sum(data, group=group)
    mx.eval(result)
    return result


def broadcast_value_sync(
    value: mx.array,
    rank: int,
    group: mx.distributed.Group,
    barrier_name: str,
    oob: Optional["OOBCoordinator"] = None,
    source_rank: int = 0,
    timeout: Optional[float] = None,
) -> mx.array:
    """Synchronous version of broadcast_value for non-async code paths.

    Args:
        value: The array to broadcast (only used if rank == source_rank)
        rank: This process's rank
        group: The MLX distributed group for the operation
        barrier_name: Name for the OOB barrier (for debugging/logging)
        oob: Optional OOB coordinator for synchronization
        source_rank: The rank that contributes the value (default: 0)
        timeout: Optional timeout override for the barrier

    Returns:
        The broadcast value on all ranks
    """
    if rank == source_rank:
        contribution = value
    else:
        contribution = mx.zeros_like(value)

    return synced_all_sum_sync(contribution, group, barrier_name, oob, timeout)
