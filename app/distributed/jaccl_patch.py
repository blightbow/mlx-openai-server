"""JACCL-safe wrappers for mx.distributed.send/recv operations.

This module provides monkey-patches for mx.distributed.send and recv_like
that add OOB coordination required by JACCL's RDMA implementation.

The Problem:
    JACCL's send/recv lacks MPI's built-in rendezvous protocol. When sender
    and receiver have asymmetric timing, RDMA operations fail or hang.
    mlx-lm's pipeline parallelism uses raw send/recv in the forward pass,
    which hangs on JACCL without coordination.

The Solution:
    Wrap send/recv with OOB receiver-initiated rendezvous:
    1. Receiver signals ready via OOB store
    2. Sender waits for ready signal
    3. Safe to perform RDMA transfer

Usage:
    # Call once after OOB is initialized and before model inference
    from app.distributed.jaccl_patch import patch_jaccl_send_recv
    patch_jaccl_send_recv()

    # All subsequent mx.distributed.send/recv_like calls are now safe
"""

import threading
from typing import Optional

import mlx.core as mx
from loguru import logger

# Module state
_patched = False
_call_counter = 0
_counter_lock = threading.Lock()

# Store original functions
_original_send = None
_original_recv_like = None


def _get_transfer_id() -> str:
    """Generate a unique transfer ID for each send/recv pair."""
    global _call_counter
    with _counter_lock:
        _call_counter += 1
        return f"pipeline_fwd_{_call_counter}"


def _oob_send(
    x: mx.array,
    dst: int,
    group: Optional[mx.distributed.Group] = None,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """OOB-coordinated send for JACCL.

    Waits for receiver to signal ready before sending.
    Forces evaluation to ensure data is actually transferred before
    signaling completion (MLX operations are lazy by default).
    """
    from .oob import get_oob

    oob = get_oob()
    if oob is None:
        # No OOB, use original (non-JACCL or single rank)
        return _original_send(x, dst, group=group, stream=stream)

    transfer_id = _get_transfer_id()

    # Wait for receiver to be ready (receiver-initiated rendezvous)
    try:
        oob.wait_ready(transfer_id, dst)
    except Exception as e:
        logger.error(f"[Rank {oob.rank}] OOB wait_ready failed for send to {dst}: {e}")
        raise

    # Now safe to send - force evaluation to ensure data is actually transferred
    # MLX operations are lazy; without eval, the send may not happen until later
    result = _original_send(x, dst, group=group, stream=stream)
    mx.eval(result)

    # Signal completion so receiver knows transfer is done
    oob.signal_complete(transfer_id)

    return result


def _oob_recv_like(
    x: mx.array,
    src: int,
    group: Optional[mx.distributed.Group] = None,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """OOB-coordinated recv_like for JACCL.

    Signals ready before receiving, forces evaluation to ensure data
    is actually received (MLX operations are lazy by default).
    """
    from .oob import get_oob

    oob = get_oob()
    if oob is None:
        # No OOB, use original (non-JACCL or single rank)
        return _original_recv_like(x, src, group=group, stream=stream)

    transfer_id = _get_transfer_id()

    # Signal that we're ready to receive
    oob.signal_ready(transfer_id)

    # Now safe to receive - force evaluation to ensure data is actually transferred
    # MLX operations are lazy; without eval, the recv may not happen until later,
    # causing timing issues with subsequent operations
    result = _original_recv_like(x, src, group=group, stream=stream)
    mx.eval(result)

    # Wait for sender to signal completion (ensures sender has finished)
    try:
        oob.wait_complete(transfer_id, src)
    except Exception as e:
        logger.error(f"[Rank {oob.rank}] OOB wait_complete failed for recv from {src}: {e}")
        raise

    return result


def patch_jaccl_send_recv() -> bool:
    """Patch mx.distributed.send/recv_like with OOB-coordinated versions.

    This should be called once after OOB is initialized and before any
    pipeline model inference. Safe to call multiple times (idempotent).

    Returns:
        True if patched, False if already patched
    """
    global _patched, _original_send, _original_recv_like

    if _patched:
        logger.debug("JACCL send/recv already patched, skipping")
        return False

    # Store originals
    _original_send = mx.distributed.send
    _original_recv_like = mx.distributed.recv_like

    # Apply patches
    mx.distributed.send = _oob_send
    mx.distributed.recv_like = _oob_recv_like

    _patched = True
    logger.info("Patched mx.distributed.send/recv_like with OOB coordination for JACCL")
    return True


def unpatch_jaccl_send_recv() -> bool:
    """Restore original mx.distributed.send/recv_like.

    Returns:
        True if unpatched, False if wasn't patched
    """
    global _patched, _original_send, _original_recv_like

    if not _patched:
        return False

    mx.distributed.send = _original_send
    mx.distributed.recv_like = _original_recv_like

    _patched = False
    _original_send = None
    _original_recv_like = None

    logger.info("Restored original mx.distributed.send/recv_like")
    return True


def is_patched() -> bool:
    """Check if send/recv are currently patched."""
    return _patched


def reset_call_counter() -> None:
    """Reset the transfer ID counter.

    Call this between inference requests to keep transfer IDs aligned
    between ranks. Both ranks must call this at the same point.
    """
    global _call_counter
    with _counter_lock:
        _call_counter = 0
