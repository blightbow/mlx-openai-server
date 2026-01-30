"""JACCL-safe wrappers for mx.distributed.send/recv operations.

This module provides monkey-patches for mx.distributed.send and recv_like
that add OOB coordination required by JACCL's RDMA implementation.

The Problem:
    JACCL's send/recv lacks MPI's built-in rendezvous protocol. When sender
    and receiver have asymmetric timing, RDMA operations fail or hang.
    mlx-lm's pipeline parallelism uses raw send/recv in the forward pass,
    which hangs on JACCL without coordination.

    Additionally, MLX operations are lazy - recv_like() returns immediately
    with a lazy array, and the actual RDMA receive buffer isn't posted until
    mx.eval() is called. If the sender completes its transfer before the
    receiver posts its buffer, data is lost and the receiver hangs forever.

The Solution:
    Use OOB barrier synchronization before each send/recv pair:
    1. Both sender and receiver reach barrier with matching transfer ID
    2. After barrier, both call their respective send/recv_like + eval
    3. RDMA transfer succeeds because both sides post operations together

    The barrier approach is simpler and more robust than ready/complete
    handshakes because it guarantees both ranks are at the same point
    before any RDMA operations begin.

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

    Uses barrier synchronization to ensure receiver has posted its RDMA
    buffer before we send. This is critical because MLX operations are lazy -
    without synchronization, data can be sent before the receive buffer exists.
    """
    from .oob import get_oob

    oob = get_oob()
    if oob is None:
        # No OOB, use original (non-JACCL or single rank)
        return _original_send(x, dst, group=group, stream=stream)

    transfer_id = _get_transfer_id()

    # Barrier ensures both sender and receiver are ready for this transfer.
    # The receiver will be at this same barrier point, about to post its
    # receive buffer. After the barrier, both sides proceed together.
    try:
        oob.barrier(f"xfer_{transfer_id}")
    except Exception as e:
        logger.error(f"[Rank {oob.rank}] OOB barrier failed for send to {dst}: {e}")
        raise

    # Now safe to send - receiver is also past barrier and posting its recv
    result = _original_send(x, dst, group=group, stream=stream)
    mx.eval(result)

    return result


def _oob_recv_like(
    x: mx.array,
    src: int,
    group: Optional[mx.distributed.Group] = None,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """OOB-coordinated recv_like for JACCL.

    Uses barrier synchronization to ensure we post our RDMA receive buffer
    at the same time the sender posts its send. This is critical because
    MLX operations are lazy - without synchronization, the sender might
    complete its transfer before our buffer exists, losing the data.
    """
    from .oob import get_oob

    oob = get_oob()
    if oob is None:
        # No OOB, use original (non-JACCL or single rank)
        return _original_recv_like(x, src, group=group, stream=stream)

    transfer_id = _get_transfer_id()

    # Barrier ensures both sender and receiver are ready for this transfer.
    # The sender will be at this same barrier point. After the barrier,
    # both sides proceed to their respective send/recv + eval together.
    try:
        oob.barrier(f"xfer_{transfer_id}")
    except Exception as e:
        logger.error(f"[Rank {oob.rank}] OOB barrier failed for recv from {src}: {e}")
        raise

    # Now safe to receive - sender is also past barrier and posting its send
    result = _original_recv_like(x, src, group=group, stream=stream)
    mx.eval(result)

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
