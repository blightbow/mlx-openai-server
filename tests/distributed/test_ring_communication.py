#!/usr/bin/env python3
"""Simple distributed communication test for mlx.launch validation.

This is a quick synchronous test - both ranks execute send/recv nearly
simultaneously, so timing asymmetry is minimal. For production JACCL
usage with asymmetric timing (e.g., HTTP serving), use OOB coordination.
See app/distributed/oob.py for the OOB coordination layer.

Run with:
    mlx.launch --backend ring --hostfile ring-hosts.json -- python tests/distributed/test_ring_communication.py
    mlx.launch --backend jaccl --hostfile jaccl-hosts.json -- python tests/distributed/test_ring_communication.py

Expected output:
    [Rank 0] Initialized group with 2 ranks
    [Rank 1] Initialized group with 2 ranks
    [Rank 0] Sending tensor to rank 1...
    [Rank 1] Waiting for tensor from rank 0...
    [Rank 1] Received: [1, 2, 3, 4, 5]
    [Rank 0] All-sum result: 2.0 (expected: 2.0)
    [Rank 1] All-sum result: 2.0 (expected: 2.0)
    SUCCESS: Distributed communication working!
"""

import os
import sys

import mlx.core as mx


def main() -> int:
    """Test basic distributed operations."""
    # Debug: Print MLX-related environment variables
    print("=== Environment Variables ===", flush=True)
    for key in sorted(os.environ.keys()):
        if key.startswith(("MLX", "JACCL", "IBV")):
            val = os.environ[key]
            # Truncate long values (like file paths with JSON)
            if len(val) > 100:
                val = val[:100] + "..."
            print(f"  {key}={val}", flush=True)

    # Print contents of MLX_IBV_DEVICES file
    ibv_file = os.environ.get("MLX_IBV_DEVICES")
    if ibv_file and os.path.exists(ibv_file):
        with open(ibv_file) as f:
            print(f"  MLX_IBV_DEVICES contents: {f.read()}", flush=True)
    print("=== End Environment ===", flush=True)

    # Check distributed availability
    print(f"mx.distributed.is_available(): {mx.distributed.is_available()}", flush=True)

    # Initialize distributed group with explicit JACCL backend
    print("Calling mx.distributed.init(backend='jaccl')...", flush=True)
    try:
        group = mx.distributed.init(backend="jaccl", strict=True)
    except Exception as e:
        print(f"JACCL init failed: {e}", flush=True)
        print("Falling back to 'any' backend...", flush=True)
        group = mx.distributed.init(backend="any")
    rank = group.rank()
    size = group.size()

    print(f"[Rank {rank}] Initialized group with {size} ranks", flush=True)

    if size < 2:
        print(f"[Rank {rank}] ERROR: Need at least 2 ranks, got {size}", flush=True)
        return 1

    # Test 1: Point-to-point communication
    # Note: This works because both ranks execute nearly simultaneously.
    # For asymmetric timing over JACCL, use OOB coordination (see oob.py).
    if rank == 0:
        # Rank 0 sends data
        data = mx.array([1, 2, 3, 4, 5])
        print(f"[Rank {rank}] Sending tensor to rank 1...", flush=True)
        sent = mx.distributed.send(data, dst=1)
        mx.eval(sent)  # Eval the send result to trigger the actual send
    else:
        # Rank 1 receives data
        template = mx.zeros((5,), dtype=mx.int32)
        print(f"[Rank {rank}] Waiting for tensor from rank 0...", flush=True)
        received = mx.distributed.recv_like(template, src=0)
        mx.eval(received)
        print(f"[Rank {rank}] Received: {received.tolist()}", flush=True)

    # Barrier via all_sum
    mx.eval(mx.distributed.all_sum(mx.array(0.0)))

    # Test 2: Collective operation (all_sum)
    local_value = mx.array(1.0)
    global_sum = mx.distributed.all_sum(local_value)
    mx.eval(global_sum)

    expected = float(size)
    print(
        f"[Rank {rank}] All-sum result: {global_sum.item()} (expected: {expected})",
        flush=True,
    )

    if abs(global_sum.item() - expected) > 0.001:
        print(f"[Rank {rank}] ERROR: All-sum mismatch!", flush=True)
        return 1

    # Final barrier
    mx.eval(mx.distributed.all_sum(mx.array(0.0)))

    if rank == 0:
        print("SUCCESS: Distributed communication working!", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
