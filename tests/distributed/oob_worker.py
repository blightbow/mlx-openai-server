#!/usr/bin/env python3
"""OOB Coordinator test worker for multi-process integration tests.

This script is spawned as a subprocess by test_oob_multiprocess.py to test
real cross-process coordination behavior that can't be tested in a single
process with shared event loops.

Usage:
    python oob_worker.py <rank> <world_size> <port> <scenario> [options]

Scenarios:
    startup_only          - Start coordinator, wait for barrier, exit
    startup_then_block    - Start coordinator, then simulate blocking (sleep)
    startup_slow          - Add delay before startup barrier completes
    barrier_then_exit     - Do startup + one barrier, then exit
    abort_sender          - Do startup, sync barrier, send abort
    abort_receiver        - Do startup, sync barrier, wait for abort in second barrier
    term_sender           - Do startup, sync barrier, signal termination
    term_receiver         - Do startup, sync barrier, wait for termination in second barrier
    sigterm_during_idle   - Start and wait for SIGTERM during idle
"""

import argparse
import asyncio
import os
import signal
import sys
import time

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.distributed.oob import (
    AbortError,
    OOBCoordinator,
    PeerTerminatedError,
    PeerTimeoutError,
)


class WorkerResult:
    """Result codes for worker exit status."""

    SUCCESS = 0
    STARTUP_TIMEOUT = 10
    BARRIER_TIMEOUT = 11
    PEER_TERMINATED = 12
    ABORT_RECEIVED = 13
    UNEXPECTED_ERROR = 20
    SIGTERM_RECEIVED = 30


def write_status(status_file: str, message: str) -> None:
    """Write status to file for test coordination."""
    if status_file:
        with open(status_file, "a") as f:
            f.write(f"{time.time():.6f} {message}\n")
            f.flush()


async def scenario_startup_only(
    oob: OOBCoordinator, args: argparse.Namespace
) -> int:
    """Just start and exit - tests basic startup barrier.

    Includes a final barrier to ensure all ranks complete before any exits.
    Without this, rank 0 could exit and close the store while rank 1 is
    still polling during startup phase 2.
    """
    write_status(args.status_file, f"rank{oob.rank}_started")

    # Final barrier ensures all ranks completed startup before any exits
    try:
        await oob.barrier("exit_sync", timeout=args.timeout)
        write_status(args.status_file, f"rank{oob.rank}_exit_sync_passed")
    except PeerTimeoutError:
        write_status(args.status_file, f"rank{oob.rank}_exit_sync_timeout")
        return WorkerResult.BARRIER_TIMEOUT

    return WorkerResult.SUCCESS


async def scenario_startup_then_block(
    oob: OOBCoordinator, args: argparse.Namespace
) -> int:
    """Start coordinator, then simulate blocking like mx.distributed.init().

    This is the scenario that caused the original deadlock: rank 0 would
    complete startup and block, preventing store server from responding.
    """
    write_status(args.status_file, f"rank{oob.rank}_started")

    # Simulate blocking operation (like mx.distributed.init)
    # Use sync sleep to actually block the event loop
    write_status(args.status_file, f"rank{oob.rank}_blocking")
    time.sleep(args.block_duration)
    write_status(args.status_file, f"rank{oob.rank}_unblocked")

    # Final barrier ensures all ranks sync before exit
    try:
        await oob.barrier("exit_sync", timeout=args.timeout)
        write_status(args.status_file, f"rank{oob.rank}_exit_sync_passed")
    except PeerTimeoutError:
        write_status(args.status_file, f"rank{oob.rank}_exit_sync_timeout")
        return WorkerResult.BARRIER_TIMEOUT

    return WorkerResult.SUCCESS


async def scenario_startup_slow(
    oob: OOBCoordinator, args: argparse.Namespace
) -> int:
    """Simulate slow startup by delaying store queries.

    Only rank 1+ should use this - rank 0 is the store server.
    The delay is injected before OOB start in main().
    """
    write_status(args.status_file, f"rank{oob.rank}_started")

    # Final barrier ensures all ranks sync before exit
    try:
        await oob.barrier("exit_sync", timeout=args.timeout)
        write_status(args.status_file, f"rank{oob.rank}_exit_sync_passed")
    except PeerTimeoutError:
        write_status(args.status_file, f"rank{oob.rank}_exit_sync_timeout")
        return WorkerResult.BARRIER_TIMEOUT

    return WorkerResult.SUCCESS


async def scenario_barrier_then_exit(
    oob: OOBCoordinator, args: argparse.Namespace
) -> int:
    """Complete startup + one explicit barrier, then exit."""
    write_status(args.status_file, f"rank{oob.rank}_started")

    try:
        await oob.barrier("test_barrier", timeout=args.timeout)
        write_status(args.status_file, f"rank{oob.rank}_barrier_passed")
    except PeerTimeoutError:
        write_status(args.status_file, f"rank{oob.rank}_barrier_timeout")
        return WorkerResult.BARRIER_TIMEOUT

    return WorkerResult.SUCCESS


async def scenario_abort_sender(
    oob: OOBCoordinator, args: argparse.Namespace
) -> int:
    """Sync at barrier, then send abort signal (rank 0 only)."""
    write_status(args.status_file, f"rank{oob.rank}_started")

    try:
        # First barrier: sync with receiver
        await oob.barrier("sync_barrier", timeout=args.timeout)
        write_status(args.status_file, f"rank{oob.rank}_sync_barrier_passed")
    except PeerTimeoutError:
        write_status(args.status_file, f"rank{oob.rank}_sync_barrier_timeout")
        return WorkerResult.BARRIER_TIMEOUT

    # Small delay to let receiver enter second barrier
    await asyncio.sleep(0.2)

    # Send abort
    await oob.abort()
    write_status(args.status_file, f"rank{oob.rank}_abort_sent")

    return WorkerResult.SUCCESS


async def scenario_abort_receiver(
    oob: OOBCoordinator, args: argparse.Namespace
) -> int:
    """Sync at barrier, then wait for abort in second barrier."""
    write_status(args.status_file, f"rank{oob.rank}_started")

    try:
        # First barrier: sync with sender
        await oob.barrier("sync_barrier", timeout=args.timeout)
        write_status(args.status_file, f"rank{oob.rank}_sync_barrier_passed")
    except PeerTimeoutError:
        write_status(args.status_file, f"rank{oob.rank}_sync_barrier_timeout")
        return WorkerResult.BARRIER_TIMEOUT

    try:
        # Second barrier: should be interrupted by abort
        await oob.barrier("abort_target_barrier", timeout=args.timeout)
        write_status(args.status_file, f"rank{oob.rank}_abort_barrier_passed")
        return WorkerResult.SUCCESS
    except AbortError:
        write_status(args.status_file, f"rank{oob.rank}_abort_received")
        return WorkerResult.ABORT_RECEIVED
    except PeerTimeoutError:
        write_status(args.status_file, f"rank{oob.rank}_abort_barrier_timeout")
        return WorkerResult.BARRIER_TIMEOUT


async def scenario_term_sender(
    oob: OOBCoordinator, args: argparse.Namespace
) -> int:
    """Sync at barrier, then signal termination."""
    write_status(args.status_file, f"rank{oob.rank}_started")

    try:
        # First barrier: sync with receiver
        await oob.barrier("sync_barrier", timeout=args.timeout)
        write_status(args.status_file, f"rank{oob.rank}_sync_barrier_passed")
    except PeerTimeoutError:
        write_status(args.status_file, f"rank{oob.rank}_sync_barrier_timeout")
        return WorkerResult.BARRIER_TIMEOUT

    # Small delay to let receiver enter second barrier
    await asyncio.sleep(0.2)

    # Signal termination
    await oob.signal_terminating()
    write_status(args.status_file, f"rank{oob.rank}_termination_sent")

    return WorkerResult.SUCCESS


async def scenario_term_receiver(
    oob: OOBCoordinator, args: argparse.Namespace
) -> int:
    """Sync at barrier, then wait for termination in second barrier."""
    write_status(args.status_file, f"rank{oob.rank}_started")

    try:
        # First barrier: sync with sender
        await oob.barrier("sync_barrier", timeout=args.timeout)
        write_status(args.status_file, f"rank{oob.rank}_sync_barrier_passed")
    except PeerTimeoutError:
        write_status(args.status_file, f"rank{oob.rank}_sync_barrier_timeout")
        return WorkerResult.BARRIER_TIMEOUT

    try:
        # Second barrier: should be interrupted by peer termination
        await oob.barrier("term_target_barrier", timeout=args.timeout)
        write_status(args.status_file, f"rank{oob.rank}_term_barrier_passed")
        return WorkerResult.SUCCESS
    except PeerTerminatedError:
        write_status(args.status_file, f"rank{oob.rank}_peer_terminated")
        return WorkerResult.PEER_TERMINATED
    except PeerTimeoutError:
        write_status(args.status_file, f"rank{oob.rank}_term_barrier_timeout")
        return WorkerResult.BARRIER_TIMEOUT


async def scenario_sigterm_during_idle(
    oob: OOBCoordinator, args: argparse.Namespace
) -> int:
    """Start and wait for SIGTERM while idle.

    Tests that SIGTERM is handled gracefully during idle periods.
    """
    write_status(args.status_file, f"rank{oob.rank}_started")

    # Set up a future that completes when SIGTERM is received
    loop = asyncio.get_event_loop()
    sigterm_future = loop.create_future()

    def sigterm_handler(signum, frame):
        write_status(args.status_file, f"rank{oob.rank}_sigterm_received")
        if not sigterm_future.done():
            loop.call_soon_threadsafe(sigterm_future.set_result, True)

    signal.signal(signal.SIGTERM, sigterm_handler)

    # Wait for SIGTERM or timeout
    try:
        await asyncio.wait_for(sigterm_future, timeout=args.timeout)
        return WorkerResult.SIGTERM_RECEIVED
    except asyncio.TimeoutError:
        write_status(args.status_file, f"rank{oob.rank}_sigterm_timeout")
        return WorkerResult.BARRIER_TIMEOUT


SCENARIOS = {
    "startup_only": scenario_startup_only,
    "startup_then_block": scenario_startup_then_block,
    "startup_slow": scenario_startup_slow,
    "barrier_then_exit": scenario_barrier_then_exit,
    "abort_sender": scenario_abort_sender,
    "abort_receiver": scenario_abort_receiver,
    "term_sender": scenario_term_sender,
    "term_receiver": scenario_term_receiver,
    "sigterm_during_idle": scenario_sigterm_during_idle,
}


async def main_async(args: argparse.Namespace) -> int:
    """Async main entry point."""
    scenario_fn = SCENARIOS.get(args.scenario)
    if scenario_fn is None:
        print(f"Unknown scenario: {args.scenario}", file=sys.stderr)
        return WorkerResult.UNEXPECTED_ERROR

    # Pre-startup delay for slow startup scenario
    if args.scenario == "startup_slow" and args.rank > 0:
        write_status(args.status_file, f"rank{args.rank}_delaying")
        await asyncio.sleep(args.startup_delay)

    write_status(args.status_file, f"rank{args.rank}_oob_starting")

    try:
        oob = OOBCoordinator(
            rank=args.rank,
            world_size=args.world_size,
            host=args.host,
            port=args.port,
            timeout_sec=args.timeout,
        )

        async with oob:
            write_status(args.status_file, f"rank{args.rank}_oob_ready")
            result = await scenario_fn(oob, args)

        write_status(args.status_file, f"rank{args.rank}_oob_stopped")
        return result

    except PeerTimeoutError as e:
        write_status(args.status_file, f"rank{args.rank}_startup_timeout:{e}")
        return WorkerResult.STARTUP_TIMEOUT
    except Exception as e:
        write_status(args.status_file, f"rank{args.rank}_error:{type(e).__name__}:{e}")
        import traceback
        traceback.print_exc()
        return WorkerResult.UNEXPECTED_ERROR


def main() -> int:
    parser = argparse.ArgumentParser(description="OOB test worker")
    parser.add_argument("rank", type=int, help="Worker rank")
    parser.add_argument("world_size", type=int, help="Total world size")
    parser.add_argument("port", type=int, help="OOB base port")
    parser.add_argument("scenario", choices=list(SCENARIOS.keys()), help="Test scenario")

    parser.add_argument("--host", default="127.0.0.1", help="Coordinator host")
    parser.add_argument("--timeout", type=float, default=10.0, help="Operation timeout")
    parser.add_argument("--status-file", dest="status_file", help="File to write status updates")
    parser.add_argument("--block-duration", dest="block_duration", type=float, default=2.0,
                        help="Duration to block in startup_then_block scenario")
    parser.add_argument("--startup-delay", dest="startup_delay", type=float, default=1.0,
                        help="Delay before startup in startup_slow scenario")

    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
