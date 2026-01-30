"""Integration tests for ZeroMQ-based OOB coordination with real sockets.

These tests run actual OOBCoordinator instances communicating over localhost,
testing the real ZeroMQ message passing and asyncio task scheduling.

JACCL/TB5 RDMA operations are still mocked, but OOB coordination is real.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field

import pytest

from app.distributed.oob import (
    AbortError,
    OOBCoordinator,
    PeerTerminatedError,
    PeerTimeoutError,
)


@dataclass
class TimingEvent:
    """A single timing event for debugging coordination."""

    timestamp: float
    rank: int
    event: str
    details: str = ""

    def __str__(self) -> str:
        return f"[{self.timestamp:.6f}] Rank {self.rank}: {self.event} {self.details}"


@dataclass
class TimingLog:
    """Collects timing events across all ranks for debugging."""

    events: list[TimingEvent] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def log(self, rank: int, event: str, details: str = "") -> None:
        """Record a timing event."""
        elapsed = time.time() - self.start_time
        self.events.append(TimingEvent(elapsed, rank, event, details))

    def dump(self) -> str:
        """Return formatted timing log sorted by timestamp."""
        sorted_events = sorted(self.events, key=lambda e: e.timestamp)
        return "\n".join(str(e) for e in sorted_events)

    def clear(self) -> None:
        """Clear all events and reset start time."""
        self.events.clear()
        self.start_time = time.time()


def get_test_port() -> int:
    """Get a random port for testing to avoid conflicts between tests."""
    return random.randint(30000, 40000)


class InstrumentedOOBCoordinator(OOBCoordinator):
    """OOBCoordinator with timing instrumentation for test debugging.

    Wraps all coordination methods to log timing events, enabling
    detailed analysis of message flow and latency during tests.

    Timing events are also written to the store with prefix "timing_"
    so they can be queried externally during deadlocks.
    """

    def __init__(
        self,
        *args,
        timing_log: TimingLog | None = None,
        write_to_store: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._timing = timing_log or TimingLog()
        self._write_to_store = write_to_store
        self._event_counter = 0

    def _log(self, event: str, details: str = "") -> None:
        """Log a timing event for this rank."""
        self._timing.log(self.rank, event, details)

        # Also write to store for external debugging
        if self._write_to_store and self.rank == 0:
            self._event_counter += 1
            elapsed = time.time() - self._timing.start_time
            key = f"timing_{self._event_counter:04d}"
            value = f"{elapsed:.6f}|{self.rank}|{event}|{details}"
            # Fire and forget - don't await, just schedule
            asyncio.create_task(self._store_timing_event(key, value))

    async def _store_timing_event(self, key: str, value: str) -> None:
        """Store a timing event (fire and forget)."""
        try:
            self._store[key] = value
        except Exception:
            pass  # Best effort

    async def start(self) -> None:
        self._log("start", "begin")
        await super().start()
        self._log("start", "complete")

    async def stop(self) -> None:
        self._log("stop", "begin")
        await super().stop()
        self._log("stop", "complete")

    async def barrier(self, name: str, timeout: float | None = None) -> None:
        self._log("barrier", f"enter '{name}'")
        # Track barrier state in store for external debugging
        if self._write_to_store:
            await self._store_set(f"barrier_state_{name}_rank{self.rank}", "waiting")

        start = time.time()
        try:
            await super().barrier(name, timeout)
            elapsed_ms = (time.time() - start) * 1000
            self._log("barrier", f"exit '{name}' ({elapsed_ms:.2f}ms)")
            if self._write_to_store:
                await self._store_set(f"barrier_state_{name}_rank{self.rank}", "passed")
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            self._log("barrier", f"failed '{name}' ({elapsed_ms:.2f}ms): {e}")
            if self._write_to_store:
                await self._store_set(f"barrier_state_{name}_rank{self.rank}", f"failed:{e}")
            raise

    async def signal_ready(self, transfer_id: str) -> None:
        self._log("signal_ready", transfer_id)
        await super().signal_ready(transfer_id)

    async def wait_ready(
        self, transfer_id: str, receiver_rank: int, timeout: float | None = None
    ) -> None:
        self._log("wait_ready", f"{transfer_id} from rank {receiver_rank}")
        start = time.time()
        try:
            await super().wait_ready(transfer_id, receiver_rank, timeout)
            elapsed_ms = (time.time() - start) * 1000
            self._log("wait_ready", f"{transfer_id} done ({elapsed_ms:.2f}ms)")
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            self._log("wait_ready", f"{transfer_id} failed ({elapsed_ms:.2f}ms): {e}")
            raise

    async def signal_complete(self, transfer_id: str) -> None:
        self._log("signal_complete", transfer_id)
        await super().signal_complete(transfer_id)

    async def wait_complete(
        self, transfer_id: str, sender_rank: int, timeout: float | None = None
    ) -> None:
        self._log("wait_complete", f"{transfer_id} from rank {sender_rank}")
        start = time.time()
        try:
            await super().wait_complete(transfer_id, sender_rank, timeout)
            elapsed_ms = (time.time() - start) * 1000
            self._log("wait_complete", f"{transfer_id} done ({elapsed_ms:.2f}ms)")
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            self._log("wait_complete", f"{transfer_id} failed ({elapsed_ms:.2f}ms): {e}")
            raise

    async def signal_terminating(self) -> None:
        self._log("signal_terminating", "")
        await super().signal_terminating()

    async def abort(self) -> None:
        self._log("abort", "broadcast")
        await super().abort()


class OOBTestHarness:
    """Test harness that runs real OOB coordinators for integration tests.

    This creates multiple InstrumentedOOBCoordinator instances that communicate
    over localhost via ZeroMQ, enabling integration testing of the coordination
    layer without requiring actual RDMA hardware.

    All coordination operations are timed and logged for debugging. Access
    the timing log via harness.timing_log.

    Usage:
        async with OOBTestHarness(world_size=2) as harness:
            # Both coordinators are started and connected
            await asyncio.gather(
                harness[0].barrier("test"),
                harness[1].barrier("test"),
            )
            # On failure, print timing info:
            print(harness.timing_log.dump())
    """

    def __init__(
        self,
        world_size: int = 2,
        base_port: int | None = None,
        timeout_sec: float = 5.0,
    ) -> None:
        """Initialize test harness.

        Args:
            world_size: Number of ranks to simulate
            base_port: Base port for sockets (random if None)
            timeout_sec: Timeout for operations (shorter for tests)
        """
        self.world_size = world_size
        self.base_port = base_port or get_test_port()
        self.timeout_sec = timeout_sec
        self.coordinators: list[InstrumentedOOBCoordinator] = []
        self.timing_log = TimingLog()
        self._started = False

    async def __aenter__(self) -> "OOBTestHarness":
        """Start all coordinators."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Stop all coordinators."""
        await self.stop()

    async def start(self) -> None:
        """Start all coordinators concurrently.

        All coordinators must start together because startup_barrier
        waits for all ranks to be present.
        """
        if self._started:
            return

        # Reset timing log for this test
        self.timing_log.clear()

        # Print diagnostic hint for debugging deadlocks
        store_port = self.base_port + 1
        print(
            f"\n[OOB Test] Coordinators starting on port {self.base_port}\n"
            f"[OOB Test] If test hangs, query state from another terminal:\n"
            f"[OOB Test]   python scripts/oob_diag.py --port {store_port} status\n"
            f"[OOB Test]   python scripts/oob_diag.py --port {store_port} dump\n"
        )

        # Create all instrumented coordinators
        for rank in range(self.world_size):
            coord = InstrumentedOOBCoordinator(
                rank=rank,
                world_size=self.world_size,
                host="127.0.0.1",
                port=self.base_port,
                timeout_sec=self.timeout_sec,
                timing_log=self.timing_log,
            )
            self.coordinators.append(coord)

        # Start all coordinators concurrently - they need each other
        # for the startup barrier to complete
        await asyncio.gather(*[coord.start() for coord in self.coordinators])

        self._started = True

    async def stop(self) -> None:
        """Stop all coordinators in reverse order."""
        if not self._started:
            return

        # Stop in reverse order (workers first, then coordinator)
        for coord in reversed(self.coordinators):
            try:
                await coord.stop()
            except Exception:
                pass  # Best effort cleanup

        self.coordinators.clear()
        self._started = False

    def __getitem__(self, rank: int) -> OOBCoordinator:
        """Get coordinator by rank."""
        return self.coordinators[rank]

    def __len__(self) -> int:
        """Get number of coordinators."""
        return len(self.coordinators)

    def dump_timing(self) -> str:
        """Get formatted timing log for debugging failures."""
        return self.timing_log.dump()


# --- Harness Tests ---


class TestOOBTestHarness:
    """Tests for the test harness itself."""

    async def test_harness_starts_and_stops(self) -> None:
        """Harness can start and stop coordinators."""
        async with OOBTestHarness(world_size=2) as harness:
            assert len(harness) == 2
            assert harness[0].rank == 0
            assert harness[1].rank == 1

    async def test_harness_startup_barrier_passes(self) -> None:
        """Startup barrier completes successfully."""
        # If startup fails, this will timeout
        async with OOBTestHarness(world_size=2, timeout_sec=5.0):
            pass  # Just testing startup/shutdown

    async def test_timing_log_captures_events(self) -> None:
        """Timing log captures coordination events with timestamps."""
        async with OOBTestHarness(world_size=2) as harness:
            await asyncio.gather(
                harness[0].barrier("timed_test"),
                harness[1].barrier("timed_test"),
            )

            # Verify timing events were captured
            log = harness.timing_log
            assert len(log.events) > 0

            # Check we have barrier events from both ranks
            barrier_events = [e for e in log.events if "barrier" in e.event]
            ranks_seen = {e.rank for e in barrier_events}
            assert 0 in ranks_seen
            assert 1 in ranks_seen

            # Verify timestamps are ordered
            timestamps = [e.timestamp for e in log.events]
            assert timestamps == sorted(timestamps)

    async def test_timing_dump_format(self) -> None:
        """Timing dump produces readable output."""
        async with OOBTestHarness(world_size=2) as harness:
            await asyncio.gather(
                harness[0].barrier("format_test"),
                harness[1].barrier("format_test"),
            )

            dump = harness.dump_timing()
            assert "Rank 0" in dump
            assert "Rank 1" in dump
            assert "barrier" in dump
            assert "format_test" in dump


# --- Barrier Integration Tests ---


class TestBarrierIntegration:
    """Integration tests for barrier synchronization."""

    async def test_barrier_two_ranks(self) -> None:
        """Two ranks can synchronize on a barrier."""
        async with OOBTestHarness(world_size=2) as harness:
            # Both ranks enter barrier concurrently
            await asyncio.gather(
                harness[0].barrier("test_barrier"),
                harness[1].barrier("test_barrier"),
            )

    async def test_barrier_multiple_sequential(self) -> None:
        """Multiple sequential barriers work correctly."""
        async with OOBTestHarness(world_size=2) as harness:
            for i in range(3):
                await asyncio.gather(
                    harness[0].barrier(f"barrier_{i}"),
                    harness[1].barrier(f"barrier_{i}"),
                )

    async def test_barrier_different_names(self) -> None:
        """Barriers with different names are independent."""
        async with OOBTestHarness(world_size=2) as harness:
            # First barrier
            await asyncio.gather(
                harness[0].barrier("alpha"),
                harness[1].barrier("alpha"),
            )
            # Second barrier with different name
            await asyncio.gather(
                harness[0].barrier("beta"),
                harness[1].barrier("beta"),
            )

    async def test_barrier_staggered_arrival(self) -> None:
        """Barrier works when one rank arrives later."""
        async with OOBTestHarness(world_size=2) as harness:

            async def delayed_barrier() -> None:
                await asyncio.sleep(0.1)  # Arrive 100ms late
                await harness[1].barrier("staggered")

            await asyncio.gather(
                harness[0].barrier("staggered"),
                delayed_barrier(),
            )

    async def test_barrier_timeout(self) -> None:
        """Barrier times out if a rank doesn't arrive."""
        async with OOBTestHarness(world_size=2, timeout_sec=0.5) as harness:
            # Only rank 0 enters barrier - should timeout
            with pytest.raises(PeerTimeoutError):
                await harness[0].barrier("solo", timeout=0.3)

    async def test_barrier_three_ranks(self) -> None:
        """Barrier works with three ranks."""
        async with OOBTestHarness(world_size=3) as harness:
            await asyncio.gather(
                harness[0].barrier("three_way"),
                harness[1].barrier("three_way"),
                harness[2].barrier("three_way"),
            )


# --- Store Integration Tests ---


class TestStoreIntegration:
    """Integration tests for key-value store operations."""

    async def test_signal_ready_wait_ready(self) -> None:
        """Receiver signals ready, sender waits for it."""
        async with OOBTestHarness(world_size=2) as harness:
            transfer_id = "xfer_001"

            # Rank 1 (receiver) signals ready
            await harness[1].signal_ready(transfer_id)

            # Rank 0 (sender) waits for ready
            await harness[0].wait_ready(transfer_id, receiver_rank=1)

    async def test_signal_complete_wait_complete(self) -> None:
        """Sender signals complete, receiver waits for it."""
        async with OOBTestHarness(world_size=2) as harness:
            transfer_id = "xfer_002"

            # Rank 0 (sender) signals complete
            await harness[0].signal_complete(transfer_id)

            # Rank 1 (receiver) waits for complete
            await harness[1].wait_complete(transfer_id, sender_rank=0)

    async def test_wait_ready_timeout(self) -> None:
        """wait_ready times out if receiver doesn't signal."""
        async with OOBTestHarness(world_size=2, timeout_sec=1.0) as harness:
            with pytest.raises(PeerTimeoutError):
                await harness[0].wait_ready("never_ready", receiver_rank=1, timeout=0.2)

    async def test_bidirectional_ready_signals(self) -> None:
        """Both ranks can signal and wait for ready."""
        async with OOBTestHarness(world_size=2) as harness:
            # Both ranks signal ready for their respective receives
            await asyncio.gather(
                harness[0].signal_ready("recv_at_0"),
                harness[1].signal_ready("recv_at_1"),
            )

            # Both ranks wait for the other's ready signal
            await asyncio.gather(
                harness[0].wait_ready("recv_at_1", receiver_rank=1),
                harness[1].wait_ready("recv_at_0", receiver_rank=0),
            )


# --- Termination Integration Tests ---


class TestTerminationIntegration:
    """Integration tests for termination signaling."""

    async def test_signal_terminating_detected(self) -> None:
        """Termination signal is detected by other ranks."""
        async with OOBTestHarness(world_size=2) as harness:
            # Initially no termination
            assert not harness[0].is_any_peer_terminating()
            assert not harness[1].is_any_peer_terminating()

            # Rank 1 signals termination
            await harness[1].signal_terminating()

            # Give time for broadcast to propagate
            await asyncio.sleep(0.2)

            # Rank 0 should detect it
            assert harness[0].is_any_peer_terminating()

    async def test_barrier_fails_after_termination(self) -> None:
        """Barrier raises PeerTerminatedError after peer terminates."""
        async with OOBTestHarness(world_size=2) as harness:
            # Rank 1 signals termination
            await harness[1].signal_terminating()
            await asyncio.sleep(0.2)

            # Rank 0's barrier should fail
            with pytest.raises(PeerTerminatedError):
                await harness[0].barrier("should_fail")


# --- Abort Integration Tests ---


class TestAbortIntegration:
    """Integration tests for abort signaling."""

    async def test_abort_broadcast(self) -> None:
        """Abort signal is broadcast to all ranks."""
        async with OOBTestHarness(world_size=2) as harness:
            # Rank 0 broadcasts abort
            await harness[0].abort()

            # Give time for broadcast to propagate
            await asyncio.sleep(0.2)

            # Both ranks should have abort flag
            assert harness[0]._abort_flag
            assert harness[1]._abort_flag

    async def test_barrier_fails_after_abort(self) -> None:
        """Barrier raises AbortError after abort."""
        async with OOBTestHarness(world_size=2) as harness:
            await harness[0].abort()
            await asyncio.sleep(0.2)

            with pytest.raises(AbortError):
                await harness[1].barrier("should_fail")


# --- Stress Tests ---


class TestOOBStress:
    """Stress tests for OOB coordination."""

    async def test_rapid_barriers(self) -> None:
        """Many rapid sequential barriers complete successfully."""
        async with OOBTestHarness(world_size=2) as harness:
            for i in range(20):
                await asyncio.gather(
                    harness[0].barrier(f"rapid_{i}"),
                    harness[1].barrier(f"rapid_{i}"),
                )

    async def test_concurrent_store_operations(self) -> None:
        """Multiple concurrent store operations work correctly."""
        async with OOBTestHarness(world_size=2) as harness:
            # Multiple transfers in flight
            transfers = [f"transfer_{i}" for i in range(5)]

            # All receivers signal ready
            await asyncio.gather(*[
                harness[1].signal_ready(t) for t in transfers
            ])

            # All senders wait for ready
            await asyncio.gather(*[
                harness[0].wait_ready(t, receiver_rank=1) for t in transfers
            ])

    async def test_interleaved_barriers_and_store(self) -> None:
        """Barriers and store operations can be interleaved."""
        async with OOBTestHarness(world_size=2) as harness:
            for i in range(5):
                # Barrier
                await asyncio.gather(
                    harness[0].barrier(f"phase_{i}"),
                    harness[1].barrier(f"phase_{i}"),
                )

                # Store operations
                await harness[1].signal_ready(f"xfer_{i}")
                await harness[0].wait_ready(f"xfer_{i}", receiver_rank=1)
