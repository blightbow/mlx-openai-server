"""Unit tests for ZeroMQ-based out-of-band coordination module.

Tests the ZeroMQ-based OOB coordinator which provides async rendezvous
primitives for JACCL send/recv operations.
"""

import asyncio

import pytest

from app.distributed.oob import (
    AbortError,
    BarrierHandle,
    OOBCoordinator,
    PeerTerminatedError,
    PeerTimeoutError,
)
from app.distributed.testing import MockOOBCoordinator


# --- Test classes ---


class TestBarrierHandle:
    """Tests for BarrierHandle non-blocking barrier tracking."""

    def test_initial_state(self) -> None:
        """Barrier handle starts incomplete."""
        handle = BarrierHandle()
        assert not handle.is_complete()
        assert not handle._cancelled

    def test_mark_complete(self) -> None:
        """Marking complete updates state."""
        handle = BarrierHandle()
        handle._mark_complete()
        assert handle.is_complete()

    async def test_wait_already_complete(self) -> None:
        """Wait returns immediately if already complete."""
        handle = BarrierHandle()
        handle._mark_complete()
        await handle.wait(timeout=0.1)  # Should not timeout

    async def test_wait_timeout(self) -> None:
        """Wait raises PeerTimeoutError on timeout."""
        handle = BarrierHandle()
        with pytest.raises(PeerTimeoutError):
            await handle.wait(timeout=0.01)

    async def test_wait_concurrent_completion(self) -> None:
        """Wait succeeds when completed by another task."""
        handle = BarrierHandle()

        async def complete_later() -> None:
            await asyncio.sleep(0.05)
            handle._mark_complete()

        task = asyncio.create_task(complete_later())
        await handle.wait(timeout=1.0)
        assert handle.is_complete()
        await task

    def test_cancel(self) -> None:
        """Cancel sets cancelled flag."""
        handle = BarrierHandle()
        handle.cancel()
        assert handle._cancelled


class TestOOBCoordinatorUnit:
    """Unit tests for OOBCoordinator that don't require network."""

    def test_initialization(self) -> None:
        """Coordinator initializes with correct state."""
        coord = OOBCoordinator(rank=0, world_size=2, host="127.0.0.1", port=29400)

        assert coord.rank == 0
        assert coord.world_size == 2
        assert coord.host == "127.0.0.1"
        assert coord.port == 29400
        assert coord._timeout_sec == 300.0
        assert not coord._shutdown
        assert not coord._abort_flag
        assert len(coord._terminating_ranks) == 0

    def test_initialization_worker(self) -> None:
        """Worker initializes with correct state."""
        coord = OOBCoordinator(rank=1, world_size=2, host="192.168.1.1", port=29400)

        assert coord.rank == 1
        assert coord.world_size == 2
        assert coord.host == "192.168.1.1"

    def test_is_any_peer_terminating_empty(self) -> None:
        """No termination when set is empty."""
        coord = OOBCoordinator(rank=0, world_size=2, host="127.0.0.1")
        assert not coord.is_any_peer_terminating()

    def test_is_any_peer_terminating_self(self) -> None:
        """Self-termination doesn't count as peer termination."""
        coord = OOBCoordinator(rank=0, world_size=2, host="127.0.0.1")
        coord._terminating_ranks.add(0)  # Self
        assert not coord.is_any_peer_terminating()

    def test_is_any_peer_terminating_peer(self) -> None:
        """Peer termination detected."""
        coord = OOBCoordinator(rank=0, world_size=2, host="127.0.0.1")
        coord._terminating_ranks.add(1)  # Peer
        assert coord.is_any_peer_terminating()

    def test_check_peers_alive_healthy(self) -> None:
        """check_peers_alive returns True when healthy."""
        coord = OOBCoordinator(rank=0, world_size=2, host="127.0.0.1")
        assert coord.check_peers_alive()

    def test_check_peers_alive_shutdown(self) -> None:
        """check_peers_alive returns False when shutdown."""
        coord = OOBCoordinator(rank=0, world_size=2, host="127.0.0.1")
        coord._shutdown = True
        assert not coord.check_peers_alive()

    def test_check_peers_alive_aborted(self) -> None:
        """check_peers_alive returns False when aborted."""
        coord = OOBCoordinator(rank=0, world_size=2, host="127.0.0.1")
        coord._abort_flag = True
        assert not coord.check_peers_alive()

    def test_check_abort_or_termination_clean(self) -> None:
        """No exception when state is clean."""
        coord = OOBCoordinator(rank=0, world_size=2, host="127.0.0.1")
        coord._check_abort_or_termination("test")  # Should not raise

    def test_check_abort_or_termination_abort(self) -> None:
        """Raises AbortError when aborted."""
        coord = OOBCoordinator(rank=0, world_size=2, host="127.0.0.1")
        coord._abort_flag = True

        with pytest.raises(AbortError) as exc_info:
            coord._check_abort_or_termination("test_op")
        assert "test_op" in str(exc_info.value)

    def test_check_abort_or_termination_terminated(self) -> None:
        """Raises PeerTerminatedError when peer terminated."""
        coord = OOBCoordinator(rank=0, world_size=2, host="127.0.0.1")
        coord._terminating_ranks.add(1)

        with pytest.raises(PeerTerminatedError) as exc_info:
            coord._check_abort_or_termination("test_op")
        assert "test_op" in str(exc_info.value)


class TestMockOOBCoordinator:
    """Tests for MockOOBCoordinator test helper."""

    async def test_barrier_tracking(self) -> None:
        """Mock tracks barrier calls."""
        oob = MockOOBCoordinator(rank=0, world_size=2)

        await oob.barrier("test_barrier")

        oob.assert_barrier_entered("test_barrier")
        assert "test_barrier" in oob.barriers_entered

    async def test_barrier_not_entered_assertion(self) -> None:
        """Assert barrier not entered works correctly."""
        oob = MockOOBCoordinator(rank=0, world_size=2)

        oob.assert_barrier_not_entered("never_called")

        await oob.barrier("called")
        with pytest.raises(AssertionError):
            oob.assert_barrier_not_entered("called")

    async def test_ready_signal_tracking(self) -> None:
        """Mock tracks ready signals."""
        oob = MockOOBCoordinator(rank=1, world_size=2)

        await oob.signal_ready("transfer_123")

        oob.assert_ready_signaled("transfer_123")
        assert oob.ready_signals["transfer_123"] == 1

    async def test_complete_signal_tracking(self) -> None:
        """Mock tracks complete signals."""
        oob = MockOOBCoordinator(rank=0, world_size=2)

        await oob.signal_complete("transfer_456")

        oob.assert_complete_signaled("transfer_456")
        assert oob.complete_signals["transfer_456"] == 0

    async def test_peer_termination_simulation(self) -> None:
        """Mock simulates peer termination."""
        oob = MockOOBCoordinator(rank=0, world_size=2)

        assert not oob.is_any_peer_terminating()
        assert oob.check_peers_alive()

        oob.simulate_peer_termination()

        assert oob.is_any_peer_terminating()
        assert not oob.check_peers_alive()

        with pytest.raises(PeerTerminatedError):
            await oob.barrier("should_fail")

    async def test_peer_termination_reset(self) -> None:
        """Mock can reset peer termination."""
        oob = MockOOBCoordinator(rank=0, world_size=2)
        oob.simulate_peer_termination()

        oob.reset_peer_termination()

        assert not oob.is_any_peer_terminating()
        await oob.barrier("should_succeed")

    async def test_abort_simulation(self) -> None:
        """Mock simulates abort signal."""
        oob = MockOOBCoordinator(rank=0, world_size=2)

        await oob.abort()

        with pytest.raises(AbortError):
            await oob.barrier("should_fail")

    async def test_abort_reset(self) -> None:
        """Mock can reset abort flag."""
        oob = MockOOBCoordinator(rank=0, world_size=2)
        oob.simulate_abort()

        oob.reset_abort()

        await oob.barrier("should_succeed")

    async def test_wait_ready_with_termination(self) -> None:
        """wait_ready raises on peer termination."""
        oob = MockOOBCoordinator(rank=0, world_size=2)
        oob.simulate_peer_termination()

        with pytest.raises(PeerTerminatedError):
            await oob.wait_ready("transfer", receiver_rank=1)

    async def test_wait_complete_with_abort(self) -> None:
        """wait_complete raises on abort."""
        oob = MockOOBCoordinator(rank=0, world_size=2)
        oob.simulate_abort()

        with pytest.raises(AbortError):
            await oob.wait_complete("transfer", sender_rank=1)

    async def test_clear_tracking(self) -> None:
        """clear_tracking resets all state."""
        oob = MockOOBCoordinator(rank=0, world_size=2)
        await oob.barrier("test")
        await oob.signal_ready("xfer")
        await oob.signal_complete("xfer")
        oob.simulate_peer_termination()
        oob.simulate_abort()

        oob.clear_tracking()

        assert len(oob.barriers_entered) == 0
        assert len(oob.ready_signals) == 0
        assert len(oob.complete_signals) == 0
        assert not oob.is_any_peer_terminating()
        assert not oob._abort_flag

    async def test_context_manager(self) -> None:
        """Mock works as async context manager."""
        async with MockOOBCoordinator(rank=0, world_size=2) as oob:
            await oob.barrier("in_context")
            oob.assert_barrier_entered("in_context")


class TestExceptionTypes:
    """Tests for exception classes."""

    def test_peer_timeout_error(self) -> None:
        """PeerTimeoutError has correct message."""
        err = PeerTimeoutError("Barrier 'test' timed out")
        assert "timed out" in str(err)

    def test_peer_terminated_error(self) -> None:
        """PeerTerminatedError has correct message."""
        err = PeerTerminatedError("Peer rank 1 terminated")
        assert "terminated" in str(err)

    def test_abort_error(self) -> None:
        """AbortError has correct message."""
        err = AbortError("Abort during barrier")
        assert "Abort" in str(err)

    def test_exception_inheritance(self) -> None:
        """All exceptions inherit from Exception."""
        assert issubclass(PeerTimeoutError, Exception)
        assert issubclass(PeerTerminatedError, Exception)
        assert issubclass(AbortError, Exception)


class TestDeriveAuthkey:
    """Tests for hostfile-based authkey derivation."""

    def test_derive_authkey_deterministic(self) -> None:
        """Same hostfile content produces same authkey."""
        from app.distributed.hostfile import HostConfig, derive_authkey

        hosts = [
            HostConfig(ssh="mac1.local", ips=["192.168.1.10"], rdma=[None, "rdma_en2"]),
            HostConfig(ssh="mac2.local", ips=["192.168.1.11"], rdma=["rdma_en2", None]),
        ]

        key1 = derive_authkey(hosts)
        key2 = derive_authkey(hosts)

        assert key1 == key2
        assert len(key1) == 16  # SHA256 truncated to 16 bytes

    def test_derive_authkey_different_hostfiles(self) -> None:
        """Different hostfile content produces different authkeys."""
        from app.distributed.hostfile import HostConfig, derive_authkey

        hosts1 = [
            HostConfig(ssh="mac1.local", ips=["192.168.1.10"], rdma=[None, "rdma_en2"]),
            HostConfig(ssh="mac2.local", ips=["192.168.1.11"], rdma=["rdma_en2", None]),
        ]
        hosts2 = [
            HostConfig(ssh="mac3.local", ips=["192.168.1.20"], rdma=[None, "rdma_en3"]),
            HostConfig(ssh="mac4.local", ips=["192.168.1.21"], rdma=["rdma_en3", None]),
        ]

        key1 = derive_authkey(hosts1)
        key2 = derive_authkey(hosts2)

        assert key1 != key2

    def test_derive_authkey_order_matters(self) -> None:
        """Host order affects authkey (prevents rank mismatch)."""
        from app.distributed.hostfile import HostConfig, derive_authkey

        hosts_forward = [
            HostConfig(ssh="mac1.local", ips=["192.168.1.10"], rdma=[None, "rdma_en2"]),
            HostConfig(ssh="mac2.local", ips=["192.168.1.11"], rdma=["rdma_en2", None]),
        ]
        hosts_reversed = [
            HostConfig(ssh="mac2.local", ips=["192.168.1.11"], rdma=["rdma_en2", None]),
            HostConfig(ssh="mac1.local", ips=["192.168.1.10"], rdma=[None, "rdma_en2"]),
        ]

        key_forward = derive_authkey(hosts_forward)
        key_reversed = derive_authkey(hosts_reversed)

        assert key_forward != key_reversed


class TestInitOOB:
    """Tests for the init_oob convenience function."""

    async def test_get_oob_before_init(self) -> None:
        """get_oob returns None before initialization."""
        import app.distributed.oob as oob_module

        # Reset global state
        oob_module._oob_coordinator = None

        from app.distributed.oob import get_oob

        assert get_oob() is None
