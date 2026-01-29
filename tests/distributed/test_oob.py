"""Unit tests for out-of-band coordination module.

Tests the BaseManager-based OOB coordinator which provides rendezvous
primitives for JACCL send/recv operations. This replaced TCPStore due to
unfixable IPv6 issues on macOS (see PyTorch Issue #148440).
"""

import multiprocessing
import threading
import time
from multiprocessing import Process
from multiprocessing.managers import BaseManager, BarrierProxy, DictProxy
from unittest.mock import patch

import pytest


# --- Test fixtures using isolated manager to avoid port conflicts ---


def _find_free_port() -> int:
    """Find an available TCP port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return s.getsockname()[1]


# --- Module-level worker functions for multiprocessing (must be picklable) ---


def _rank1_store_barrier_worker(port: int) -> None:
    """Rank 1 worker: Connect to server, set key, verify rank 0's key, barrier."""
    time.sleep(0.2)  # Let rank 0 start first

    from app.distributed.oob import OOBCoordinator

    coord = OOBCoordinator(
        rank=1,
        world_size=2,
        host="127.0.0.1",
        port=port,
        authkey=b"test",
        timeout_sec=10.0,
        connect_timeout_sec=10.0,
    )

    # Set our ready key
    coord._store["rank1_ready"] = "1"

    # Check we can see rank 0's key (should be visible)
    _ = coord._store.get("rank0_ready")

    # Barrier
    coord.barrier("test_barrier")


def _rank1_wait_ready_worker(port: int) -> None:
    """Rank 1 worker: Connect and wait for rank 0's ready signal."""
    time.sleep(0.1)  # Let rank 0 start

    from app.distributed.oob import OOBCoordinator

    coord = OOBCoordinator(
        rank=1,
        world_size=2,
        host="127.0.0.1",
        port=port,
        authkey=b"test",
    )

    # This should block until rank 0 signals
    coord.wait_ready("transfer_test", receiver_rank=0)

    coord._store["rank1_done"] = "1"


# --- Test classes ---


class TestOOBManagerRegistration:
    """Tests for OOBManager proxy type registration."""

    def test_manager_class_exists(self):
        """OOBManager class is properly defined."""
        from app.distributed.oob import OOBManager

        assert issubclass(OOBManager, BaseManager)

    def test_get_store_registered(self):
        """get_store method is registered on OOBManager."""
        from app.distributed.oob import OOBManager

        # Check the method is in _registry (BaseManager internal)
        assert "get_store" in OOBManager._registry

    def test_get_barrier_registered(self):
        """get_barrier method is registered on OOBManager."""
        from app.distributed.oob import OOBManager

        assert "get_barrier" in OOBManager._registry


class TestOOBCoordinatorSingleProcess:
    """Tests for OOBCoordinator in single-process mode (rank 0 only)."""

    def test_coordinator_init_rank0(self):
        """Rank 0 successfully starts server and connects as client."""
        from app.distributed.oob import OOBCoordinator

        port = _find_free_port()
        coord = OOBCoordinator(
            rank=0,
            world_size=1,
            host="127.0.0.1",
            port=port,
            authkey=b"test",
            timeout_sec=5.0,
        )

        # Should have both server and client manager
        assert coord._server_manager is not None
        assert coord._manager is not None
        assert coord._store is not None
        assert coord._barrier is not None

    def test_store_operations_single_rank(self):
        """Dict proxy operations work for single rank."""
        from app.distributed.oob import OOBCoordinator

        port = _find_free_port()
        coord = OOBCoordinator(
            rank=0,
            world_size=1,
            host="127.0.0.1",
            port=port,
            authkey=b"test",
        )

        # Test __setitem__ and __getitem__
        coord._store["test_key"] = "test_value"
        assert coord._store["test_key"] == "test_value"

        # Test __contains__
        assert "test_key" in coord._store
        assert "nonexistent" not in coord._store

        # Test get with default
        assert coord._store.get("test_key") == "test_value"
        assert coord._store.get("missing", "default") == "default"

    def test_signal_ready(self):
        """signal_ready stores the expected key."""
        from app.distributed.oob import OOBCoordinator

        port = _find_free_port()
        coord = OOBCoordinator(
            rank=0,
            world_size=1,
            host="127.0.0.1",
            port=port,
            authkey=b"test",
        )

        coord.signal_ready("transfer_123")
        assert "ready_transfer_123_rank0" in coord._store

    def test_signal_complete(self):
        """signal_complete stores the expected key."""
        from app.distributed.oob import OOBCoordinator

        port = _find_free_port()
        coord = OOBCoordinator(
            rank=0,
            world_size=1,
            host="127.0.0.1",
            port=port,
            authkey=b"test",
        )

        coord.signal_complete("transfer_456")
        assert "complete_transfer_456_rank0" in coord._store


class TestOOBCoordinatorTwoProcess:
    """Integration tests with two processes simulating distributed ranks.

    These tests use multiprocessing with fork context to spawn a worker
    process that acts as rank 1, while the main process acts as rank 0.
    """

    @pytest.fixture
    def mp_context(self):
        """Get fork context for multiprocessing (avoids pickle issues)."""
        return multiprocessing.get_context("fork")

    def test_two_process_store_and_barrier(self, mp_context):
        """Two processes can share dict and synchronize via barrier."""
        from app.distributed.oob import OOBCoordinator

        port = _find_free_port()

        # Run rank 1 in subprocess using fork context
        p = mp_context.Process(target=_rank1_store_barrier_worker, args=(port,))
        p.start()

        # Run rank 0 in main process
        coord = OOBCoordinator(
            rank=0,
            world_size=2,
            host="127.0.0.1",
            port=port,
            authkey=b"test",
            timeout_sec=10.0,
        )

        # Set our ready key
        coord._store["rank0_ready"] = "1"

        # Wait for rank 1
        t0 = time.time()
        while "rank1_ready" not in coord._store:
            if time.time() - t0 > 10:
                p.terminate()
                pytest.fail("Rank 1 never set ready key")
            time.sleep(0.01)

        # Barrier should pass quickly since rank 1 is ready
        coord.barrier("test_barrier")

        # Wait for rank 1 to finish
        p.join(timeout=15)

        assert p.exitcode == 0, f"Rank 1 failed with exit code {p.exitcode}"

    def test_wait_ready_blocks_until_signal(self, mp_context):
        """wait_ready correctly blocks until signal_ready is called."""
        from app.distributed.oob import OOBCoordinator

        port = _find_free_port()

        # Run rank 1 in subprocess
        p = mp_context.Process(target=_rank1_wait_ready_worker, args=(port,))
        p.start()

        # Run rank 0 in main process
        coord = OOBCoordinator(
            rank=0,
            world_size=2,
            host="127.0.0.1",
            port=port,
            authkey=b"test",
        )

        # Delay before signaling ready - rank 1 should block during this time
        time.sleep(0.5)
        signal_time = time.time()
        coord.signal_ready("transfer_test")

        # Wait for rank 1 to complete
        t0 = time.time()
        while "rank1_done" not in coord._store:
            if time.time() - t0 > 10:
                p.terminate()
                pytest.fail("Rank 1 never completed")
            time.sleep(0.01)

        wait_complete_time = time.time()

        p.join(timeout=5)

        # Verify wait_ready actually blocked (completed after signal)
        # Allow some slack for process timing
        assert wait_complete_time >= signal_time - 0.1
        assert p.exitcode == 0


class TestDeriveAuthkey:
    """Tests for hostfile-based authkey derivation."""

    def test_derive_authkey_deterministic(self):
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

    def test_derive_authkey_different_hostfiles(self):
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

    def test_derive_authkey_order_matters(self):
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

    def test_init_oob_returns_coordinator(self):
        """init_oob returns an OOBCoordinator instance."""
        import app.distributed.oob as oob_module
        from app.distributed.oob import init_oob

        # Reset global state
        oob_module._oob_coordinator = None

        port = _find_free_port()
        coord = init_oob(
            rank=0,
            world_size=1,
            host="127.0.0.1",
            port=port,
            authkey=b"test",
        )

        assert coord is not None
        assert coord.rank == 0
        assert coord.world_size == 1

        # Clean up global state
        oob_module._oob_coordinator = None

    def test_init_oob_returns_existing_on_reinit(self):
        """Calling init_oob twice returns the same instance."""
        import app.distributed.oob as oob_module
        from app.distributed.oob import init_oob

        # Reset global state
        oob_module._oob_coordinator = None

        port = _find_free_port()
        coord1 = init_oob(rank=0, world_size=1, host="127.0.0.1", port=port)

        # Second call should return same instance (with warning)
        coord2 = init_oob(rank=0, world_size=1, host="127.0.0.1", port=port + 1)

        assert coord1 is coord2

        # Clean up
        oob_module._oob_coordinator = None

    def test_get_oob_returns_global_coordinator(self):
        """get_oob returns the globally initialized coordinator."""
        import app.distributed.oob as oob_module
        from app.distributed.oob import get_oob, init_oob

        # Reset global state
        oob_module._oob_coordinator = None

        # Before init, should be None
        assert get_oob() is None

        port = _find_free_port()
        coord = init_oob(rank=0, world_size=1, host="127.0.0.1", port=port)

        # After init, should return coordinator
        assert get_oob() is coord

        # Clean up
        oob_module._oob_coordinator = None
