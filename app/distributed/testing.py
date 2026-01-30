"""Mock infrastructure for unit testing distributed code without RDMA hardware.

This module provides async mock implementations of OOBCoordinator and
mx.distributed.Group that enable comprehensive unit testing of distributed
code paths in a single process without requiring actual RDMA hardware or
multi-process setups.

Usage
-----
Testing synced_all_sum with mock OOB:
    from app.distributed.testing import MockOOBCoordinator, MockDistributedGroup
    from app.distributed.helpers import synced_all_sum

    oob = MockOOBCoordinator(rank=0, world_size=2)
    group = MockDistributedGroup(rank=0, size=2)

    with patch("mlx.core.distributed.all_sum", mock_all_sum):
        result = await synced_all_sum(mx.array([1.0]), group, "test", oob)

    oob.assert_barrier_entered("test")

Testing termination behavior:
    oob = MockOOBCoordinator(rank=0, world_size=2)
    oob.simulate_peer_termination()

    assert oob.is_any_peer_terminating()

Testing abort propagation:
    oob = MockOOBCoordinator(rank=0, world_size=2)
    await oob.abort()

    with pytest.raises(AbortError):
        await oob.barrier("should_fail")
"""

from typing import Optional, Set

from .oob import AbortError, PeerTerminatedError, PeerTimeoutError


class MockDistributedGroup:
    """Mock implementation of mx.distributed.Group for unit tests.

    This class mimics the interface of mx.distributed.Group, allowing tests
    to verify rank and size handling without actual distributed setup.

    Attributes:
        _rank: The simulated rank of this process
        _size: The simulated world size

    Example:
        group = MockDistributedGroup(rank=0, size=4)
        assert group.rank() == 0
        assert group.size() == 4
    """

    def __init__(self, rank: int = 0, size: int = 1) -> None:
        """Initialize the mock group.

        Args:
            rank: The simulated rank (default: 0)
            size: The simulated world size (default: 1)
        """
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        """Return the simulated rank."""
        return self._rank

    def size(self) -> int:
        """Return the simulated world size."""
        return self._size


class MockOOBCoordinator:
    """Async mock implementation of OOBCoordinator for single-process unit tests.

    This mock tracks all coordination calls (barriers, ready signals, etc.)
    and allows tests to verify correct usage patterns without actual network
    coordination. All methods are async to match the real OOBCoordinator interface.

    Attributes:
        rank: Simulated rank
        world_size: Simulated world size
        barriers_entered: Set of barrier names that were entered
        ready_signals: Dict mapping transfer_id to rank that signaled
        complete_signals: Dict mapping transfer_id to rank that signaled
        _peer_terminating: Flag to simulate peer termination
        _abort_flag: Flag to simulate abort signal

    Example:
        oob = MockOOBCoordinator(rank=0, world_size=2)

        # Use in code being tested
        await oob.barrier("warmup")
        await oob.signal_ready("transfer_1")

        # Verify in tests
        oob.assert_barrier_entered("warmup")
        oob.assert_ready_signaled("transfer_1")
    """

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        timeout_sec: float = 300.0,
    ) -> None:
        """Initialize the mock coordinator.

        Args:
            rank: Simulated rank (default: 0)
            world_size: Simulated world size (default: 1)
            timeout_sec: Simulated timeout (default: 300.0)
        """
        self.rank = rank
        self.world_size = world_size
        self._timeout_sec = timeout_sec
        self.host = "127.0.0.1"
        self.port = 29400

        # Tracking sets/dicts for assertions
        self.barriers_entered: Set[str] = set()
        self.ready_signals: dict[str, int] = {}
        self.complete_signals: dict[str, int] = {}

        # Termination/abort simulation
        self._peer_terminating = False
        self._self_terminating = False
        self._abort_flag = False

        # Store simulation
        self._store: dict[str, str] = {}

    # --- Async Context Manager ---

    async def __aenter__(self) -> "MockOOBCoordinator":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Async context manager exit."""
        await self.stop()

    async def start(self) -> None:
        """Simulate starting the coordinator (no-op in mock)."""
        pass

    async def stop(self) -> None:
        """Simulate stopping the coordinator (no-op in mock)."""
        pass

    # --- Barrier ---

    async def barrier(self, name: str, timeout: Optional[float] = None) -> None:
        """Record a barrier call.

        Args:
            name: Barrier name for tracking
            timeout: Ignored in mock (for API compatibility)

        Raises:
            AbortError: If abort is simulated
            PeerTerminatedError: If peer termination is simulated
        """
        self._check_abort_or_termination(f"barrier {name}")
        self.barriers_entered.add(name)

    # --- Store Operations ---

    async def signal_ready(self, transfer_id: str) -> None:
        """Record a ready signal.

        Args:
            transfer_id: The transfer identifier
        """
        self.ready_signals[transfer_id] = self.rank
        self._store[f"ready_{transfer_id}_rank{self.rank}"] = "1"

    async def wait_ready(
        self,
        transfer_id: str,
        receiver_rank: int,
        timeout: Optional[float] = None,
    ) -> None:
        """Simulate waiting for ready signal.

        In mock mode, this succeeds immediately if the ready signal exists,
        or raises appropriate exception if abort/termination is simulated.

        Args:
            transfer_id: The transfer identifier
            receiver_rank: The rank to wait for
            timeout: Used for timeout simulation if configured

        Raises:
            AbortError: If abort is simulated
            PeerTerminatedError: If peer termination is simulated
            PeerTimeoutError: If timeout simulation is configured
        """
        self._check_abort_or_termination(f"wait_ready {transfer_id}")
        # In mock, we don't actually wait - just check state

    async def signal_complete(self, transfer_id: str) -> None:
        """Record a complete signal.

        Args:
            transfer_id: The transfer identifier
        """
        self.complete_signals[transfer_id] = self.rank
        self._store[f"complete_{transfer_id}_rank{self.rank}"] = "1"

    async def wait_complete(
        self,
        transfer_id: str,
        sender_rank: int,
        timeout: Optional[float] = None,
    ) -> None:
        """Simulate waiting for complete signal.

        Args:
            transfer_id: The transfer identifier
            sender_rank: The rank to wait for
            timeout: Used for timeout simulation if configured

        Raises:
            AbortError: If abort is simulated
            PeerTerminatedError: If peer termination is simulated
        """
        self._check_abort_or_termination(f"wait_complete {transfer_id}")

    # --- Termination/Abort ---

    async def signal_terminating(self) -> None:
        """Record that this rank is terminating."""
        self._self_terminating = True

    def is_any_peer_terminating(self) -> bool:
        """Check if simulated peer termination is active.

        Returns:
            True if simulate_peer_termination() was called
        """
        return self._peer_terminating

    def check_peers_alive(self) -> bool:
        """Simulate peer health check.

        Returns:
            True unless peer termination is simulated
        """
        return not self._peer_terminating and not self._abort_flag

    async def abort(self) -> None:
        """Simulate broadcast abort."""
        self._abort_flag = True

    def _check_abort_or_termination(self, operation: str) -> None:
        """Check for abort/termination and raise appropriate exception."""
        if self._abort_flag:
            raise AbortError(f"Abort during {operation}")
        if self._peer_terminating:
            raise PeerTerminatedError(f"Peer terminated during {operation}")

    # --- Test Helper Methods ---

    def simulate_peer_termination(self) -> None:
        """Simulate a peer signaling termination.

        After calling this, is_any_peer_terminating() will return True
        and coordination operations will raise PeerTerminatedError.
        """
        self._peer_terminating = True

    def reset_peer_termination(self) -> None:
        """Reset the peer termination flag."""
        self._peer_terminating = False

    def simulate_abort(self) -> None:
        """Simulate receiving an abort signal.

        After calling this, coordination operations will raise AbortError.
        """
        self._abort_flag = True

    def reset_abort(self) -> None:
        """Reset the abort flag."""
        self._abort_flag = False

    def assert_barrier_entered(self, name: str) -> None:
        """Assert that a specific barrier was entered.

        Args:
            name: The barrier name to check

        Raises:
            AssertionError: If the barrier was not entered
        """
        assert name in self.barriers_entered, (
            f"Barrier '{name}' was not entered. "
            f"Entered barriers: {self.barriers_entered}"
        )

    def assert_barrier_not_entered(self, name: str) -> None:
        """Assert that a specific barrier was NOT entered.

        Args:
            name: The barrier name to check

        Raises:
            AssertionError: If the barrier was entered
        """
        assert name not in self.barriers_entered, (
            f"Barrier '{name}' was unexpectedly entered. "
            f"Entered barriers: {self.barriers_entered}"
        )

    def assert_ready_signaled(self, transfer_id: str) -> None:
        """Assert that a ready signal was sent for a transfer.

        Args:
            transfer_id: The transfer identifier to check

        Raises:
            AssertionError: If ready was not signaled
        """
        assert transfer_id in self.ready_signals, (
            f"Ready signal for '{transfer_id}' was not sent. "
            f"Signaled transfers: {list(self.ready_signals.keys())}"
        )

    def assert_complete_signaled(self, transfer_id: str) -> None:
        """Assert that a complete signal was sent for a transfer.

        Args:
            transfer_id: The transfer identifier to check

        Raises:
            AssertionError: If complete was not signaled
        """
        assert transfer_id in self.complete_signals, (
            f"Complete signal for '{transfer_id}' was not sent. "
            f"Signaled transfers: {list(self.complete_signals.keys())}"
        )

    def clear_tracking(self) -> None:
        """Clear all tracking data for a fresh test."""
        self.barriers_entered.clear()
        self.ready_signals.clear()
        self.complete_signals.clear()
        self._store.clear()
        self._peer_terminating = False
        self._self_terminating = False
        self._abort_flag = False


def mock_all_sum(data, group=None, stream=None):
    """Mock implementation of mx.distributed.all_sum.

    Returns the input unchanged, simulating a single-rank all_sum where
    the result equals the input.

    Args:
        data: The input array
        group: Ignored (mock doesn't use distributed groups)
        stream: Ignored (mock doesn't use streams)

    Returns:
        The input data unchanged

    Example:
        with patch("mlx.core.distributed.all_sum", mock_all_sum):
            result = await synced_all_sum(mx.array([1.0, 2.0]), group, "test")
            assert mx.array_equal(result, mx.array([1.0, 2.0]))
    """
    return data


def mock_eval(*args):
    """Mock implementation of mx.eval.

    No-op that accepts any arguments, for use in tests that don't need
    actual evaluation.

    Args:
        *args: Ignored
    """
    pass
