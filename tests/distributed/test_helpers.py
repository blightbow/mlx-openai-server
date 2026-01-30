"""Unit tests for distributed synchronization helpers.

Tests the async helper functions that encapsulate OOB barrier + all_sum patterns
for JACCL collective operations. Uses mock infrastructure to test without
RDMA hardware.
"""

from unittest.mock import MagicMock, patch

import pytest

import mlx.core as mx

from app.distributed.helpers import (
    AbortError,
    PeerTerminatedError,
    PeerTimeoutError,
    broadcast_value,
    broadcast_value_sync,
    safe_collective,
    synced_all_sum,
    synced_all_sum_sync,
)
from app.distributed.testing import (
    MockDistributedGroup,
    MockOOBCoordinator,
    mock_all_sum,
)


class TestMockDistributedGroup:
    """Tests for MockDistributedGroup helper class."""

    def test_default_values(self) -> None:
        """Default group is rank 0, size 1."""
        group = MockDistributedGroup()
        assert group.rank() == 0
        assert group.size() == 1

    def test_custom_rank_size(self) -> None:
        """Can specify custom rank and size."""
        group = MockDistributedGroup(rank=2, size=4)
        assert group.rank() == 2
        assert group.size() == 4


class TestMockOOBCoordinator:
    """Tests for MockOOBCoordinator helper class."""

    def test_default_values(self) -> None:
        """Default coordinator is rank 0, world_size 1."""
        oob = MockOOBCoordinator()
        assert oob.rank == 0
        assert oob.world_size == 1

    def test_custom_rank_world_size(self) -> None:
        """Can specify custom rank and world_size."""
        oob = MockOOBCoordinator(rank=1, world_size=4)
        assert oob.rank == 1
        assert oob.world_size == 4

    async def test_barrier_tracking(self) -> None:
        """Barrier calls are tracked by name."""
        oob = MockOOBCoordinator()

        await oob.barrier("warmup")
        await oob.barrier("sync_tokens")

        assert "warmup" in oob.barriers_entered
        assert "sync_tokens" in oob.barriers_entered
        assert "nonexistent" not in oob.barriers_entered

    async def test_signal_ready_tracking(self) -> None:
        """Ready signals are tracked with transfer ID and rank."""
        oob = MockOOBCoordinator(rank=1, world_size=2)

        await oob.signal_ready("transfer_1")

        assert "transfer_1" in oob.ready_signals
        assert oob.ready_signals["transfer_1"] == 1

    async def test_signal_complete_tracking(self) -> None:
        """Complete signals are tracked with transfer ID and rank."""
        oob = MockOOBCoordinator(rank=0, world_size=2)

        await oob.signal_complete("transfer_2")

        assert "transfer_2" in oob.complete_signals
        assert oob.complete_signals["transfer_2"] == 0

    def test_peer_termination_default_false(self) -> None:
        """Peer termination is False by default."""
        oob = MockOOBCoordinator()
        assert not oob.is_any_peer_terminating()

    def test_simulate_peer_termination(self) -> None:
        """Can simulate peer termination."""
        oob = MockOOBCoordinator()

        oob.simulate_peer_termination()

        assert oob.is_any_peer_terminating()
        assert not oob.check_peers_alive()

    def test_reset_peer_termination(self) -> None:
        """Can reset peer termination flag."""
        oob = MockOOBCoordinator()
        oob.simulate_peer_termination()

        oob.reset_peer_termination()

        assert not oob.is_any_peer_terminating()

    async def test_assert_barrier_entered_passes(self) -> None:
        """assert_barrier_entered passes when barrier was entered."""
        oob = MockOOBCoordinator()
        await oob.barrier("test_barrier")

        # Should not raise
        oob.assert_barrier_entered("test_barrier")

    def test_assert_barrier_entered_fails(self) -> None:
        """assert_barrier_entered fails when barrier was not entered."""
        oob = MockOOBCoordinator()

        with pytest.raises(AssertionError, match="was not entered"):
            oob.assert_barrier_entered("missing_barrier")

    def test_assert_barrier_not_entered_passes(self) -> None:
        """assert_barrier_not_entered passes when barrier was not entered."""
        oob = MockOOBCoordinator()

        # Should not raise
        oob.assert_barrier_not_entered("some_barrier")

    async def test_assert_barrier_not_entered_fails(self) -> None:
        """assert_barrier_not_entered fails when barrier was entered."""
        oob = MockOOBCoordinator()
        await oob.barrier("entered_barrier")

        with pytest.raises(AssertionError, match="unexpectedly entered"):
            oob.assert_barrier_not_entered("entered_barrier")

    async def test_assert_ready_signaled_passes(self) -> None:
        """assert_ready_signaled passes when signal was sent."""
        oob = MockOOBCoordinator()
        await oob.signal_ready("transfer_x")

        # Should not raise
        oob.assert_ready_signaled("transfer_x")

    def test_assert_ready_signaled_fails(self) -> None:
        """assert_ready_signaled fails when signal was not sent."""
        oob = MockOOBCoordinator()

        with pytest.raises(AssertionError, match="was not sent"):
            oob.assert_ready_signaled("missing_transfer")

    async def test_clear_tracking(self) -> None:
        """clear_tracking resets all tracking data."""
        oob = MockOOBCoordinator()
        await oob.barrier("barrier1")
        await oob.signal_ready("transfer1")
        await oob.signal_complete("transfer2")
        oob.simulate_peer_termination()

        oob.clear_tracking()

        assert len(oob.barriers_entered) == 0
        assert len(oob.ready_signals) == 0
        assert len(oob.complete_signals) == 0
        assert not oob.is_any_peer_terminating()


class TestMockAllSum:
    """Tests for mock_all_sum function."""

    def test_returns_input_unchanged(self) -> None:
        """mock_all_sum returns input data unchanged."""
        data = mx.array([1.0, 2.0, 3.0])
        result = mock_all_sum(data)
        assert mx.array_equal(result, data)

    def test_ignores_group_and_stream(self) -> None:
        """mock_all_sum ignores group and stream parameters."""
        data = mx.array([42])
        group = MockDistributedGroup()
        result = mock_all_sum(data, group=group, stream=mx.cpu)
        assert mx.array_equal(result, data)


class TestSyncedAllSum:
    """Tests for async synced_all_sum helper function."""

    async def test_no_oob_skips_barrier(self) -> None:
        """When oob is None, no barrier is called."""
        group = MockDistributedGroup()
        data = mx.array([1.0, 2.0])

        with patch("mlx.core.distributed.all_sum", mock_all_sum):
            result = await synced_all_sum(data, group, "test_barrier", oob=None)

        assert mx.array_equal(result, data)

    async def test_with_oob_calls_barrier(self) -> None:
        """When oob is provided, barrier is called with correct name."""
        oob = MockOOBCoordinator()
        group = MockDistributedGroup()
        data = mx.array([1.0, 2.0])

        with patch("mlx.core.distributed.all_sum", mock_all_sum):
            result = await synced_all_sum(data, group, "warmup_barrier", oob=oob)

        oob.assert_barrier_entered("warmup_barrier")
        assert mx.array_equal(result, data)

    async def test_calls_mx_eval(self) -> None:
        """synced_all_sum calls mx.eval on the result."""
        oob = MockOOBCoordinator()
        group = MockDistributedGroup()
        data = mx.array([1.0])

        with patch("mlx.core.distributed.all_sum", mock_all_sum):
            with patch("mlx.core.eval") as mock_eval:
                await synced_all_sum(data, group, "test", oob=oob)
                mock_eval.assert_called_once()

    async def test_passes_group_to_all_sum(self) -> None:
        """synced_all_sum passes the group to mx.distributed.all_sum."""
        oob = MockOOBCoordinator()
        group = MockDistributedGroup(rank=1, size=4)
        data = mx.array([1.0])

        mock_fn = MagicMock(return_value=data)
        with patch("mlx.core.distributed.all_sum", mock_fn):
            await synced_all_sum(data, group, "test", oob=oob)

        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["group"] is group


class TestBroadcastValue:
    """Tests for async broadcast_value helper function."""

    async def test_source_rank_contributes_value(self) -> None:
        """Source rank contributes its value."""
        oob = MockOOBCoordinator(rank=0, world_size=2)
        group = MockDistributedGroup(rank=0, size=2)
        value = mx.array([42.0])

        with patch("mlx.core.distributed.all_sum", mock_all_sum):
            result = await broadcast_value(value, rank=0, group=group, barrier_name="test", oob=oob)

        # With mock_all_sum, result is input unchanged
        assert mx.array_equal(result, value)
        oob.assert_barrier_entered("test")

    async def test_non_source_rank_contributes_zeros(self) -> None:
        """Non-source ranks contribute zeros."""
        oob = MockOOBCoordinator(rank=1, world_size=2)
        group = MockDistributedGroup(rank=1, size=2)
        value = mx.array([42.0])

        all_sum_calls = []

        def tracking_all_sum(data, group=None):
            all_sum_calls.append(data)
            return data

        with patch("mlx.core.distributed.all_sum", tracking_all_sum):
            await broadcast_value(value, rank=1, group=group, barrier_name="test", oob=oob)

        # Non-source rank should contribute zeros
        assert len(all_sum_calls) == 1
        assert mx.array_equal(all_sum_calls[0], mx.zeros_like(value))

    async def test_custom_source_rank(self) -> None:
        """Can specify a custom source rank."""
        oob = MockOOBCoordinator(rank=2, world_size=4)
        group = MockDistributedGroup(rank=2, size=4)
        value = mx.array([100.0])

        all_sum_calls = []

        def tracking_all_sum(data, group=None):
            all_sum_calls.append(data)
            return data

        with patch("mlx.core.distributed.all_sum", tracking_all_sum):
            await broadcast_value(
                value, rank=2, group=group, barrier_name="test", oob=oob, source_rank=2
            )

        # Rank 2 is source, so should contribute value
        assert len(all_sum_calls) == 1
        assert mx.array_equal(all_sum_calls[0], value)


class TestSafeCollective:
    """Tests for async safe_collective helper function."""

    async def test_executes_collective_when_healthy(self) -> None:
        """Collective is executed when no termination."""
        oob = MockOOBCoordinator()
        result_value = mx.array([42.0])

        result = await safe_collective(lambda: result_value, "test", oob=oob)

        assert mx.array_equal(result, result_value)

    async def test_raises_on_termination_before(self) -> None:
        """Raises PeerTerminatedError when peer is terminating before collective."""
        oob = MockOOBCoordinator()
        oob.simulate_peer_termination()

        with pytest.raises(PeerTerminatedError):
            await safe_collective(lambda: mx.array([1.0]), "test", oob=oob)

    async def test_returns_none_with_raise_false(self) -> None:
        """Returns None when raise_on_termination is False."""
        oob = MockOOBCoordinator()
        oob.simulate_peer_termination()

        result = await safe_collective(
            lambda: mx.array([1.0]), "test", oob=oob, raise_on_termination=False
        )

        assert result is None

    async def test_no_oob_always_executes(self) -> None:
        """Without OOB, collective is always executed."""
        result_value = mx.array([123.0])

        result = await safe_collective(lambda: result_value, "test", oob=None)

        assert mx.array_equal(result, result_value)

    async def test_collective_called_once(self) -> None:
        """Collective function is called exactly once."""
        oob = MockOOBCoordinator()
        call_count = [0]

        def counting_collective():
            call_count[0] += 1
            return mx.array([1.0])

        await safe_collective(counting_collective, "test", oob=oob)

        assert call_count[0] == 1


class TestPeerTerminatedError:
    """Tests for PeerTerminatedError exception class."""

    def test_is_exception(self) -> None:
        """PeerTerminatedError is an Exception subclass."""
        assert issubclass(PeerTerminatedError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """PeerTerminatedError can be raised and caught."""
        with pytest.raises(PeerTerminatedError):
            raise PeerTerminatedError("test message")

    def test_message_preserved(self) -> None:
        """Exception message is preserved."""
        try:
            raise PeerTerminatedError("peer 2 died")
        except PeerTerminatedError as e:
            assert "peer 2 died" in str(e)


class TestPeerTimeoutError:
    """Tests for PeerTimeoutError exception class."""

    def test_is_exception(self) -> None:
        """PeerTimeoutError is an Exception subclass."""
        assert issubclass(PeerTimeoutError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """PeerTimeoutError can be raised and caught."""
        with pytest.raises(PeerTimeoutError):
            raise PeerTimeoutError("timeout waiting for peer")

    def test_message_preserved(self) -> None:
        """Exception message is preserved."""
        try:
            raise PeerTimeoutError("timeout after 30s")
        except PeerTimeoutError as e:
            assert "timeout after 30s" in str(e)

    def test_distinct_from_terminated(self) -> None:
        """PeerTimeoutError is distinct from PeerTerminatedError."""
        assert PeerTimeoutError is not PeerTerminatedError
        with pytest.raises(PeerTimeoutError):
            raise PeerTimeoutError("timeout")
        # Should not catch as PeerTerminatedError
        try:
            raise PeerTimeoutError("timeout")
        except PeerTerminatedError:
            pytest.fail("PeerTimeoutError should not be caught as PeerTerminatedError")
        except PeerTimeoutError:
            pass  # Expected


class TestAbortError:
    """Tests for AbortError exception class."""

    def test_is_exception(self) -> None:
        """AbortError is an Exception subclass."""
        assert issubclass(AbortError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """AbortError can be raised and caught."""
        with pytest.raises(AbortError):
            raise AbortError("abort signal received")

    def test_message_preserved(self) -> None:
        """Exception message is preserved."""
        try:
            raise AbortError("abort during barrier")
        except AbortError as e:
            assert "abort during barrier" in str(e)


class TestMockOOBCoordinatorAbortBehavior:
    """Tests for MockOOBCoordinator abort and termination behavior."""

    async def test_barrier_raises_on_peer_termination(self) -> None:
        """barrier() raises PeerTerminatedError when peer is terminating."""
        oob = MockOOBCoordinator()
        oob.simulate_peer_termination()

        with pytest.raises(PeerTerminatedError):
            await oob.barrier("test")

    async def test_barrier_raises_on_abort(self) -> None:
        """barrier() raises AbortError when abort signal received."""
        oob = MockOOBCoordinator()
        oob.simulate_abort()

        with pytest.raises(AbortError):
            await oob.barrier("test")

    async def test_wait_ready_raises_on_peer_termination(self) -> None:
        """wait_ready() raises PeerTerminatedError when peer is terminating."""
        oob = MockOOBCoordinator()
        oob.simulate_peer_termination()

        with pytest.raises(PeerTerminatedError):
            await oob.wait_ready("transfer_1", receiver_rank=1)

    async def test_wait_complete_raises_on_abort(self) -> None:
        """wait_complete() raises AbortError when abort signal received."""
        oob = MockOOBCoordinator()
        oob.simulate_abort()

        with pytest.raises(AbortError):
            await oob.wait_complete("transfer_1", sender_rank=0)


class TestIntegrationPatterns:
    """Integration tests demonstrating real usage patterns."""

    async def test_warmup_pattern(self) -> None:
        """Test the warmup synchronization pattern from main.py."""
        oob = MockOOBCoordinator(rank=0, world_size=2)
        group = MockDistributedGroup(rank=0, size=2)

        # Simulate warmup: all ranks contribute 1.0, expect sum of world_size
        warmup_input = mx.array([1.0])

        with patch("mlx.core.distributed.all_sum", mock_all_sum):
            result = await synced_all_sum(warmup_input, group, "warmup", oob=oob)

        # With mock, result equals input (real would be 2.0 for world_size=2)
        oob.assert_barrier_entered("warmup")
        assert result is not None

    async def test_file_count_broadcast_pattern(self) -> None:
        """Test the file count broadcast pattern from file_sync.py."""
        # Rank 0 broadcasts file count
        oob_rank0 = MockOOBCoordinator(rank=0, world_size=2)
        group_rank0 = MockDistributedGroup(rank=0, size=2)
        file_count = mx.array([5])

        with patch("mlx.core.distributed.all_sum", mock_all_sum):
            result = await broadcast_value(
                file_count, rank=0, group=group_rank0, barrier_name="file_count", oob=oob_rank0
            )

        oob_rank0.assert_barrier_entered("file_count")
        assert mx.array_equal(result, file_count)

    async def test_termination_safe_token_sync(self) -> None:
        """Test termination-safe token synchronization pattern."""
        oob = MockOOBCoordinator(rank=0, world_size=2)
        group = MockDistributedGroup(rank=0, size=2)

        with patch("mlx.core.distributed.all_sum", mock_all_sum):
            result = await synced_all_sum(
                mx.array([100]), group, "token_length", oob=oob
            )

        assert mx.array_equal(result, mx.array([100]))
        oob.assert_barrier_entered("token_length")

    async def test_termination_safe_early_exit(self) -> None:
        """Test that termination causes early exit without executing collective."""
        oob = MockOOBCoordinator(rank=0, world_size=2)
        oob.simulate_peer_termination()
        collective_called = [False]

        def should_not_run():
            collective_called[0] = True
            return mx.array([1.0])

        result = await safe_collective(
            should_not_run, "test", oob=oob, raise_on_termination=False
        )

        assert result is None
        assert not collective_called[0]  # Collective should not have been called
