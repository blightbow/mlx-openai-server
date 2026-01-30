"""Unit tests for distributed synchronization helpers.

Tests the helper functions that encapsulate OOB barrier + all_sum patterns
for JACCL collective operations. Uses mock infrastructure to test without
RDMA hardware.
"""

from unittest.mock import MagicMock, patch

import pytest

import mlx.core as mx

from app.distributed.helpers import (
    PeerTerminatedError,
    broadcast_value,
    safe_collective,
    synced_all_sum,
)
from app.distributed.testing import (
    MockDistributedGroup,
    MockOOBCoordinator,
    mock_all_sum,
)


class TestMockDistributedGroup:
    """Tests for MockDistributedGroup helper class."""

    def test_default_values(self):
        """Default group is rank 0, size 1."""
        group = MockDistributedGroup()
        assert group.rank() == 0
        assert group.size() == 1

    def test_custom_rank_size(self):
        """Can specify custom rank and size."""
        group = MockDistributedGroup(rank=2, size=4)
        assert group.rank() == 2
        assert group.size() == 4


class TestMockOOBCoordinator:
    """Tests for MockOOBCoordinator helper class."""

    def test_default_values(self):
        """Default coordinator is rank 0, world_size 1."""
        oob = MockOOBCoordinator()
        assert oob.rank == 0
        assert oob.world_size == 1

    def test_custom_rank_world_size(self):
        """Can specify custom rank and world_size."""
        oob = MockOOBCoordinator(rank=1, world_size=4)
        assert oob.rank == 1
        assert oob.world_size == 4

    def test_barrier_tracking(self):
        """Barrier calls are tracked by name."""
        oob = MockOOBCoordinator()

        oob.barrier("warmup")
        oob.barrier("sync_tokens")

        assert "warmup" in oob.barriers_entered
        assert "sync_tokens" in oob.barriers_entered
        assert "nonexistent" not in oob.barriers_entered

    def test_barrier_none_name_not_tracked(self):
        """Barrier with None name is not tracked."""
        oob = MockOOBCoordinator()
        oob.barrier(None)
        assert len(oob.barriers_entered) == 0

    def test_signal_ready_tracking(self):
        """Ready signals are tracked with transfer ID and rank."""
        oob = MockOOBCoordinator(rank=1, world_size=2)

        oob.signal_ready("transfer_1")

        assert "transfer_1" in oob.ready_signals
        assert oob.ready_signals["transfer_1"] == 1

    def test_signal_complete_tracking(self):
        """Complete signals are tracked with transfer ID and rank."""
        oob = MockOOBCoordinator(rank=0, world_size=2)

        oob.signal_complete("transfer_2")

        assert "transfer_2" in oob.complete_signals
        assert oob.complete_signals["transfer_2"] == 0

    def test_peer_termination_default_false(self):
        """Peer termination is False by default."""
        oob = MockOOBCoordinator()
        assert not oob.is_any_peer_terminating()

    def test_simulate_peer_termination(self):
        """Can simulate peer termination."""
        oob = MockOOBCoordinator()

        oob.simulate_peer_termination()

        assert oob.is_any_peer_terminating()
        assert not oob.check_peers_alive()

    def test_reset_peer_termination(self):
        """Can reset peer termination flag."""
        oob = MockOOBCoordinator()
        oob.simulate_peer_termination()

        oob.reset_peer_termination()

        assert not oob.is_any_peer_terminating()

    def test_assert_barrier_entered_passes(self):
        """assert_barrier_entered passes when barrier was entered."""
        oob = MockOOBCoordinator()
        oob.barrier("test_barrier")

        # Should not raise
        oob.assert_barrier_entered("test_barrier")

    def test_assert_barrier_entered_fails(self):
        """assert_barrier_entered fails when barrier was not entered."""
        oob = MockOOBCoordinator()

        with pytest.raises(AssertionError, match="was not entered"):
            oob.assert_barrier_entered("missing_barrier")

    def test_assert_barrier_not_entered_passes(self):
        """assert_barrier_not_entered passes when barrier was not entered."""
        oob = MockOOBCoordinator()

        # Should not raise
        oob.assert_barrier_not_entered("some_barrier")

    def test_assert_barrier_not_entered_fails(self):
        """assert_barrier_not_entered fails when barrier was entered."""
        oob = MockOOBCoordinator()
        oob.barrier("entered_barrier")

        with pytest.raises(AssertionError, match="unexpectedly entered"):
            oob.assert_barrier_not_entered("entered_barrier")

    def test_assert_ready_signaled_passes(self):
        """assert_ready_signaled passes when signal was sent."""
        oob = MockOOBCoordinator()
        oob.signal_ready("transfer_x")

        # Should not raise
        oob.assert_ready_signaled("transfer_x")

    def test_assert_ready_signaled_fails(self):
        """assert_ready_signaled fails when signal was not sent."""
        oob = MockOOBCoordinator()

        with pytest.raises(AssertionError, match="was not sent"):
            oob.assert_ready_signaled("missing_transfer")

    def test_clear_tracking(self):
        """clear_tracking resets all tracking data."""
        oob = MockOOBCoordinator()
        oob.barrier("barrier1")
        oob.signal_ready("transfer1")
        oob.signal_complete("transfer2")
        oob.simulate_peer_termination()

        oob.clear_tracking()

        assert len(oob.barriers_entered) == 0
        assert len(oob.ready_signals) == 0
        assert len(oob.complete_signals) == 0
        assert not oob.is_any_peer_terminating()


class TestMockAllSum:
    """Tests for mock_all_sum function."""

    def test_returns_input_unchanged(self):
        """mock_all_sum returns input data unchanged."""
        data = mx.array([1.0, 2.0, 3.0])
        result = mock_all_sum(data)
        assert mx.array_equal(result, data)

    def test_ignores_group_and_stream(self):
        """mock_all_sum ignores group and stream parameters."""
        data = mx.array([42])
        group = MockDistributedGroup()
        result = mock_all_sum(data, group=group, stream=mx.cpu)
        assert mx.array_equal(result, data)


class TestSyncedAllSum:
    """Tests for synced_all_sum helper function."""

    def test_no_oob_skips_barrier(self):
        """When oob is None, no barrier is called."""
        group = MockDistributedGroup()
        data = mx.array([1.0, 2.0])

        with patch("mlx.core.distributed.all_sum", mock_all_sum):
            result = synced_all_sum(data, group, "test_barrier", oob=None)

        assert mx.array_equal(result, data)

    def test_with_oob_calls_barrier(self):
        """When oob is provided, barrier is called with correct name."""
        oob = MockOOBCoordinator()
        group = MockDistributedGroup()
        data = mx.array([1.0, 2.0])

        with patch("mlx.core.distributed.all_sum", mock_all_sum):
            result = synced_all_sum(data, group, "warmup_barrier", oob=oob)

        oob.assert_barrier_entered("warmup_barrier")
        assert mx.array_equal(result, data)

    def test_calls_mx_eval(self):
        """synced_all_sum calls mx.eval on the result."""
        oob = MockOOBCoordinator()
        group = MockDistributedGroup()
        data = mx.array([1.0])

        with patch("mlx.core.distributed.all_sum", mock_all_sum):
            with patch("mlx.core.eval") as mock_eval:
                synced_all_sum(data, group, "test", oob=oob)
                mock_eval.assert_called_once()

    def test_passes_group_to_all_sum(self):
        """synced_all_sum passes the group to mx.distributed.all_sum."""
        oob = MockOOBCoordinator()
        group = MockDistributedGroup(rank=1, size=4)
        data = mx.array([1.0])

        mock_fn = MagicMock(return_value=data)
        with patch("mlx.core.distributed.all_sum", mock_fn):
            synced_all_sum(data, group, "test", oob=oob)

        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["group"] is group


class TestBroadcastValue:
    """Tests for broadcast_value helper function."""

    def test_source_rank_contributes_value(self):
        """Source rank contributes its value, result equals value."""
        oob = MockOOBCoordinator(rank=0, world_size=2)
        group = MockDistributedGroup(rank=0, size=2)
        value = mx.array([42.0])

        with patch("mlx.core.distributed.all_sum", mock_all_sum):
            result = broadcast_value(
                value, rank=0, group=group, barrier_name="bcast", oob=oob
            )

        assert mx.array_equal(result, value)
        oob.assert_barrier_entered("bcast")

    def test_non_source_rank_contributes_zeros(self):
        """Non-source rank contributes zeros."""
        oob = MockOOBCoordinator(rank=1, world_size=2)
        group = MockDistributedGroup(rank=1, size=2)
        value = mx.array([42.0])

        with patch("mlx.core.distributed.all_sum", mock_all_sum):
            result = broadcast_value(
                value, rank=1, group=group, barrier_name="bcast", oob=oob
            )

        # Non-source contributes zeros, so with mock_all_sum, result is zeros
        assert mx.array_equal(result, mx.zeros_like(value))

    def test_custom_source_rank(self):
        """Can specify a non-zero source rank."""
        oob = MockOOBCoordinator(rank=2, world_size=4)
        group = MockDistributedGroup(rank=2, size=4)
        value = mx.array([99.0])

        with patch("mlx.core.distributed.all_sum", mock_all_sum):
            result = broadcast_value(
                value,
                rank=2,
                group=group,
                barrier_name="bcast",
                oob=oob,
                source_rank=2,
            )

        # Rank 2 is source, so it contributes value
        assert mx.array_equal(result, value)

    def test_no_oob_works(self):
        """broadcast_value works without OOB coordinator."""
        group = MockDistributedGroup(rank=0, size=1)
        value = mx.array([123.0])

        with patch("mlx.core.distributed.all_sum", mock_all_sum):
            result = broadcast_value(
                value, rank=0, group=group, barrier_name="bcast", oob=None
            )

        assert mx.array_equal(result, value)


class TestSafeCollective:
    """Tests for safe_collective helper function."""

    def test_executes_collective_when_no_termination(self):
        """Collective executes and returns result when no peer is terminating."""
        oob = MockOOBCoordinator()
        expected = mx.array([1.0, 2.0, 3.0])

        result = safe_collective(
            lambda: expected, "test_collective", oob=oob
        )

        assert mx.array_equal(result, expected)

    def test_raises_on_termination_before_collective(self):
        """Raises PeerTerminatedError if peer terminating before collective."""
        oob = MockOOBCoordinator()
        oob.simulate_peer_termination()

        with pytest.raises(PeerTerminatedError, match="before collective"):
            safe_collective(
                lambda: mx.array([1.0]), "test_op", oob=oob
            )

    def test_raises_on_termination_after_collective(self):
        """Raises PeerTerminatedError if peer terminates after collective."""
        oob = MockOOBCoordinator()

        def collective_that_triggers_termination():
            # Simulate peer terminating during the collective
            oob.simulate_peer_termination()
            return mx.array([1.0])

        with pytest.raises(PeerTerminatedError, match="after collective"):
            safe_collective(
                collective_that_triggers_termination, "test_op", oob=oob
            )

    def test_returns_none_when_no_raise(self):
        """Returns None instead of raising when raise_on_termination=False."""
        oob = MockOOBCoordinator()
        oob.simulate_peer_termination()

        result = safe_collective(
            lambda: mx.array([1.0]),
            "test_op",
            oob=oob,
            raise_on_termination=False,
        )

        assert result is None

    def test_no_oob_skips_termination_check(self):
        """When oob is None, no termination check is performed."""
        # This should not raise even though we can't check termination
        result = safe_collective(
            lambda: mx.array([42.0]), "test_op", oob=None
        )

        assert mx.array_equal(result, mx.array([42.0]))

    def test_collective_fn_called_exactly_once(self):
        """The collective function is called exactly once."""
        oob = MockOOBCoordinator()
        call_count = [0]

        def counting_collective():
            call_count[0] += 1
            return mx.array([1.0])

        safe_collective(counting_collective, "test", oob=oob)

        assert call_count[0] == 1


class TestPeerTerminatedError:
    """Tests for PeerTerminatedError exception class."""

    def test_is_exception(self):
        """PeerTerminatedError is an Exception subclass."""
        assert issubclass(PeerTerminatedError, Exception)

    def test_can_be_raised_and_caught(self):
        """PeerTerminatedError can be raised and caught."""
        with pytest.raises(PeerTerminatedError):
            raise PeerTerminatedError("test message")

    def test_message_preserved(self):
        """Exception message is preserved."""
        try:
            raise PeerTerminatedError("peer 2 died")
        except PeerTerminatedError as e:
            assert "peer 2 died" in str(e)


class TestIntegrationPatterns:
    """Integration tests demonstrating real usage patterns."""

    def test_warmup_pattern(self):
        """Test the warmup synchronization pattern from main.py."""
        oob = MockOOBCoordinator(rank=0, world_size=2)
        group = MockDistributedGroup(rank=0, size=2)

        # Simulate warmup: all ranks contribute 1.0, expect sum of world_size
        warmup_input = mx.array([1.0])

        with patch("mlx.core.distributed.all_sum", mock_all_sum):
            result = synced_all_sum(warmup_input, group, "warmup", oob=oob)

        # With mock, result equals input (real would be 2.0 for world_size=2)
        oob.assert_barrier_entered("warmup")
        assert result is not None

    def test_file_count_broadcast_pattern(self):
        """Test the file count broadcast pattern from file_sync.py."""
        # Rank 0 broadcasts file count
        oob_rank0 = MockOOBCoordinator(rank=0, world_size=2)
        group_rank0 = MockDistributedGroup(rank=0, size=2)
        file_count = mx.array([5])

        with patch("mlx.core.distributed.all_sum", mock_all_sum):
            result = broadcast_value(
                file_count, rank=0, group=group_rank0, barrier_name="file_count", oob=oob_rank0
            )

        oob_rank0.assert_barrier_entered("file_count")
        assert mx.array_equal(result, file_count)

    def test_termination_safe_token_sync(self):
        """Test termination-safe token synchronization pattern."""
        oob = MockOOBCoordinator(rank=0, world_size=2)
        group = MockDistributedGroup(rank=0, size=2)

        def token_sync():
            with patch("mlx.core.distributed.all_sum", mock_all_sum):
                return synced_all_sum(
                    mx.array([100]),  # token length
                    group,
                    "token_length",
                    oob=oob,
                )

        result = safe_collective(token_sync, "token_sync", oob=oob)

        assert mx.array_equal(result, mx.array([100]))
        oob.assert_barrier_entered("token_length")

    def test_termination_safe_early_exit(self):
        """Test that termination causes early exit without executing collective."""
        oob = MockOOBCoordinator(rank=0, world_size=2)
        oob.simulate_peer_termination()
        collective_called = [False]

        def should_not_run():
            collective_called[0] = True
            return mx.array([1.0])

        result = safe_collective(
            should_not_run, "test", oob=oob, raise_on_termination=False
        )

        assert result is None
        assert not collective_called[0]  # Collective should not have been called
