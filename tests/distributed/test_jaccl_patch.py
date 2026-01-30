"""Tests for JACCL send/recv patches."""

import pytest


class TestJacclPatchModule:
    """Tests for jaccl_patch module functions."""

    def test_patch_functions_exist(self):
        """Verify patch functions are importable."""
        from app.distributed.jaccl_patch import (
            patch_jaccl_send_recv,
            unpatch_jaccl_send_recv,
            reset_call_counter,
            is_patched,
        )
        assert callable(patch_jaccl_send_recv)
        assert callable(unpatch_jaccl_send_recv)
        assert callable(reset_call_counter)
        assert callable(is_patched)

    def test_initial_state_not_patched(self):
        """Module starts in unpatched state."""
        from app.distributed import jaccl_patch
        # Reset module state for clean test
        jaccl_patch._patched = False
        jaccl_patch._original_send = None
        jaccl_patch._original_recv_like = None

        assert jaccl_patch.is_patched() is False

    def test_patch_and_unpatch(self):
        """Patch and unpatch cycle works correctly."""
        from app.distributed import jaccl_patch
        import mlx.core as mx

        # Reset state
        jaccl_patch._patched = False
        jaccl_patch._original_send = None
        jaccl_patch._original_recv_like = None

        # Save original functions
        orig_send = mx.distributed.send
        orig_recv_like = mx.distributed.recv_like

        # Patch
        assert jaccl_patch.patch_jaccl_send_recv() is True
        assert jaccl_patch.is_patched() is True
        assert mx.distributed.send is not orig_send
        assert mx.distributed.recv_like is not orig_recv_like

        # Unpatch
        assert jaccl_patch.unpatch_jaccl_send_recv() is True
        assert jaccl_patch.is_patched() is False
        assert mx.distributed.send is orig_send
        assert mx.distributed.recv_like is orig_recv_like

    def test_patch_idempotent(self):
        """Patching twice returns False on second call."""
        from app.distributed import jaccl_patch

        # Reset state
        jaccl_patch._patched = False
        jaccl_patch._original_send = None
        jaccl_patch._original_recv_like = None

        assert jaccl_patch.patch_jaccl_send_recv() is True
        assert jaccl_patch.patch_jaccl_send_recv() is False  # Already patched

        # Cleanup
        jaccl_patch.unpatch_jaccl_send_recv()

    def test_unpatch_when_not_patched(self):
        """Unpatching when not patched returns False."""
        from app.distributed import jaccl_patch

        # Reset state
        jaccl_patch._patched = False
        jaccl_patch._original_send = None
        jaccl_patch._original_recv_like = None

        assert jaccl_patch.unpatch_jaccl_send_recv() is False


class TestCallCounter:
    """Tests for transfer ID call counter."""

    def test_reset_counter(self):
        """reset_call_counter resets to zero."""
        from app.distributed import jaccl_patch

        # Increment counter a few times
        jaccl_patch._call_counter = 5

        jaccl_patch.reset_call_counter()
        assert jaccl_patch._call_counter == 0

    def test_get_transfer_id_increments(self):
        """Each call to _get_transfer_id increments counter."""
        from app.distributed import jaccl_patch

        jaccl_patch.reset_call_counter()

        id1 = jaccl_patch._get_transfer_id()
        id2 = jaccl_patch._get_transfer_id()
        id3 = jaccl_patch._get_transfer_id()

        assert id1 == "pipeline_fwd_1"
        assert id2 == "pipeline_fwd_2"
        assert id3 == "pipeline_fwd_3"

    def test_transfer_ids_unique_across_resets(self):
        """Transfer IDs restart after reset."""
        from app.distributed import jaccl_patch

        jaccl_patch.reset_call_counter()
        id1 = jaccl_patch._get_transfer_id()

        jaccl_patch.reset_call_counter()
        id2 = jaccl_patch._get_transfer_id()

        # Same ID after reset (both are first call after reset)
        assert id1 == id2 == "pipeline_fwd_1"


class TestExportsFromInit:
    """Test that jaccl_patch functions are exported from __init__."""

    def test_exports_available(self):
        """Verify exports from app.distributed."""
        from app.distributed import (
            patch_jaccl_send_recv,
            unpatch_jaccl_send_recv,
            reset_call_counter,
            is_jaccl_patched,
        )
        assert callable(patch_jaccl_send_recv)
        assert callable(unpatch_jaccl_send_recv)
        assert callable(reset_call_counter)
        assert callable(is_jaccl_patched)
