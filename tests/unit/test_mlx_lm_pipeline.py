"""Unit tests for MLX_LM pipeline parallel mode.

These tests verify the distributed initialization and model loading
behavior in pipeline mode using mocks. They document the expected
correct behavior for pipeline parallel inference.

Note: Pipeline tests require mlx-lm >= 0.30.0 with sharded_load support.
Tests are skipped if sharded_load is not available.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Import the module to make its attributes available for patching
import app.models.mlx_lm as mlx_lm_module

# Check if sharded_load is available for pipeline tests
HAS_SHARDED_LOAD = mlx_lm_module.sharded_load is not None

requires_sharded_load = pytest.mark.skipif(
    not HAS_SHARDED_LOAD,
    reason="sharded_load requires mlx-lm >= 0.30.0",
)


class TestPipelineModeAvailability:
    """Tests for pipeline mode availability detection."""

    def test_sharded_load_availability_matches_module(self):
        """Verify our availability check matches the module state."""
        assert HAS_SHARDED_LOAD == (mlx_lm_module.sharded_load is not None)

    @pytest.mark.skipif(
        HAS_SHARDED_LOAD,
        reason="Test only runs when sharded_load is NOT available",
    )
    def test_pipeline_mode_raises_without_sharded_load(self, mocker):
        """Verify clear error when pipeline mode requested without sharded_load."""
        # Mock distributed init (would be called before the error)
        mock_group = MagicMock()
        mock_group.rank.return_value = 0
        mock_group.size.return_value = 1
        mocker.patch.object(
            mlx_lm_module.mx.distributed,
            "init",
            return_value=mock_group,
        )

        # Mock OutlinesTransformerTokenizer
        mocker.patch.object(
            mlx_lm_module,
            "OutlinesTransformerTokenizer",
            return_value=MagicMock(),
        )

        from app.models.mlx_lm import MLX_LM

        # ImportError is wrapped in ValueError by the generic exception handler
        with pytest.raises(ValueError, match="mlx-lm >= 0.30.0"):
            MLX_LM(model_path="test-model", pipeline=True)


@requires_sharded_load
class TestMLXLMPipelineMode:
    """Tests for pipeline parallel mode initialization.

    These tests require mlx-lm >= 0.30.0 with sharded_load support.
    """

    @pytest.mark.pipeline
    def test_pipeline_mode_initializes_distributed_group(self, mocker):
        """Verify mx.distributed.init() is called when pipeline=True.

        In pipeline mode, the distributed group must be initialized
        BEFORE the model is loaded (sharded_load requires the group).
        """
        # Mock the distributed init
        mock_group = MagicMock()
        mock_group.rank.return_value = 0
        mock_group.size.return_value = 4
        mock_init = mocker.patch.object(
            mlx_lm_module.mx.distributed,
            "init",
            return_value=mock_group,
        )

        # Mock sharded_load
        mock_model = MagicMock()
        mock_model.model_type = "llama"
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.bos_token = "<s>"
        mock_tokenizer.chat_template = None
        mocker.patch.object(
            mlx_lm_module,
            "sharded_load",
            return_value=(mock_model, mock_tokenizer),
        )

        # Mock OutlinesTransformerTokenizer
        mocker.patch.object(
            mlx_lm_module,
            "OutlinesTransformerTokenizer",
            return_value=MagicMock(),
        )

        from app.models.mlx_lm import MLX_LM

        # This should not raise - distributed.init() should be called first
        model = MLX_LM(
            model_path="test-model",
            pipeline=True,
        )

        mock_init.assert_called_once()
        assert model.pipeline is True
        assert model.rank == 0

    @pytest.mark.pipeline
    def test_pipeline_mode_uses_sharded_load(self, mocker):
        """Verify sharded_load is called instead of load when pipeline=True."""
        # Mock distributed
        mock_group = MagicMock()
        mock_group.rank.return_value = 0
        mock_group.size.return_value = 4
        mocker.patch.object(
            mlx_lm_module.mx.distributed,
            "init",
            return_value=mock_group,
        )

        # Mock both load functions
        mock_model = MagicMock()
        mock_model.model_type = "llama"
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.bos_token = "<s>"
        mock_tokenizer.chat_template = None

        mock_sharded_load = mocker.patch.object(
            mlx_lm_module,
            "sharded_load",
            return_value=(mock_model, mock_tokenizer),
        )
        mock_load = mocker.patch.object(
            mlx_lm_module,
            "load",
            return_value=(mock_model, mock_tokenizer),
        )

        # Mock OutlinesTransformerTokenizer
        mocker.patch.object(
            mlx_lm_module,
            "OutlinesTransformerTokenizer",
            return_value=MagicMock(),
        )

        from app.models.mlx_lm import MLX_LM

        MLX_LM(model_path="test-model", pipeline=True)

        mock_sharded_load.assert_called_once()
        mock_load.assert_not_called()

    @pytest.mark.pipeline
    def test_non_pipeline_mode_uses_regular_load(self, mocker):
        """Verify load() is called when pipeline=False."""
        # Mock both load functions
        mock_model = MagicMock()
        mock_model.model_type = "llama"
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.bos_token = "<s>"
        mock_tokenizer.chat_template = None

        mock_sharded_load = mocker.patch.object(
            mlx_lm_module,
            "sharded_load",
            return_value=(mock_model, mock_tokenizer),
        )
        mock_load = mocker.patch.object(
            mlx_lm_module,
            "load",
            return_value=(mock_model, mock_tokenizer),
        )

        # Mock OutlinesTransformerTokenizer
        mocker.patch.object(
            mlx_lm_module,
            "OutlinesTransformerTokenizer",
            return_value=MagicMock(),
        )

        # Mock distributed init (should NOT be called in non-pipeline mode)
        mock_init = mocker.patch.object(
            mlx_lm_module.mx.distributed,
            "init",
        )

        from app.models.mlx_lm import MLX_LM

        model = MLX_LM(model_path="test-model", pipeline=False)

        mock_load.assert_called_once()
        mock_sharded_load.assert_not_called()
        mock_init.assert_not_called()
        assert model.pipeline is False

    @pytest.mark.pipeline
    def test_pipeline_stores_group_and_rank(self, mocker):
        """Verify group and rank are stored on the model instance."""
        mock_group = MagicMock()
        mock_group.rank.return_value = 2
        mock_group.size.return_value = 4
        mocker.patch.object(
            mlx_lm_module.mx.distributed,
            "init",
            return_value=mock_group,
        )

        mock_model = MagicMock()
        mock_model.model_type = "llama"
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.bos_token = "<s>"
        mock_tokenizer.chat_template = None
        mocker.patch.object(
            mlx_lm_module,
            "sharded_load",
            return_value=(mock_model, mock_tokenizer),
        )
        mocker.patch.object(
            mlx_lm_module,
            "OutlinesTransformerTokenizer",
            return_value=MagicMock(),
        )

        from app.models.mlx_lm import MLX_LM

        model = MLX_LM(model_path="test-model", pipeline=True)

        assert model.group == mock_group
        assert model.rank == 2

    @pytest.mark.pipeline
    def test_sharded_load_receives_group(self, mocker):
        """Verify sharded_load is called with the distributed group.

        The sharded_load function requires the group for both the
        tensor_parallel_group and pipeline_parallel_group parameters.
        """
        mock_group = MagicMock()
        mock_group.rank.return_value = 0
        mock_group.size.return_value = 4
        mocker.patch.object(
            mlx_lm_module.mx.distributed,
            "init",
            return_value=mock_group,
        )

        mock_model = MagicMock()
        mock_model.model_type = "llama"
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.bos_token = "<s>"
        mock_tokenizer.chat_template = None

        mock_sharded_load = mocker.patch.object(
            mlx_lm_module,
            "sharded_load",
            return_value=(mock_model, mock_tokenizer),
        )
        mocker.patch.object(
            mlx_lm_module,
            "OutlinesTransformerTokenizer",
            return_value=MagicMock(),
        )

        from app.models.mlx_lm import MLX_LM

        MLX_LM(model_path="test-model", pipeline=True)

        # Verify sharded_load was called with the group
        mock_sharded_load.assert_called_once()
        call_args = mock_sharded_load.call_args
        # sharded_load(model_path, group, group)
        assert call_args[0][0] == "test-model"
        assert call_args[0][1] == mock_group
        assert call_args[0][2] == mock_group


class TestMLXLMNonPipelineMode:
    """Tests for non-pipeline (standard) mode."""

    def test_default_mode_is_non_pipeline(self, mocker):
        """Verify pipeline defaults to False."""
        mock_model = MagicMock()
        mock_model.model_type = "llama"
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.bos_token = "<s>"
        mock_tokenizer.chat_template = None
        mocker.patch.object(
            mlx_lm_module,
            "load",
            return_value=(mock_model, mock_tokenizer),
        )
        mocker.patch.object(
            mlx_lm_module,
            "OutlinesTransformerTokenizer",
            return_value=MagicMock(),
        )

        from app.models.mlx_lm import MLX_LM

        model = MLX_LM(model_path="test-model")

        assert model.pipeline is False

    def test_trust_remote_code_passed_to_load(self, mocker):
        """Verify trust_remote_code is passed to tokenizer_config."""
        mock_model = MagicMock()
        mock_model.model_type = "llama"
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.bos_token = "<s>"
        mock_tokenizer.chat_template = None

        mock_load = mocker.patch.object(
            mlx_lm_module,
            "load",
            return_value=(mock_model, mock_tokenizer),
        )
        mocker.patch.object(
            mlx_lm_module,
            "OutlinesTransformerTokenizer",
            return_value=MagicMock(),
        )

        from app.models.mlx_lm import MLX_LM

        MLX_LM(model_path="test-model", trust_remote_code=True)

        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["tokenizer_config"]["trust_remote_code"] is True
