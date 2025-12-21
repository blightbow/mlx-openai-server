"""Unit tests for distributed file synchronization module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestDetectBackend:
    """Tests for backend detection heuristics."""

    def test_explicit_override(self):
        """MLX_DISTRIBUTED_BACKEND env var takes precedence."""
        from app.distributed.file_sync import detect_backend

        with patch.dict(os.environ, {"MLX_DISTRIBUTED_BACKEND": "jaccl"}):
            assert detect_backend() == "jaccl"

        with patch.dict(os.environ, {"MLX_DISTRIBUTED_BACKEND": "mpi"}):
            assert detect_backend() == "mpi"

    def test_jaccl_detection(self):
        """JACCL detected via MLX_JACCL_COORDINATOR env var."""
        from app.distributed.file_sync import detect_backend

        with patch.dict(
            os.environ,
            {"MLX_JACCL_COORDINATOR": "192.168.0.1"},
            clear=True,
        ):
            assert detect_backend() == "jaccl"

    def test_mpi_detection(self):
        """MPI detected via OMPI_* or PMI_* env vars."""
        from app.distributed.file_sync import detect_backend

        with patch.dict(
            os.environ,
            {"OMPI_COMM_WORLD_SIZE": "2"},
            clear=True,
        ):
            assert detect_backend() == "mpi"

        with patch.dict(
            os.environ,
            {"PMI_RANK": "0"},
            clear=True,
        ):
            assert detect_backend() == "mpi"

    def test_ring_detection(self):
        """Ring backend detected via MLX_HOSTFILE (without JACCL)."""
        from app.distributed.file_sync import detect_backend

        with patch.dict(
            os.environ,
            {"MLX_HOSTFILE": "/tmp/hosts.txt"},
            clear=True,
        ):
            assert detect_backend() == "ring"

    def test_unknown_fallback(self):
        """Returns 'unknown' when no backend detected."""
        from app.distributed.file_sync import detect_backend

        with patch.dict(os.environ, {}, clear=True):
            assert detect_backend() == "unknown"


class TestChunkSize:
    """Tests for chunk size selection."""

    def test_chunk_sizes_by_backend(self):
        """Each backend gets appropriate chunk size."""
        from app.distributed.file_sync import (
            get_chunk_size,
            CHUNK_SIZE_JACCL,
            CHUNK_SIZE_MPI,
            CHUNK_SIZE_RING,
        )

        assert get_chunk_size("jaccl") == CHUNK_SIZE_JACCL
        assert get_chunk_size("ring") == CHUNK_SIZE_RING
        assert get_chunk_size("mpi") == CHUNK_SIZE_MPI
        assert get_chunk_size("unknown") == CHUNK_SIZE_MPI  # conservative fallback

    def test_env_override(self):
        """MLX_FILE_SYNC_CHUNK_SIZE overrides default."""
        from app.distributed.file_sync import get_chunk_size

        with patch.dict(os.environ, {"MLX_FILE_SYNC_CHUNK_SIZE": "1048576"}):
            assert get_chunk_size("jaccl") == 1048576


class TestFileClassification:
    """Tests for metadata vs weight file classification."""

    def test_metadata_files(self):
        """Metadata patterns correctly identified."""
        from app.distributed.file_sync import get_metadata_files

        manifest = [
            ("config.json", 1000),
            ("tokenizer.json", 5000),
            ("vocab.txt", 2000),
            ("tokenizer.model", 3000),
            ("chat_template.jinja", 500),
            ("handler.py", 1500),
            ("model-00001.safetensors", 1_000_000_000),
        ]
        metadata = get_metadata_files(manifest)

        assert len(metadata) == 6
        names = [name for name, _ in metadata]
        assert "config.json" in names
        assert "tokenizer.json" in names
        assert "vocab.txt" in names
        assert "tokenizer.model" in names
        assert "chat_template.jinja" in names
        assert "handler.py" in names
        assert "model-00001.safetensors" not in names

    def test_weight_files(self):
        """Weight files correctly identified."""
        from app.distributed.file_sync import get_weight_files

        manifest = [
            ("config.json", 1000),
            ("model-00001-of-00004.safetensors", 1_000_000_000),
            ("model-00002-of-00004.safetensors", 1_000_000_000),
            ("model.safetensors", 500_000_000),
        ]
        weights = get_weight_files(manifest)

        assert len(weights) == 3
        names = [name for name, _ in weights]
        assert "model-00001-of-00004.safetensors" in names
        assert "model-00002-of-00004.safetensors" in names
        assert "model.safetensors" in names
        assert "config.json" not in names


class TestPipelineFileComputation:
    """Tests for computing which files each rank needs in pipeline mode."""

    @pytest.fixture
    def model_dir(self):
        """Create a temporary model directory with config and index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # 8 layers, 4 weight files
            config = {"num_hidden_layers": 8}
            with open(tmppath / "config.json", "w") as f:
                json.dump(config, f)

            weight_map = {
                # Embeddings (rank 0 needs)
                "model.embed_tokens.weight": "model-00001.safetensors",
                # Layers 0-1
                "model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
                "model.layers.1.self_attn.q_proj.weight": "model-00001.safetensors",
                # Layers 2-3
                "model.layers.2.self_attn.q_proj.weight": "model-00002.safetensors",
                "model.layers.3.self_attn.q_proj.weight": "model-00002.safetensors",
                # Layers 4-5
                "model.layers.4.self_attn.q_proj.weight": "model-00003.safetensors",
                "model.layers.5.self_attn.q_proj.weight": "model-00003.safetensors",
                # Layers 6-7
                "model.layers.6.self_attn.q_proj.weight": "model-00004.safetensors",
                "model.layers.7.self_attn.q_proj.weight": "model-00004.safetensors",
                # LM head (last rank needs)
                "lm_head.weight": "model-00004.safetensors",
            }
            with open(tmppath / "model.safetensors.index.json", "w") as f:
                json.dump({"weight_map": weight_map}, f)

            yield tmppath

    def test_two_rank_pipeline(self, model_dir):
        """Two ranks split 8 layers evenly (4 each)."""
        from app.distributed.file_sync import compute_pipeline_files

        # Rank 0: layers 0-3 + embeddings
        rank0_files = compute_pipeline_files(
            model_dir / "model.safetensors.index.json",
            model_dir / "config.json",
            rank=0,
            world_size=2,
        )
        assert "model-00001.safetensors" in rank0_files  # layers 0-1 + embed
        assert "model-00002.safetensors" in rank0_files  # layers 2-3

        # Rank 1: layers 4-7 + lm_head
        rank1_files = compute_pipeline_files(
            model_dir / "model.safetensors.index.json",
            model_dir / "config.json",
            rank=1,
            world_size=2,
        )
        assert "model-00003.safetensors" in rank1_files  # layers 4-5
        assert "model-00004.safetensors" in rank1_files  # layers 6-7 + lm_head

        # Rank 1 should NOT have rank 0's layer-only files
        assert "model-00002.safetensors" not in rank1_files

    def test_four_rank_pipeline(self, model_dir):
        """Four ranks split 8 layers (2 each)."""
        from app.distributed.file_sync import compute_pipeline_files

        # Rank 0: layers 0-1
        rank0_files = compute_pipeline_files(
            model_dir / "model.safetensors.index.json",
            model_dir / "config.json",
            rank=0,
            world_size=4,
        )
        assert "model-00001.safetensors" in rank0_files

        # Rank 3: layers 6-7 + lm_head
        rank3_files = compute_pipeline_files(
            model_dir / "model.safetensors.index.json",
            model_dir / "config.json",
            rank=3,
            world_size=4,
        )
        assert "model-00004.safetensors" in rank3_files

    def test_missing_index_returns_none(self, model_dir):
        """Missing index file returns None (fallback to full transfer)."""
        from app.distributed.file_sync import compute_pipeline_files

        # Remove index file
        (model_dir / "model.safetensors.index.json").unlink()

        result = compute_pipeline_files(
            model_dir / "model.safetensors.index.json",
            model_dir / "config.json",
            rank=0,
            world_size=2,
        )
        assert result is None

    def test_embed_lm_head_separation(self):
        """Embeddings go to rank 0, lm_head goes to last rank only."""
        from app.distributed.file_sync import compute_pipeline_files

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # 4 layers, separate files for embed and lm_head
            config = {"num_hidden_layers": 4}
            with open(tmppath / "config.json", "w") as f:
                json.dump(config, f)

            weight_map = {
                # Embeddings in dedicated file
                "model.embed_tokens.weight": "embed.safetensors",
                # Layers 0-1
                "model.layers.0.weight": "layers-0-1.safetensors",
                "model.layers.1.weight": "layers-0-1.safetensors",
                # Layers 2-3
                "model.layers.2.weight": "layers-2-3.safetensors",
                "model.layers.3.weight": "layers-2-3.safetensors",
                # LM head and final norm in dedicated file
                "lm_head.weight": "lm_head.safetensors",
                "model.norm.weight": "lm_head.safetensors",
            }
            with open(tmppath / "model.safetensors.index.json", "w") as f:
                json.dump({"weight_map": weight_map}, f)

            # Rank 0: layers 0-1 + embeddings (NOT lm_head)
            rank0_files = compute_pipeline_files(
                tmppath / "model.safetensors.index.json",
                tmppath / "config.json",
                rank=0,
                world_size=2,
            )
            assert "embed.safetensors" in rank0_files
            assert "layers-0-1.safetensors" in rank0_files
            assert "lm_head.safetensors" not in rank0_files

            # Rank 1: layers 2-3 + lm_head (NOT embeddings)
            rank1_files = compute_pipeline_files(
                tmppath / "model.safetensors.index.json",
                tmppath / "config.json",
                rank=1,
                world_size=2,
            )
            assert "layers-2-3.safetensors" in rank1_files
            assert "lm_head.safetensors" in rank1_files
            assert "embed.safetensors" not in rank1_files


class TestNodePrefix:
    """Tests for log message node prefix."""

    def test_mlx_launch_format(self):
        """Under mlx.launch, uses [hostname/rank] format."""
        from app.distributed.file_sync import get_node_prefix

        with patch.dict(os.environ, {"MLX_RANK": "1"}):
            prefix = get_node_prefix(1)
            assert "/" in prefix
            assert "1" in prefix

    def test_standalone_format(self):
        """Without mlx.launch, uses [Rank N] format."""
        from app.distributed.file_sync import get_node_prefix

        with patch.dict(os.environ, {}, clear=True):
            prefix = get_node_prefix(0)
            assert prefix == "[Rank 0]"
