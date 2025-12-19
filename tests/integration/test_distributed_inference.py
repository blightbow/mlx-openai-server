"""Integration tests for distributed pipeline parallel inference.

These tests require actual multi-node hardware setup and are skipped
by default. Enable with: pytest --run-distributed --hostfile hosts.json

Prerequisites:
- macOS 26.2+ with RDMA enabled (rdma_ctl enable in recovery mode)
- Thunderbolt 5 connection between nodes
- SSH key authentication between nodes
- JACCL hostfile configured

See docs/DISTRIBUTED_TESTING.md for setup instructions.
"""

from __future__ import annotations

import os
import subprocess

import pytest


@pytest.mark.distributed
class TestDistributedCapabilityDetection:
    """Tests for hardware capability detection."""

    def test_hardware_detection_runs_without_error(
        self,
        distributed_capability,
    ):
        """Verify hardware detection completes without exceptions."""
        # The fixture itself tests that detection works
        assert distributed_capability is not None
        assert distributed_capability.macos_version is not None

    def test_mlx_version_detected(self, distributed_capability):
        """Verify MLX version is detected."""
        if distributed_capability.mlx_version is None:
            pytest.skip("MLX not installed")

        assert distributed_capability.mlx_version != "unknown"

    def test_backend_determination(self, distributed_capability):
        """Verify backend is determined correctly."""
        assert distributed_capability.backend in ("jaccl", "mpi", "none")

    def test_capability_reports_availability(self, distributed_capability):
        """Verify capability reports availability status."""
        # is_available should be True only if world_size > 1
        # This may be False on single-node without --run-distributed
        assert isinstance(distributed_capability.is_available, bool)


@pytest.mark.distributed
class TestHostConnectivity:
    """Tests for multi-node connectivity (requires hostfile)."""

    def test_hostfile_loaded(
        self,
        skip_without_distributed,
        distributed_config,
    ):
        """Verify hostfile is loaded when provided."""
        if distributed_config.hostfile_path is None:
            pytest.skip("No hostfile provided (use --hostfile)")

        assert distributed_config.hostfile_path.exists()
        assert len(distributed_config.capability.detected_hosts) > 0

    def test_ssh_connectivity(
        self,
        skip_without_distributed,
        distributed_config,
    ):
        """Verify SSH connectivity to all hosts."""
        if not distributed_config.capability.detected_hosts:
            pytest.skip("No hosts detected")

        for host in distributed_config.capability.detected_hosts:
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, "echo ok"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, f"SSH failed to {host}: {result.stderr}"


@pytest.mark.distributed
@pytest.mark.rdma
class TestRDMADevices:
    """Tests for RDMA/JACCL backend (requires TB5 hardware)."""

    def test_rdma_devices_present(
        self,
        skip_without_rdma,
        distributed_capability,
    ):
        """Verify RDMA devices are detected."""
        assert distributed_capability.rdma_available
        assert distributed_capability.backend == "jaccl"

    def test_thunderbolt5_detected(
        self,
        skip_without_rdma,
        distributed_capability,
    ):
        """Verify Thunderbolt 5 is detected."""
        # TB5 detection is best-effort; RDMA being available is sufficient
        if not distributed_capability.thunderbolt5_detected:
            pytest.skip("TB5 not explicitly detected (may still work)")

        assert distributed_capability.thunderbolt5_detected


@pytest.mark.distributed
@pytest.mark.slow
class TestPipelineInference:
    """End-to-end tests for pipeline parallel inference.

    These tests launch the server in distributed mode and verify
    it responds correctly. Requires full multi-node setup.
    """

    def test_pipeline_server_health_check(
        self,
        skip_without_distributed,
        distributed_config,
        http_client,
    ):
        """Verify server /health responds in pipeline mode.

        This test assumes the server is already running in pipeline
        mode on the configured URL (MLX_URL environment variable).
        """
        # Check if server is running
        import httpx

        try:
            response = http_client.get("/health", timeout=10.0)
        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Server not running (start with mlx.launch)")

        assert response.status_code == 200
        data = response.json()
        assert data.get("status") in ("ok", "healthy", "ready")

    def test_pipeline_models_endpoint(
        self,
        skip_without_distributed,
        http_client,
        server_available,
    ):
        """Verify /v1/models returns model list in pipeline mode."""
        if not server_available:
            pytest.skip("Server not available")

        response = http_client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0

    def test_pipeline_chat_completion(
        self,
        skip_without_distributed,
        http_client,
        server_available,
        model_id,
    ):
        """Verify chat completion works in pipeline mode."""
        if not server_available:
            pytest.skip("Server not available")

        if not model_id:
            # Try to get model from /v1/models
            response = http_client.get("/v1/models")
            if response.status_code == 200:
                data = response.json()
                if data["data"]:
                    model_id = data["data"][0]["id"]

        if not model_id:
            pytest.skip("No model available")

        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": "Say 'test' and nothing else."},
            ],
            "max_tokens": 10,
            "temperature": 0,
        }

        response = http_client.post(
            "/v1/chat/completions",
            json=payload,
            timeout=60.0,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"]

    def test_pipeline_streaming(
        self,
        skip_without_distributed,
        http_client,
        server_available,
        model_id,
    ):
        """Verify streaming works in pipeline mode."""
        if not server_available:
            pytest.skip("Server not available")

        if not model_id:
            response = http_client.get("/v1/models")
            if response.status_code == 200:
                data = response.json()
                if data["data"]:
                    model_id = data["data"][0]["id"]

        if not model_id:
            pytest.skip("No model available")

        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": "Count from 1 to 3."},
            ],
            "stream": True,
            "max_tokens": 20,
            "temperature": 0,
        }

        chunks_received = 0
        with http_client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
            timeout=60.0,
        ) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line and line.startswith("data:"):
                    chunks_received += 1

        assert chunks_received > 0, "No streaming chunks received"


@pytest.mark.distributed
class TestDistributedEnvironment:
    """Tests for distributed environment configuration."""

    def test_env_mlx_metal_fast_synch(
        self,
        skip_without_distributed,
        distributed_capability,
    ):
        """Verify MLX_METAL_FAST_SYNCH recommendation.

        This env var is critical for low-latency distributed communication.
        However, it requires building MLX from source as of 0.24.0.
        """
        if not distributed_capability.fast_sync_supported:
            pytest.skip(
                "MLX_METAL_FAST_SYNCH requires MLX built from source. "
                "See: https://github.com/ml-explore/mlx/issues/1993"
            )

        # If supported, verify it's set
        fast_sync = os.getenv("MLX_METAL_FAST_SYNCH")
        if fast_sync != "1":
            pytest.fail(
                "MLX_METAL_FAST_SYNCH=1 recommended for distributed testing. "
                "Set this environment variable for optimal latency."
            )
