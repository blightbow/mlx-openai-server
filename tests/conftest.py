"""Root pytest configuration and shared fixtures.

This module provides:
- Session-scoped fixtures for HTTP client, model loading
- Hardware detection for distributed testing
- Skip markers for conditional test execution
- Mock fixtures for unit testing
- Distributed testing fixtures and CLI options
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import httpx

from tests.distributed.hardware import (
    DistributedCapability,
    detect_distributed_capability,
)


# ============================================================================
# Hardware Detection Functions
# ============================================================================


def has_mlx_distributed() -> bool:
    """Check if MLX distributed is available with >1 rank."""
    try:
        import mlx.core as mx

        group = mx.distributed.init()
        return group.size() > 1
    except Exception:
        return False


def has_rdma() -> bool:
    """Check if RDMA/JACCL is available (macOS 26.2+)."""
    import subprocess

    try:
        result = subprocess.run(
            ["ibv_devices"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def has_mlx() -> bool:
    """Check if MLX is installed."""
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


# ============================================================================
# Skip Markers
# ============================================================================

requires_distributed = pytest.mark.skipif(
    not has_mlx_distributed(),
    reason="MLX distributed not available (requires multi-node setup)",
)

requires_rdma = pytest.mark.skipif(
    not has_rdma(),
    reason="RDMA/JACCL not available (requires macOS 26.2+ with Thunderbolt 5)",
)

requires_mlx = pytest.mark.skipif(
    not has_mlx(),
    reason="MLX not installed",
)


# ============================================================================
# Session-Scoped Utilities
# ============================================================================


@pytest.fixture(scope="session")
def monkeypatch_session() -> pytest.MonkeyPatch:
    """Session-scoped monkeypatch.

    Use for environment variables that should persist across the test session.
    """
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


# ============================================================================
# API Testing Fixtures (moved from test_llm_contracts.py)
# ============================================================================


def env_base_url() -> str:
    """Get the base URL for the MLX server from environment variables."""
    raw = os.getenv("MLX_URL", "http://127.0.0.1:8000")
    return raw.rstrip("/")


def build_headers() -> dict[str, str]:
    """Build HTTP headers for API requests."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MLX_API_KEY")
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


@pytest.fixture(scope="session")
def base_url() -> str:
    """Fixture providing the base URL for the MLX server."""
    return env_base_url()


@pytest.fixture(scope="session")
def headers() -> dict[str, str]:
    """Fixture providing HTTP headers for API requests."""
    return build_headers()


@pytest.fixture(scope="session")
def http_client(base_url: str, headers: dict[str, str]) -> "httpx.Client":
    """Fixture providing an HTTP client configured for the MLX server."""
    import httpx

    client = httpx.Client(base_url=base_url, timeout=30.0, headers=headers)
    yield client
    client.close()


@pytest.fixture(scope="session")
def model_id() -> str | None:
    """Fixture providing the model ID from environment variables."""
    return os.getenv("MLX_MODEL_ID")


@pytest.fixture(scope="session")
def server_available(http_client: "httpx.Client") -> bool:
    """Fixture that checks if the MLX server is available."""
    import httpx

    try:
        response = http_client.get("/health", timeout=5.0)
    except (httpx.ConnectError, httpx.TimeoutException):
        return False
    else:
        return response.status_code == 200


# ============================================================================
# Test Model Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_model_id() -> str:
    """Model ID for testing - uses tiny model by default for fast CI."""
    return os.getenv("MLX_TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")


@pytest.fixture(scope="session")
def ci_model_config() -> dict[str, int | float]:
    """Configuration optimized for CI testing - minimal generation."""
    return {
        "max_tokens": 10,
        "context_length": 512,
        "temperature": 0.0,
    }


# ============================================================================
# Mock Fixtures for Unit Testing
# ============================================================================


@pytest.fixture
def mock_distributed_group(mocker):
    """Mock mx.distributed for unit tests.

    Returns a FakeDistributed instance for assertions.
    """
    from tests.distributed.mocks import create_mock_distributed

    return create_mock_distributed(mocker, rank=0, world_size=4)


@pytest.fixture
def mock_sharded_load(mocker):
    """Mock sharded_load for unit tests.

    Patches at the usage location (app.models.mlx_lm) not the definition.
    Returns (mock_model, mock_tokenizer) tuple.
    """
    mock_model = mocker.MagicMock()
    mock_model.model_type = "llama"
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.bos_token = "<s>"

    mocker.patch(
        "app.models.mlx_lm.sharded_load",
        return_value=(mock_model, mock_tokenizer),
    )
    return mock_model, mock_tokenizer


@pytest.fixture
def mock_load(mocker):
    """Mock load for unit tests.

    Patches at the usage location (app.models.mlx_lm) not the definition.
    Returns (mock_model, mock_tokenizer) tuple.
    """
    mock_model = mocker.MagicMock()
    mock_model.model_type = "llama"
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.bos_token = "<s>"

    mocker.patch(
        "app.models.mlx_lm.load",
        return_value=(mock_model, mock_tokenizer),
    )
    return mock_model, mock_tokenizer


# ============================================================================
# Factory Fixtures
# ============================================================================


@pytest.fixture
def make_chat_messages():
    """Factory for chat message lists."""

    def _make(
        user_content: str = "Hello",
        system_content: str | None = None,
    ) -> list[dict[str, str]]:
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
        return messages

    return _make


# ============================================================================
# Distributed Testing CLI Options and Fixtures
# ============================================================================


@dataclass
class DistributedTestConfig:
    """Configuration for distributed test execution.

    Attributes
    ----------
    capability : DistributedCapability
        Detected hardware capabilities.
    hostfile_path : Path | None
        Path to the JACCL hostfile if provided.
    test_model : str
        Model ID to use for distributed tests.
    force_mocks : bool
        Whether to force mock mode even if hardware is available.
    """

    capability: DistributedCapability
    hostfile_path: Path | None
    test_model: str
    force_mocks: bool


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add distributed testing CLI options."""
    parser.addoption(
        "--run-distributed",
        action="store_true",
        default=False,
        help="Enable real distributed tests (requires multi-node setup)",
    )
    parser.addoption(
        "--hostfile",
        action="store",
        default=None,
        help="Path to JACCL hostfile for distributed tests",
    )
    parser.addoption(
        "--force-mocks",
        action="store_true",
        default=False,
        help="Force mock mode even if hardware is available",
    )


@pytest.fixture(scope="session")
def distributed_capability(request: pytest.FixtureRequest) -> DistributedCapability:
    """Detect distributed computing capabilities.

    Returns hardware detection results including RDMA availability,
    Thunderbolt 5 detection, and MLX distributed status.
    """
    hostfile = request.config.getoption("--hostfile")
    return detect_distributed_capability(hostfile=hostfile)


@pytest.fixture(scope="session")
def distributed_config(
    request: pytest.FixtureRequest,
    distributed_capability: DistributedCapability,
) -> DistributedTestConfig:
    """Provide complete distributed test configuration.

    Combines hardware detection with CLI options and environment variables.
    """
    hostfile = request.config.getoption("--hostfile")
    force_mocks = request.config.getoption("--force-mocks")

    hostfile_path = Path(hostfile) if hostfile else None

    test_model = os.getenv(
        "MLX_TEST_MODEL",
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    )

    return DistributedTestConfig(
        capability=distributed_capability,
        hostfile_path=hostfile_path,
        test_model=test_model,
        force_mocks=force_mocks,
    )


@pytest.fixture(scope="session")
def run_distributed_enabled(request: pytest.FixtureRequest) -> bool:
    """Check if distributed tests are enabled via CLI."""
    return request.config.getoption("--run-distributed")


@pytest.fixture
def skip_without_distributed(
    run_distributed_enabled: bool,
    distributed_capability: DistributedCapability,
) -> None:
    """Skip test if distributed testing is not enabled or available.

    Use this fixture in tests that require real distributed hardware:

        def test_real_distributed(skip_without_distributed):
            # This test only runs with --run-distributed and hardware
            ...
    """
    if not run_distributed_enabled:
        pytest.skip("Distributed tests disabled (use --run-distributed to enable)")

    if not distributed_capability.is_available:
        pytest.skip(
            f"Distributed not available: {distributed_capability.error_message}"
        )


@pytest.fixture
def skip_without_rdma(
    run_distributed_enabled: bool,
    distributed_capability: DistributedCapability,
) -> None:
    """Skip test if RDMA/JACCL is not available.

    Use this fixture in tests that specifically require JACCL backend:

        def test_rdma_collective(skip_without_rdma):
            # This test only runs with JACCL backend
            ...
    """
    if not run_distributed_enabled:
        pytest.skip("Distributed tests disabled (use --run-distributed to enable)")

    if not distributed_capability.rdma_available:
        pytest.skip("RDMA/JACCL not available (requires macOS 26.2+ with TB5)")
