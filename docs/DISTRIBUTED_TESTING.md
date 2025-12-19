# Distributed Testing Guide

This guide explains how to set up and run distributed tests for mlx-openai-server's pipeline parallel inference feature.

## Overview

The test framework supports three layers:

| Layer | Command | Requirements |
|-------|---------|--------------|
| Unit Tests | `pytest tests/unit/` | None (mocked) |
| Integration Tests | `pytest tests/integration/` | Running server |
| Distributed Tests | `pytest --run-distributed` | Multi-node JACCL setup |

## Quick Start

### Running Mocked Tests (Default)

```bash
# Install dev dependencies
uv sync --group dev

# Run all mocked tests
pytest tests/

# Run only unit tests
pytest tests/unit/

# Run tests with specific marker
pytest -m pipeline
```

### Running Integration Tests

```bash
# Start the server first
mlx-openai-server launch --model mlx-community/Qwen2.5-0.5B-Instruct

# In another terminal, run integration tests
pytest -m integration
```

## Distributed Testing Setup

Distributed tests require a multi-node setup with Thunderbolt 5 and RDMA.

### Prerequisites

- macOS 26.2+ (Sequoia or later)
- Thunderbolt 5 connection between Mac nodes
- SSH key authentication configured

### Step 1: Enable RDMA (Recovery Mode Required)

1. Shut down each Mac
2. Hold power button to enter Recovery Mode
3. Open Terminal from Utilities menu
4. Run:
   ```bash
   rdma_ctl enable
   ```
5. Restart normally

### Step 2: Configure SSH

Set up passwordless SSH between nodes:

```bash
# On primary node, generate key if needed
ssh-keygen -t ed25519 -C "mlx-distributed"

# Copy to secondary node(s)
ssh-copy-id user@node2.local

# Test connectivity
ssh node2.local "echo Connected"
```

### Step 3: Create Hostfile

Create a JSON hostfile for JACCL:

```bash
# Generate hostfile
mlx.distributed_config \
    --hosts node1.local,node2.local \
    --over thunderbolt \
    --backend jaccl \
    --auto-setup \
    --output hosts.json
```

Example `hosts.json`:
```json
{
  "hosts": [
    {"hostname": "node1.local", "port": 22},
    {"hostname": "node2.local", "port": 22}
  ],
  "backend": "jaccl"
}
```

### Step 4: Download Test Model

The test framework uses a small model by default:

```bash
# Pre-download the test model on all nodes
python -c "from mlx_lm.utils import load; load('mlx-community/Qwen2.5-0.5B-Instruct-4bit')"
```

### Step 5: Set Environment Variables (Optional but Recommended)

```bash
# Enable fast GPU-CPU synchronization (if MLX built from source)
export MLX_METAL_FAST_SYNCH=1

# Custom test model
export MLX_TEST_MODEL="mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# Server URL for integration tests
export MLX_URL="http://127.0.0.1:8000"
```

### Step 6: Run Distributed Tests

```bash
# Run with hostfile
pytest --run-distributed --hostfile hosts.json

# Run only RDMA-specific tests
pytest --run-distributed --hostfile hosts.json -m rdma

# Verbose output
pytest --run-distributed --hostfile hosts.json -v
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.integration` | Requires running server |
| `@pytest.mark.distributed` | Requires multi-node setup |
| `@pytest.mark.pipeline` | Exercises pipeline parallel mode |
| `@pytest.mark.rdma` | Requires JACCL/RDMA backend |
| `@pytest.mark.slow` | Long-running tests |

### Combining Markers

```bash
# Run only pipeline unit tests
pytest -m "pipeline and not distributed"

# Run all distributed tests except slow
pytest --run-distributed -m "distributed and not slow"
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--run-distributed` | Enable distributed tests (skipped by default) |
| `--hostfile PATH` | Path to JACCL hostfile |
| `--force-mocks` | Force mock mode even if hardware available |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLX_TEST_MODEL` | Model for test inference | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` |
| `MLX_URL` | Server URL for integration tests | `http://127.0.0.1:8000` |
| `MLX_MODEL_ID` | Override model ID in tests | (auto-detect) |
| `MLX_HOSTFILE` | Default hostfile path | (none) |
| `MLX_METAL_FAST_SYNCH` | Enable fast sync (requires source build) | (not set) |

## Troubleshooting

### RDMA Not Detected

```
Error: RDMA/JACCL not available
```

1. Verify macOS version: `sw_vers` (must be 26.2+)
2. Check RDMA enabled: `ibv_devices` should show devices
3. Verify Thunderbolt connection: `system_profiler SPThunderboltDataType`

### SSH Connection Failed

```
Error: SSH failed to node2.local
```

1. Test SSH manually: `ssh -v node2.local`
2. Ensure SSH key copied: `ssh-copy-id user@node2.local`
3. Check firewall: Remote Login must be enabled in System Preferences

### MLX_METAL_FAST_SYNCH Not Supported

```
Warning: MLX_METAL_FAST_SYNCH requires MLX built from source
```

As of MLX 0.24.0, this feature requires building MLX from source. PyPI releases
don't include the `input_coherent` kernel needed for fast synchronization.

See: https://github.com/ml-explore/mlx/issues/1993

To build MLX from source:
```bash
git clone https://github.com/ml-explore/mlx.git
cd mlx
pip install .
```

### Tests Hang or Timeout

1. Check network connectivity between nodes
2. Verify hostfile paths are correct
3. Ensure model is downloaded on all nodes
4. Check for port conflicts

## Architecture

```
tests/
├── conftest.py                    # Root fixtures (session-scoped)
├── distributed/
│   ├── conftest.py                # Distributed CLI options & fixtures
│   ├── hardware.py                # Hardware detection
│   └── mocks.py                   # FakeDistributed for unit tests
├── unit/
│   ├── test_mlx_lm_pipeline.py    # Pipeline mode unit tests
│   └── test_server_rank_gating.py # Rank gating tests
└── integration/
    └── test_distributed_inference.py # Real hardware tests
```

### Fixture Hierarchy

```
Session-scoped (conftest.py):
  ├── http_client
  ├── base_url / headers
  ├── test_model_id
  └── distributed_capability

Function-scoped:
  ├── mock_distributed_group
  ├── mock_sharded_load
  └── skip_without_distributed
```

## Writing New Tests

### Unit Test (Mocked)

```python
@pytest.mark.pipeline
def test_my_feature(mocker, mock_distributed_group):
    """Test with mocked distributed."""
    # mock_distributed_group is pre-configured
    assert mock_distributed_group.was_initialized is False

    from app.models.mlx_lm import MLX_LM
    # ... test code
```

### Integration Test (Real Hardware)

```python
@pytest.mark.distributed
def test_real_feature(skip_without_distributed, distributed_config):
    """Test requiring real hardware."""
    # skip_without_distributed raises skip if hardware not available
    assert distributed_config.capability.is_available
    # ... test code
```

## References

- [MLX Distributed Documentation](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [JACCL Getting Started](https://ml-explore.github.io/mlx/build/html/usage/distributed.html#getting-started-with-jaccl)
- [MLX GitHub Issues](https://github.com/ml-explore/mlx/issues)
