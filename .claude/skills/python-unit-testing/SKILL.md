---
name: python-unit-testing
description: Standards and patterns for Python unit tests with pytest. Use when writing tests, creating test fixtures, setting up test infrastructure, or reviewing test code.
---

# Python Unit Testing Standards

Apply these standards when writing or reviewing Python tests with pytest.

## Directory Structure

Mirror the application layout:

```
tests/
├── conftest.py              # Root fixtures (session-scoped here ONLY)
├── unit/                    # Fast, isolated, mocked tests
│   ├── conftest.py          # Unit-specific fixtures
│   └── module/
│       └── test_component.py
├── integration/             # Tests requiring real services
│   └── conftest.py
└── distributed/             # Multi-process/hardware tests
    ├── conftest.py
    └── mocks.py
```

## Fixture Scope Rules

| Scope | Location | Use Case |
|-------|----------|----------|
| `function` | Any conftest.py | Default; isolated per test |
| `class` | Any conftest.py | Shared within test class |
| `module` | Any conftest.py | Shared within file |
| `session` | Root conftest.py ONLY | Expensive one-time setup (model loading, server startup) |

**Critical**: Session-scoped fixtures placed outside root conftest.py invoke per-module, defeating the purpose.

## Factory Fixture Pattern

For test data requiring multiple variations:

```python
@pytest.fixture
def make_request():
    """Factory for test request objects."""
    def _make(model: str = "test", stream: bool = False, **overrides):
        return Request(model=model, stream=stream, **overrides)
    return _make

def test_streaming(make_request):
    request = make_request(stream=True)
    assert request.stream is True
```

## Mocking

Prefer pytest-mock over unittest.mock for automatic cleanup:

```python
# PREFERRED: pytest-mock
def test_with_mock(mocker):
    mock_fn = mocker.patch("module.function")
    mock_fn.return_value = "test"
    # Automatic cleanup after test

# ACCEPTABLE: unittest.mock
from unittest.mock import patch, MagicMock

def test_with_mock():
    with patch("module.function") as mock_fn:
        mock_fn.return_value = "test"
```

### Environment Variables

Use monkeypatch, never os.environ directly:

```python
def test_config(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    # Automatically reverted after test
```

For session scope:

```python
@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()
```

## Markers

Register all markers in pyproject.toml:

```toml
[tool.pytest.ini_options]
markers = [
    "integration: requires external services",
    "distributed: requires multi-node setup",
    "slow: long-running tests",
]
addopts = "--strict-markers"
```

Apply markers:

```python
@pytest.mark.integration
def test_api_call():
    pass

@pytest.mark.slow
@pytest.mark.distributed
def test_large_model():
    pass
```

## Hardware-Gated Tests

Create skip markers for hardware requirements:

```python
# In conftest.py
def has_gpu():
    try:
        import mlx.core as mx
        return True
    except Exception:
        return False

requires_gpu = pytest.mark.skipif(
    not has_gpu(),
    reason="GPU not available"
)

# Usage
@requires_gpu
def test_gpu_inference():
    pass
```

## Parameterized Tests

Use parametrize for multiple input variations:

```python
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("shape", [(10,), (10, 20)])
def test_operation(dtype, shape):
    # Test runs 4 times: 2 dtypes x 2 shapes
    pass
```

Indirect parametrization for fixture-based setup:

```python
@pytest.fixture
def config(request):
    configs = {"small": {"size": 10}, "large": {"size": 100}}
    return configs[request.param]

@pytest.mark.parametrize("config", ["small", "large"], indirect=True)
def test_with_config(config):
    assert "size" in config
```

## Distributed Testing Pattern

Use FakeProcessGroup for testing code structure without real communication:

```python
@dataclass
class FakeGroup:
    _rank: int = 0
    _size: int = 4

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size

class FakeDistributed:
    def __init__(self, rank: int = 0, world_size: int = 4):
        self._group = FakeGroup(_rank=rank, _size=world_size)

    def init(self) -> FakeGroup:
        return self._group
```

**Caveat**: FakeProcessGroup produces wrong results for actual communication verification. Use only for testing code structure and call patterns.

## Test Categories

Layer tests by execution requirements:

| Layer | Execution | Scope |
|-------|-----------|-------|
| Unit | Every commit | Mocked, < 1s each |
| Integration | Every PR | May use services |
| Hardware | Manual/scheduled | Real hardware |

## CI Model Selection

For ML tests, use minimal models:

```python
@pytest.fixture(scope="session")
def test_model_id():
    return os.getenv("TEST_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

@pytest.fixture(scope="session")
def ci_config():
    return {
        "max_tokens": 10,
        "context_length": 512,
        "temperature": 0.0,  # Deterministic
    }
```

## Anti-Patterns

Avoid these patterns:

1. **Session fixtures outside root conftest.py** - Defeats session scope
2. **Using os.environ directly** - Use monkeypatch for cleanup
3. **Magic values without constants** - Define at module level
4. **Missing marker registration** - Use `--strict-markers`
5. **Implicit test dependencies** - Each test should be independently runnable
6. **Testing implementation details** - Test behavior, not internals
