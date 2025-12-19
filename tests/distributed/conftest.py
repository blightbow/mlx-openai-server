"""Distributed testing fixtures.

Note: Most distributed fixtures are defined in tests/conftest.py
to ensure they're available to all test modules including
tests/integration/. This file provides additional utilities
specific to the distributed test module.

See tests/conftest.py for:
- pytest_addoption (CLI options)
- distributed_capability
- distributed_config
- skip_without_distributed
- skip_without_rdma
"""

from __future__ import annotations

# Re-export for convenience
from .hardware import DistributedCapability, detect_distributed_capability

__all__ = [
    "DistributedCapability",
    "detect_distributed_capability",
]
