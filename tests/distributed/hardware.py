"""Hardware detection utilities for distributed testing.

Detects availability of MLX distributed features, RDMA/JACCL backend,
and Thunderbolt 5 connectivity for real distributed testing.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass


@dataclass
class DistributedCapability:
    """Detected distributed computing capabilities.

    Attributes
    ----------
    is_available : bool
        Whether any distributed backend is available.
    rdma_available : bool
        Whether RDMA/JACCL backend is available (macOS 26.2+).
    thunderbolt5_detected : bool
        Whether Thunderbolt 5 interface was detected.
    backend : str
        Recommended backend ("jaccl", "mpi", or "none").
    detected_hosts : list[str]
        List of detected peer hosts (from hostfile or discovery).
    macos_version : str
        Current macOS version string.
    mlx_version : str | None
        MLX version if available.
    fast_sync_supported : bool
        Whether MLX_METAL_FAST_SYNCH is supported (requires source build).
    error_message : str | None
        Error message if detection failed.
    """

    is_available: bool
    rdma_available: bool
    thunderbolt5_detected: bool
    backend: str
    detected_hosts: list[str]
    macos_version: str
    mlx_version: str | None
    fast_sync_supported: bool
    error_message: str | None


def get_macos_version() -> str:
    """Get macOS version string.

    Returns
    -------
    str
        macOS version (e.g., "15.2.0") or "unknown" if detection fails.
    """
    if platform.system() != "Darwin":
        return "not-macos"
    try:
        return platform.mac_ver()[0]
    except Exception:
        return "unknown"


def check_rdma_devices() -> tuple[bool, list[str]]:
    """Check for RDMA devices via ibv_devices.

    Returns
    -------
    tuple[bool, list[str]]
        (devices_found, list_of_device_names)
    """
    try:
        result = subprocess.run(
            ["ibv_devices"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False, []

        devices = []
        for line in result.stdout.splitlines():
            line = line.strip()
            # Skip header lines
            if line.startswith("device") or not line:
                continue
            # First column is device name
            parts = line.split()
            if parts:
                devices.append(parts[0])

        return len(devices) > 0, devices

    except FileNotFoundError:
        # ibv_devices not installed (RDMA not available)
        return False, []
    except subprocess.TimeoutExpired:
        return False, []
    except Exception:
        return False, []


def check_thunderbolt5() -> bool:
    """Check for Thunderbolt 5 interfaces.

    Returns
    -------
    bool
        True if TB5 interface detected.
    """
    try:
        # Use system_profiler to check for Thunderbolt
        result = subprocess.run(
            ["system_profiler", "SPThunderboltDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False

        # Check for Thunderbolt 5 indicators
        output = result.stdout.lower()
        return "thunderbolt 5" in output or "usb4" in output

    except FileNotFoundError:
        return False
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def get_mlx_version() -> str | None:
    """Get installed MLX version.

    Returns
    -------
    str | None
        MLX version string or None if not installed.
    """
    try:
        import mlx.core as mx  # type: ignore[import-not-found]

        # Try mlx.core first (where __version__ is typically defined)
        version = getattr(mx, "__version__", None)
        if version:
            return version

        # Fallback: try the mlx package itself
        import mlx  # type: ignore[import-not-found]

        return getattr(mlx, "__version__", "unknown")
    except ImportError:
        return None


def check_mlx_distributed() -> tuple[bool, int]:
    """Check if MLX distributed is available and functional.

    Returns
    -------
    tuple[bool, int]
        (is_available, world_size)
    """
    try:
        import mlx.core as mx  # type: ignore[import-not-found]

        group = mx.distributed.init()
        return True, group.size()
    except Exception:
        return False, 0


def check_fast_sync_support() -> bool:
    """Check if MLX_METAL_FAST_SYNCH is supported.

    This requires building MLX from source as of 0.24.0.
    PyPI releases don't include the input_coherent kernel.

    Returns
    -------
    bool
        True if fast sync appears to be supported.
    """
    # Check if env var is already set
    if os.getenv("MLX_METAL_FAST_SYNCH"):
        return True

    # Try to detect support by checking for the kernel
    try:
        import mlx.core as mx  # type: ignore[import-not-found]

        # This is a heuristic - actual detection would require
        # checking compiled kernel availability
        version = getattr(mx, "__version__", "0.0.0")
        major, minor, *_ = version.split(".")
        # Future versions may include this by default
        return int(major) > 0 or int(minor) > 24
    except Exception:
        return False


def load_hostfile(path: str) -> list[str]:
    """Load hosts from a JACCL hostfile.

    Parameters
    ----------
    path : str
        Path to the hostfile (JSON format).

    Returns
    -------
    list[str]
        List of hostnames from the file.
    """
    import json
    from pathlib import Path

    hostfile_path = Path(path)
    if not hostfile_path.exists():
        return []

    try:
        with hostfile_path.open() as f:
            data = json.load(f)

        # JACCL hostfile format
        if isinstance(data, dict) and "hosts" in data:
            return [h.get("hostname", h.get("host", "")) for h in data["hosts"]]

        # Simple list format
        if isinstance(data, list):
            return [str(h) for h in data]

        return []
    except Exception:
        return []


def detect_distributed_capability(
    hostfile: str | None = None,
) -> DistributedCapability:
    """Detect distributed computing capabilities.

    Parameters
    ----------
    hostfile : str | None
        Optional path to JACCL hostfile for host detection.

    Returns
    -------
    DistributedCapability
        Detected capabilities and configuration.
    """
    macos_version = get_macos_version()
    mlx_version = get_mlx_version()

    # Check for RDMA/JACCL
    rdma_available, rdma_devices = check_rdma_devices()
    tb5_detected = check_thunderbolt5()

    # Check MLX distributed
    mlx_dist_available, world_size = check_mlx_distributed()

    # Check fast sync
    fast_sync = check_fast_sync_support()

    # Load hosts from hostfile if provided
    hosts: list[str] = []
    if hostfile:
        hosts = load_hostfile(hostfile)
    elif os.getenv("MLX_HOSTFILE"):
        hosts = load_hostfile(os.getenv("MLX_HOSTFILE", ""))

    # Determine backend
    if rdma_available:
        backend = "jaccl"
    elif mlx_dist_available:
        backend = "mpi"
    else:
        backend = "none"

    # Determine overall availability
    is_available = mlx_dist_available and world_size > 1

    error_message = None
    if not mlx_version:
        error_message = "MLX not installed"
    elif not mlx_dist_available:
        error_message = "MLX distributed init failed"
    elif world_size <= 1:
        error_message = "Single-node only (world_size=1)"

    return DistributedCapability(
        is_available=is_available,
        rdma_available=rdma_available,
        thunderbolt5_detected=tb5_detected,
        backend=backend,
        detected_hosts=hosts,
        macos_version=macos_version,
        mlx_version=mlx_version,
        fast_sync_supported=fast_sync,
        error_message=error_message,
    )
