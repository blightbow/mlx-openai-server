"""Hostfile loader for direct execution without mlx.launch.

This module parses mlx.launch format hostfiles and sets up JACCL environment
variables, enabling direct process execution without the mlx.launch wrapper.

Workflow:
1. One-time setup: Use `mlx.distributed_config` to generate hostfile
2. Runtime: Each node runs directly with --hostfile and --rank

Example hostfile (cluster.json):
    [
        {"ssh": "mac1.local", "ips": ["192.168.1.10"], "rdma": [null, "rdma_en2"]},
        {"ssh": "mac2.local", "ips": ["192.168.1.11"], "rdma": ["rdma_en2", null]}
    ]

Example usage:
    # Rank 0:
    python -m app.main --hostfile cluster.json --rank 0 --distributed pipeline ...

    # Rank 1:
    python -m app.main --hostfile cluster.json --rank 1 --distributed pipeline ...
"""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class HostConfig:
    """Single host from mlx.launch hostfile format.

    Attributes:
        ssh: SSH hostname (ignored in direct mode, kept for compatibility)
        ips: List of IP addresses (first used for coordinator)
        rdma: RDMA device names per peer (null for self, device name for peers)
    """

    ssh: str
    ips: list[str]
    rdma: list[Optional[str]]


def load_hostfile(path: str) -> list[HostConfig]:
    """Load mlx.launch format hostfile (JSON array).

    Args:
        path: Path to JSON hostfile

    Returns:
        List of HostConfig objects, one per rank (order = rank)

    Raises:
        FileNotFoundError: If hostfile doesn't exist
        ValueError: If hostfile format is invalid
    """
    hostfile_path = Path(path)
    if not hostfile_path.exists():
        raise FileNotFoundError(f"Hostfile not found: {path}")

    with open(hostfile_path, "r") as f:
        try:
            hosts_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in hostfile: {e}")

    if not isinstance(hosts_data, list):
        raise ValueError("Hostfile must be a JSON array of host objects")

    hosts = []
    for i, host in enumerate(hosts_data):
        if not isinstance(host, dict):
            raise ValueError(f"Host {i} must be a JSON object")

        # Required fields
        if "ips" not in host or not host["ips"]:
            raise ValueError(f"Host {i} missing required 'ips' field")
        if "rdma" not in host:
            raise ValueError(f"Host {i} missing required 'rdma' field")

        hosts.append(
            HostConfig(
                ssh=host.get("ssh", f"host{i}"),
                ips=host["ips"],
                rdma=host["rdma"],
            )
        )

    if len(hosts) < 2:
        raise ValueError("Hostfile must contain at least 2 hosts for distributed mode")

    # Validate RDMA matrix dimensions
    world_size = len(hosts)
    for i, host in enumerate(hosts):
        if len(host.rdma) != world_size:
            raise ValueError(
                f"Host {i} rdma array has {len(host.rdma)} entries, expected {world_size}"
            )

    return hosts


def setup_jaccl_env(
    hosts: list[HostConfig],
    rank: int,
    port: int = 32323,
) -> str:
    """Set JACCL environment variables for direct execution.

    Sets:
        MLX_RANK: Process rank
        MLX_WORLD_SIZE: Total number of processes
        MLX_JACCL_COORDINATOR: ip:port of rank 0
        MLX_IBV_DEVICES: Path to temp file with RDMA device mappings

    Args:
        hosts: List of host configs from load_hostfile()
        rank: This process's rank
        port: Coordinator port (default: 32323)

    Returns:
        Path to temporary IBV_DEVICES file (caller should not delete)

    Raises:
        ValueError: If rank is out of range or coordinator has no IPs
    """
    world_size = len(hosts)

    if rank < 0 or rank >= world_size:
        raise ValueError(f"Rank {rank} out of range [0, {world_size})")

    # Coordinator is rank 0's first IP
    if not hosts[0].ips:
        raise ValueError("Rank 0 host has no IP addresses")
    coordinator_ip = hosts[0].ips[0]

    # Set MLX_RANK and MLX_WORLD_SIZE
    os.environ["MLX_RANK"] = str(rank)
    os.environ["MLX_WORLD_SIZE"] = str(world_size)

    # Enable fast Metal synchronization for JACCL (macOS 15+/Metal 3.2+).
    # Critical for low-latency RDMA - reduces CPU-GPU sync overhead.
    os.environ["MLX_METAL_FAST_SYNCH"] = "1"

    # Set MLX_JACCL_COORDINATOR
    coordinator_addr = f"{coordinator_ip}:{port}"
    os.environ["MLX_JACCL_COORDINATOR"] = coordinator_addr

    # Create IBV_DEVICES temp file with RDMA mappings
    # Format: JSON array of arrays, one per rank, listing RDMA devices to each peer
    ibv_devices = [host.rdma for host in hosts]
    ibv_json = json.dumps(ibv_devices)

    # Write to temp file (don't delete - JACCL reads it during init)
    ibv_file = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="mlx_ibv_devices_",
        suffix=".json",
        delete=False,
    )
    ibv_file.write(ibv_json)
    ibv_file.close()

    os.environ["MLX_IBV_DEVICES"] = ibv_file.name

    logger.info(
        f"[Rank {rank}] JACCL env configured: "
        f"world_size={world_size}, "
        f"coordinator={coordinator_addr}, "
        f"ibv_devices={ibv_file.name}, "
        f"fast_synch=1"
    )

    return ibv_file.name


def get_oob_host_from_hostfile(hosts: list[HostConfig]) -> str:
    """Extract OOB coordinator host from hostfile.

    OOB coordinator runs on same host as JACCL coordinator (rank 0).

    Args:
        hosts: List of host configs from load_hostfile()

    Returns:
        IP address for OOB coordinator (rank 0's first IP)
    """
    if not hosts or not hosts[0].ips:
        raise ValueError("Cannot determine OOB host: rank 0 has no IP addresses")
    return hosts[0].ips[0]
