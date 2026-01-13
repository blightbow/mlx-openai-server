"""Out-of-band coordination for JACCL send/recv rendezvous.

This module provides rendezvous primitives for JACCL's RDMA data plane.
It solves the timing asymmetry problem that causes SIGBUS crashes when
using mx.distributed.send/recv over JACCL.

IMPORTANT: This module is JACCL-specific. Ring backend doesn't need OOB
(neighbor-only send/recv with implicit sync). MPI has built-in rendezvous.
For non-JACCL backends, skip OOB initialization entirely.

The Problem:
    JACCL's send/recv lacks MPI's rendezvous protocol. When sender and receiver
    have asymmetric timing (e.g., receiver waiting, sender not ready), RDMA
    operations fail with GPU timeout or SIGBUS.

The Solution:
    Use PyTorch's TCPStore as an out-of-band signaling channel:
    - Receiver signals "ready" via TCPStore
    - Sender waits for ready signal, then sends via JACCL
    - Barriers synchronize ranks between transfer phases
    - Natural backpressure, no timing issues

This is the standard "receiver-initiated rendezvous" pattern used in MPI
implementations, but using TCPStore instead of MPI's internal protocols.

Startup Coordination:
    OOB initialization can happen BEFORE mx.distributed.init(). Workers retry
    connecting to rank 0's TCPStore with exponential backoff. Once connected,
    rank 0 is guaranteed to be ready, making JACCL init safe.

Usage:
    # Initialize on all ranks (workers will retry until rank 0 is ready)
    oob = OOBCoordinator(rank, world_size, coordinator_ip, port)

    # Receiver-initiated transfer
    if rank == receiver:
        oob.signal_ready(transfer_id)
        data = mx.distributed.recv_like(template, src=sender, group=jaccl_group)
    elif rank == sender:
        oob.wait_ready(transfer_id, receiver)
        mx.distributed.send(data, dst=receiver, group=jaccl_group)

    # Barrier between transfer phases
    oob.barrier("phase_name")
"""

import os
import random
import time
from typing import Optional

from loguru import logger

# Lazy import to avoid loading torch at module import time
_store: Optional["torch.distributed.TCPStore"] = None


class OOBCoordinator:
    """Out-of-band coordinator for JACCL send/recv rendezvous.

    Provides rendezvous primitives (signal_ready, wait_ready, barrier) that
    work alongside JACCL's RDMA data plane. All coordination happens over TCP,
    completely separate from the RDMA path.

    This coordinator is JACCL-specific. Do not initialize for other backends:
    - Ring: Uses neighbor-only send/recv with implicit synchronization
    - MPI: Has built-in rendezvous protocol

    The TCPStore provides blocking wait() operations, perfect for implementing
    the receiver-initiated rendezvous pattern needed by JACCL.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        host: str,
        port: int = 29400,
        timeout_sec: float = 300.0,
        connect_timeout_sec: float = 300.0,
    ):
        """Initialize the OOB coordinator.

        For rank 0 (master), this starts the TCPStore server immediately.
        For workers (rank > 0), this retries connecting to the master with
        exponential backoff until successful or connect_timeout_sec is reached.

        This allows workers to start before rank 0 - they will simply wait
        until rank 0 is ready, making startup order-independent.

        Args:
            rank: This process's rank (0 = coordinator/master)
            world_size: Total number of processes
            host: IP/hostname of the coordinator (rank 0)
            port: TCP port for the store (default: 29400)
            timeout_sec: Timeout for store operations in seconds (default: 300)
            connect_timeout_sec: Timeout for initial connection attempts (default: 300)
        """
        import torch.distributed as dist

        self.rank = rank
        self.world_size = world_size
        self.host = host
        self.port = port

        # Initialize TCPStore
        # Rank 0 runs the server, others connect as clients
        is_master = rank == 0

        logger.info(
            f"[Rank {rank}] Initializing OOB coordinator "
            f"({'master' if is_master else 'client'}) -> {host}:{port}"
        )

        if is_master:
            # Master starts immediately - no retry needed
            # use_libuv=False avoids IPv6-mapped addresses that cause issues
            # on TB5 links configured with IPv4 only
            self._store = dist.TCPStore(
                host_name=host,
                port=port,
                world_size=world_size,
                is_master=True,
                timeout=dist.timedelta(seconds=timeout_sec),
                use_libuv=False,
            )
        else:
            # Workers retry with exponential backoff until master is ready
            self._store = self._connect_with_retry(
                dist=dist,
                host=host,
                port=port,
                world_size=world_size,
                timeout_sec=timeout_sec,
                connect_timeout_sec=connect_timeout_sec,
            )

        # Track barrier count for unique barrier IDs
        self._barrier_count = 0

        logger.info(f"[Rank {rank}] OOB coordinator initialized")

    def _connect_with_retry(
        self,
        dist,
        host: str,
        port: int,
        world_size: int,
        timeout_sec: float,
        connect_timeout_sec: float,
    ):
        """Connect to TCPStore master with exponential backoff.

        Uses jittered exponential backoff to avoid thundering herd when
        multiple workers start simultaneously before rank 0.

        Args:
            dist: torch.distributed module
            host: Master hostname/IP
            port: Master port
            world_size: Total number of processes
            timeout_sec: Timeout for store operations once connected
            connect_timeout_sec: Total time to spend retrying connection

        Returns:
            Connected TCPStore instance

        Raises:
            TimeoutError: If unable to connect within connect_timeout_sec
        """
        deadline = time.time() + connect_timeout_sec
        base_interval = 0.5  # Start with 500ms
        max_interval = 10.0  # Cap at 10s
        attempt = 0
        warned = False

        while time.time() < deadline:
            try:
                # Short timeout for connection attempt (not operations)
                # use_libuv=False for IPv4-only TB5 links
                store = dist.TCPStore(
                    host_name=host,
                    port=port,
                    world_size=world_size,
                    is_master=False,
                    timeout=dist.timedelta(seconds=min(5.0, timeout_sec)),
                    use_libuv=False,
                )
                if attempt > 0:
                    logger.info(
                        f"[Rank {self.rank}] Connected to OOB master after {attempt} retries"
                    )
                return store

            except Exception as e:
                attempt += 1
                remaining = deadline - time.time()

                if not warned:
                    # First failure: warn user that we're waiting for rank 0
                    logger.warning(
                        f"[Rank {self.rank}] Cannot reach OOB master at {host}:{port} "
                        f"({e.__class__.__name__}). Waiting for rank 0 to start..."
                    )
                    warned = True

                if remaining <= 0:
                    break

                # Exponential backoff with jitter to avoid thundering herd
                # Jitter: randomize between 50-100% of the interval
                interval = min(base_interval * (2 ** min(attempt, 6)), max_interval)
                jittered = interval * (0.5 + random.random() * 0.5)
                sleep_time = min(jittered, remaining)

                logger.debug(
                    f"[Rank {self.rank}] Retry {attempt} in {sleep_time:.1f}s "
                    f"({remaining:.0f}s remaining)"
                )
                time.sleep(sleep_time)

        raise TimeoutError(
            f"[Rank {self.rank}] Could not connect to OOB master at {host}:{port} "
            f"after {connect_timeout_sec}s. Is rank 0 running?"
        )

    def signal_ready(self, transfer_id: str) -> None:
        """Signal that this rank is ready to receive a transfer.

        Call this BEFORE posting the JACCL recv operation.

        Args:
            transfer_id: Unique identifier for this transfer
        """
        key = f"ready_{transfer_id}_rank{self.rank}"
        self._store.set(key, "1")
        logger.debug(f"[Rank {self.rank}] Signaled ready for transfer {transfer_id}")

    def wait_ready(self, transfer_id: str, receiver_rank: int) -> None:
        """Wait for a receiver to signal ready.

        Call this BEFORE sending via JACCL to ensure receiver has posted recv.

        Args:
            transfer_id: Unique identifier for this transfer
            receiver_rank: Rank of the receiver to wait for
        """
        key = f"ready_{transfer_id}_rank{receiver_rank}"
        logger.debug(
            f"[Rank {self.rank}] Waiting for rank {receiver_rank} "
            f"to be ready for transfer {transfer_id}"
        )
        self._store.wait([key])
        logger.debug(
            f"[Rank {self.rank}] Rank {receiver_rank} is ready for transfer {transfer_id}"
        )

    def signal_complete(self, transfer_id: str) -> None:
        """Signal that a transfer has completed.

        Optional - use if sender needs to know when receiver has finished.

        Args:
            transfer_id: Unique identifier for this transfer
        """
        key = f"complete_{transfer_id}_rank{self.rank}"
        self._store.set(key, "1")
        logger.debug(f"[Rank {self.rank}] Signaled complete for transfer {transfer_id}")

    def wait_complete(self, transfer_id: str, sender_rank: int) -> None:
        """Wait for a transfer to complete.

        Args:
            transfer_id: Unique identifier for this transfer
            sender_rank: Rank that performed the send
        """
        key = f"complete_{transfer_id}_rank{sender_rank}"
        self._store.wait([key])

    def barrier(self, name: Optional[str] = None) -> None:
        """Synchronize all ranks.

        All ranks must call this method. Blocks until all ranks have arrived.

        Args:
            name: Optional name for debugging (auto-generated if not provided)
        """
        if name is None:
            name = f"barrier_{self._barrier_count}"
            self._barrier_count += 1

        # Each rank signals arrival
        arrive_key = f"{name}_arrive_rank{self.rank}"
        self._store.set(arrive_key, "1")

        # Wait for all ranks to arrive
        all_keys = [f"{name}_arrive_rank{r}" for r in range(self.world_size)]
        logger.debug(f"[Rank {self.rank}] Barrier {name}: waiting for all ranks")
        self._store.wait(all_keys)
        logger.debug(f"[Rank {self.rank}] Barrier {name}: all ranks arrived")

# Global OOB coordinator instance (initialized lazily)
_oob_coordinator: Optional[OOBCoordinator] = None


def init_oob(
    rank: int,
    world_size: int,
    host: Optional[str] = None,
    port: int = 29400,
) -> OOBCoordinator:
    """Initialize the global OOB coordinator.

    This should be called once at startup, after MLX distributed init.

    Args:
        rank: This process's rank
        world_size: Total number of processes
        host: Coordinator IP/hostname. If None, reads from MLX_OOB_HOST
              or falls back to MLX_JACCL_COORDINATOR.
        port: TCP port (default: 29400)

    Returns:
        The initialized OOBCoordinator instance
    """
    global _oob_coordinator

    if _oob_coordinator is not None:
        logger.warning("[OOB] Coordinator already initialized, returning existing instance")
        return _oob_coordinator

    # Determine host from environment if not provided
    if host is None:
        host = os.environ.get("MLX_OOB_HOST")
        if host is None:
            # Fall back to JACCL coordinator (format: "host:port")
            jaccl_coord = os.environ.get("MLX_JACCL_COORDINATOR", "")
            if ":" in jaccl_coord:
                host = jaccl_coord.split(":")[0]
            else:
                host = jaccl_coord or "127.0.0.1"

    _oob_coordinator = OOBCoordinator(rank, world_size, host, port)
    return _oob_coordinator


def get_oob() -> Optional[OOBCoordinator]:
    """Get the global OOB coordinator instance.

    Returns:
        The OOBCoordinator instance, or None if not initialized
    """
    return _oob_coordinator


def oob_barrier(name: Optional[str] = None) -> None:
    """Convenience function for barrier synchronization.

    Args:
        name: Optional barrier name for debugging
    """
    if _oob_coordinator is None:
        raise RuntimeError("OOB coordinator not initialized. Call init_oob() first.")
    _oob_coordinator.barrier(name)
