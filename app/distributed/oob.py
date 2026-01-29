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
    Use Python's multiprocessing.managers as an out-of-band signaling channel:
    - Receiver signals "ready" via shared dict
    - Sender polls for ready signal, then sends via JACCL
    - Native Barrier synchronizes ranks between transfer phases
    - Natural backpressure, no timing issues

This is the standard "receiver-initiated rendezvous" pattern used in MPI
implementations, but using BaseManager instead of MPI's internal protocols.

Why not PyTorch TCPStore?
    TCPStore has unfixable IPv6 socket issues on macOS causing 35-80+ second
    initialization delays. The root cause is in PyTorch's C++ socket code
    (getaddrinfo called with nullptr), and macOS distributed support is
    explicitly unmaintained (see PyTorch Issue #148440).

    Python's multiprocessing.managers uses explicit IPv4 sockets when given
    a tuple address, avoiding this issue entirely.

Startup Coordination:
    OOB initialization can happen BEFORE mx.distributed.init(). Workers retry
    connecting to rank 0's manager with exponential backoff. Once connected,
    rank 0 is guaranteed to be ready, making JACCL init safe.

Usage:
    # Initialize on all ranks (workers will retry until rank 0 is ready)
    oob = OOBCoordinator(rank, world_size, coordinator_ip, port, authkey)

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
import threading
import time
from multiprocessing.managers import BaseManager, BarrierProxy, DictProxy
from typing import Optional

from loguru import logger

# Module-level shared objects (only rank 0's process uses these directly;
# other ranks access them via manager proxies)
_shared_store: dict = {}
_shared_barrier: Optional[threading.Barrier] = None


class OOBManager(BaseManager):
    """Custom manager for OOB coordination.

    Provides shared dict and barrier accessible across network via proxies.
    """

    pass


# Register shared object accessors with explicit proxy types.
# The callables return module-level objects; proxies expose their methods.
OOBManager.register(
    "get_store",
    callable=lambda: _shared_store,
    proxytype=DictProxy,
    exposed=[
        "__contains__",
        "__delitem__",
        "__getitem__",
        "__setitem__",
        "__len__",
        "clear",
        "get",
        "items",
        "keys",
        "pop",
        "update",
        "values",
    ],
)
OOBManager.register(
    "get_barrier",
    callable=lambda: _shared_barrier,
    proxytype=BarrierProxy,
    exposed=["wait", "abort", "reset", "parties", "n_waiting", "broken"],
)


class OOBCoordinator:
    """Out-of-band coordinator for JACCL send/recv rendezvous.

    Provides rendezvous primitives (signal_ready, wait_ready, barrier) that
    work alongside JACCL's RDMA data plane. All coordination happens over TCP,
    completely separate from the RDMA path.

    This coordinator is JACCL-specific. Do not initialize for other backends:
    - Ring: Uses neighbor-only send/recv with implicit synchronization
    - MPI: Has built-in rendezvous protocol

    Uses Python's multiprocessing.managers for cross-process communication,
    which creates explicit IPv4 sockets (avoiding PyTorch's IPv6 issues).
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        host: str,
        port: int = 29400,
        authkey: bytes = b"oob",
        timeout_sec: float = 300.0,
        connect_timeout_sec: float = 300.0,
    ):
        """Initialize the OOB coordinator.

        For rank 0 (master), this starts the manager server in a background
        thread. For workers (rank > 0), this retries connecting to the master
        with exponential backoff until successful or connect_timeout_sec is
        reached.

        This allows workers to start before rank 0 - they will simply wait
        until rank 0 is ready, making startup order-independent.

        Args:
            rank: This process's rank (0 = coordinator/master)
            world_size: Total number of processes
            host: IP/hostname of the coordinator (rank 0)
            port: TCP port for the manager (default: 29400)
            authkey: Authentication key for manager connections. All ranks must
                use the same key. See derive_authkey() for cluster-specific keys.
            timeout_sec: Timeout for operations in seconds (default: 300)
            connect_timeout_sec: Timeout for initial connection attempts (default: 300)
        """
        global _shared_store, _shared_barrier

        self.rank = rank
        self.world_size = world_size
        self.host = host
        self.port = port
        self._timeout_sec = timeout_sec

        logger.info(
            f"[Rank {rank}] Initializing OOB coordinator "
            f"({'server' if rank == 0 else 'client'}) -> {host}:{port}"
        )

        if rank == 0:
            # Initialize shared objects BEFORE starting server.
            # These live in rank 0's process memory; the manager provides
            # proxies to other ranks.
            _shared_store = {}
            _shared_barrier = threading.Barrier(world_size)

            # Create manager and run server in background thread.
            # Using get_server().serve_forever() instead of start() keeps the
            # server in our process (not spawned), so lambda callables can
            # access the module-level shared objects.
            self._server_manager = OOBManager(
                address=("0.0.0.0", port), authkey=authkey
            )
            server = self._server_manager.get_server()
            self._server_thread = threading.Thread(
                target=server.serve_forever, daemon=True
            )
            self._server_thread.start()
            logger.info(f"[Rank {rank}] OOB server started on port {port}")

            # Small delay to ensure server is listening before we connect
            time.sleep(0.05)

        # ALL ranks (including rank 0) connect as clients.
        # This provides a uniform code path for accessing shared objects.
        if rank == 0:
            # Rank 0 connects to localhost (its own server)
            self._manager = OOBManager(address=("127.0.0.1", port), authkey=authkey)
            self._manager.connect()
        else:
            # Workers connect to rank 0's server with retry
            self._manager = OOBManager(address=(host, port), authkey=authkey)
            self._connect_with_retry(connect_timeout_sec)

        # Get proxies to the shared objects (same objects for all ranks)
        t0 = time.time()
        self._store = self._manager.get_store()
        self._barrier = self._manager.get_barrier()
        logger.debug(f"[Rank {rank}] Got proxies in {time.time() - t0:.3f}s")

        # Synchronize all ranks before returning
        logger.info(f"[Rank {rank}] OOB connected, waiting for all ranks...")
        t0 = time.time()
        self._barrier.wait()
        logger.info(
            f"[Rank {rank}] OOB coordinator initialized (barrier took {time.time()-t0:.2f}s)"
        )

    def _connect_with_retry(self, connect_timeout_sec: float) -> None:
        """Connect to manager server with exponential backoff.

        Uses jittered exponential backoff to avoid thundering herd when
        multiple workers start simultaneously before rank 0.

        Args:
            connect_timeout_sec: Total time to spend retrying connection

        Raises:
            TimeoutError: If unable to connect within connect_timeout_sec
        """
        logger.info(
            f"[Rank {self.rank}] Connecting to OOB server at {self.host}:{self.port} "
            f"(timeout={connect_timeout_sec}s)..."
        )
        deadline = time.time() + connect_timeout_sec
        base_interval = 0.5  # Start with 500ms
        max_interval = 10.0  # Cap at 10s
        attempt = 0
        warned = False

        while time.time() < deadline:
            try:
                self._manager.connect()
                if attempt > 0:
                    logger.info(
                        f"[Rank {self.rank}] Connected to OOB server after {attempt} retries"
                    )
                return

            except Exception as e:
                attempt += 1
                remaining = deadline - time.time()

                if not warned:
                    # First failure: warn user that we're waiting for rank 0
                    logger.warning(
                        f"[Rank {self.rank}] Cannot reach OOB server at {self.host}:{self.port} "
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
            f"[Rank {self.rank}] Could not connect to OOB server at {self.host}:{self.port} "
            f"after {connect_timeout_sec}s. Is rank 0 running?"
        )

    def signal_ready(self, transfer_id: str) -> None:
        """Signal that this rank is ready to receive a transfer.

        Call this BEFORE posting the JACCL recv operation.

        Args:
            transfer_id: Unique identifier for this transfer
        """
        key = f"ready_{transfer_id}_rank{self.rank}"
        self._store[key] = "1"
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
        # Poll until key appears (dict proxy doesn't have blocking wait)
        while key not in self._store:
            time.sleep(0.001)
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
        self._store[key] = "1"
        logger.debug(f"[Rank {self.rank}] Signaled complete for transfer {transfer_id}")

    def wait_complete(self, transfer_id: str, sender_rank: int) -> None:
        """Wait for a transfer to complete.

        Args:
            transfer_id: Unique identifier for this transfer
            sender_rank: Rank that performed the send
        """
        key = f"complete_{transfer_id}_rank{sender_rank}"
        while key not in self._store:
            time.sleep(0.001)

    def barrier(self, name: Optional[str] = None) -> None:
        """Synchronize all ranks.

        All ranks must call this method. Blocks until all ranks have arrived.

        Args:
            name: Optional name for debugging (ignored, kept for API compat)
        """
        # Use native threading.Barrier via proxy - much simpler than key-based
        logger.debug(f"[Rank {self.rank}] Barrier{f' {name}' if name else ''}: waiting")
        self._barrier.wait()
        logger.debug(f"[Rank {self.rank}] Barrier{f' {name}' if name else ''}: passed")


# Global OOB coordinator instance (initialized lazily)
_oob_coordinator: Optional[OOBCoordinator] = None


def init_oob(
    rank: int,
    world_size: int,
    host: Optional[str] = None,
    port: int = 29400,
    authkey: Optional[bytes] = None,
) -> OOBCoordinator:
    """Initialize the global OOB coordinator.

    This should be called once at startup, before MLX distributed init.

    Args:
        rank: This process's rank
        world_size: Total number of processes
        host: Coordinator IP/hostname. If None, reads from MLX_OOB_HOST
              or falls back to MLX_JACCL_COORDINATOR.
        port: TCP port (default: 29400)
        authkey: Authentication key. If None, uses b'oob' (suitable for
                 isolated JACCL networks). Use derive_authkey() for
                 cluster-specific keys when multiple clusters share a network.

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

    # Default authkey if not provided
    if authkey is None:
        authkey = b"oob"

    _oob_coordinator = OOBCoordinator(rank, world_size, host, port, authkey)
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
