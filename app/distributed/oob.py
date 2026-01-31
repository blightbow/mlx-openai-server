"""Out-of-band coordination for JACCL send/recv rendezvous using ZeroMQ.

This module provides rendezvous primitives for JACCL's RDMA data plane using
ZeroMQ (pyzmq) sockets. It solves the timing asymmetry problem that causes
SIGBUS crashes when using mx.distributed.send/recv over JACCL.

IMPORTANT: This module is JACCL-specific. Ring backend doesn't need OOB
(neighbor-only send/recv with implicit sync). MPI has built-in rendezvous.
For non-JACCL backends, skip OOB initialization entirely.

Architecture:
    ZeroMQ provides socket patterns for coordination:
    - REQ/REP: Barrier arrivals (count-based, immediate release)
    - REQ/REP: Key-value store operations
    - PUB/SUB: Broadcast (barrier release, termination, abort)

    All operations are async-first, cancellable via asyncio.CancelledError.

Socket Topology:
    Rank 0 (Coordinator):
      - REP socket: tcp://0.0.0.0:29400 (bind) - barrier arrivals
      - REP socket: tcp://0.0.0.0:29401 (bind) - store operations
      - REP socket: tcp://0.0.0.0:29402 (bind) - termination relay
      - PUB socket: tcp://0.0.0.0:29403 (bind) - broadcast

    Workers (Rank 1+):
      - REQ socket: tcp://{host}:29400 (connect) - barrier
      - REQ socket: tcp://{host}:29401 (connect) - store
      - REQ socket: tcp://{host}:29402 (connect) - termination
      - SUB socket: tcp://{host}:29403 (connect) - broadcast

Why ZeroMQ over pynng?
    Research showed pynng has ~10µs jitter vs ZeroMQ's ~1µs. Given JACCL RDMA
    latency of 5-9µs, pynng's jitter is comparable to RDMA operation time itself.
    Additionally, pynng's SURVEY pattern is time-based (waits for timeout) while
    ZeroMQ's REQ/REP enables count-based barriers (immediate release when all
    ranks arrive).

Usage:
    # Initialize on all ranks (async context required)
    async with OOBCoordinator(rank, world_size, coordinator_ip, port) as oob:
        # Barrier synchronization
        await oob.barrier("phase_name")

        # Receiver-initiated transfer
        if rank == receiver:
            await oob.signal_ready(transfer_id)
            data = mx.distributed.recv_like(template, src=sender, group=jaccl_group)
        elif rank == sender:
            await oob.wait_ready(transfer_id, receiver)
            mx.distributed.send(data, dst=receiver, group=jaccl_group)
"""

import asyncio
import os
import time
from typing import Optional, Set

import zmq
import zmq.asyncio
from loguru import logger


class PeerTimeoutError(Exception):
    """Raised when an OOB wait operation times out.

    This exception indicates that an OOB coordination operation (barrier,
    wait_ready, wait_complete) timed out waiting for a peer. This typically
    means a peer has crashed or become unresponsive without signaling
    termination.
    """

    pass


class PeerTerminatedError(Exception):
    """Raised when a peer has signaled termination during a collective operation.

    This exception indicates that a distributed peer has signaled it is
    terminating, and the collective operation should be aborted to prevent
    RDMA operations against a dead peer.
    """

    pass


class AbortError(Exception):
    """Raised when an abort signal is received.

    This exception indicates that an external abort signal was broadcast,
    and all pending operations should terminate.
    """

    pass


class BarrierHandle:
    """Handle for non-blocking barrier completion tracking (Ibarrier pattern).

    This allows checking barrier completion without blocking, similar to
    MPI_Ibarrier + MPI_Test pattern.
    """

    def __init__(self) -> None:
        """Initialize barrier handle."""
        self._complete = False
        self._event = asyncio.Event()
        self._cancelled = False

    def is_complete(self) -> bool:
        """Non-blocking check if all ranks arrived.

        Returns:
            True if all ranks have arrived at the barrier
        """
        return self._complete

    async def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for barrier completion.

        Args:
            timeout: Timeout in seconds, or None for no timeout

        Raises:
            PeerTimeoutError: If timeout expires before completion
            asyncio.CancelledError: If wait is cancelled
        """
        if self._complete:
            return

        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise PeerTimeoutError(f"Barrier wait timed out after {timeout}s") from None

    def cancel(self) -> None:
        """Cancel waiting - barrier remains usable for others."""
        self._cancelled = True

    def _mark_complete(self) -> None:
        """Mark barrier as complete (called by coordinator)."""
        self._complete = True
        self._event.set()


class OOBCoordinator:
    """ZeroMQ-based async OOB coordinator for JACCL.

    Uses socket patterns:
    - REQ/REP for barrier (count-based immediate release)
    - REQ/REP for key-value store
    - PUB/SUB for broadcast (barrier release, termination, abort)

    All operations are async. Cancellable via asyncio.CancelledError.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        host: str,
        port: int = 29400,
        timeout_sec: float = 300.0,
    ) -> None:
        """Initialize the OOB coordinator.

        Note: Call start() or use as async context manager to begin operation.

        Args:
            rank: This process's rank (0 = coordinator/master)
            world_size: Total number of processes
            host: IP/hostname of the coordinator (rank 0)
            port: Base TCP port for sockets (default: 29400)
                  Uses port, port+1, port+2, port+3 for barrier, store, term, broadcast
            timeout_sec: Default timeout for operations (default: 300)
        """
        self.rank = rank
        self.world_size = world_size
        self.host = host
        self.port = port
        self._timeout_sec = timeout_sec

        # ZeroMQ context and sockets (created in start())
        self._ctx: Optional[zmq.asyncio.Context] = None

        # Coordinator sockets (rank 0 only)
        self._barrier_rep: Optional[zmq.asyncio.Socket] = None
        self._store_rep: Optional[zmq.asyncio.Socket] = None
        self._term_rep: Optional[zmq.asyncio.Socket] = None
        self._pub: Optional[zmq.asyncio.Socket] = None

        # Worker sockets (all ranks)
        self._barrier_req: Optional[zmq.asyncio.Socket] = None
        self._store_req: Optional[zmq.asyncio.Socket] = None
        self._term_req: Optional[zmq.asyncio.Socket] = None
        self._sub: Optional[zmq.asyncio.Socket] = None

        # State
        self._store: dict[str, str] = {}
        self._terminating_ranks: Set[int] = set()
        self._abort_flag = False
        self._shutdown = False

        # Background tasks
        self._tasks: list[asyncio.Task[None]] = []

        # Barrier state for coordinator
        self._barrier_waiters: dict[str, list[bytes]] = {}  # name -> list of rank identities
        self._barrier_events: dict[str, asyncio.Event] = {}

        # Lock for serializing store operations (REQ/REP requires strict alternation)
        self._store_lock = asyncio.Lock()

        logger.info(
            f"[Rank {rank}] OOB coordinator created "
            f"(host={host}, port={port}, world_size={world_size})"
        )

    async def __aenter__(self) -> "OOBCoordinator":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Async context manager exit."""
        await self.stop()

    async def start(self) -> None:
        """Start the OOB coordinator.

        Creates sockets and starts background tasks.
        """
        logger.info(f"[Rank {self.rank}] Starting OOB coordinator...")

        self._ctx = zmq.asyncio.Context()

        if self.rank == 0:
            await self._start_coordinator()
        else:
            await self._start_worker()

        # All ranks subscribe to broadcast
        await self._start_subscriber()

        # Start background tasks
        if self.rank == 0:
            self._tasks.append(asyncio.create_task(self._barrier_server_loop()))
            self._tasks.append(asyncio.create_task(self._store_server_loop()))
            self._tasks.append(asyncio.create_task(self._term_relay_loop()))

        self._tasks.append(asyncio.create_task(self._broadcast_listener()))

        # Give background tasks a chance to start their first iteration
        for _ in range(3):
            await asyncio.sleep(0.05)

        logger.debug(f"[Rank {self.rank}] Background tasks started")

        # Wait for all ranks to connect (simple barrier via store)
        await self._startup_barrier()

        logger.info(f"[Rank {self.rank}] OOB coordinator started")

    async def _start_coordinator(self) -> None:
        """Start coordinator sockets (rank 0 only)."""
        # Barrier arrivals (REP)
        self._barrier_rep = self._ctx.socket(zmq.REP)
        self._configure_socket(self._barrier_rep)
        self._barrier_rep.bind(f"tcp://0.0.0.0:{self.port}")

        # Store operations (REP)
        self._store_rep = self._ctx.socket(zmq.REP)
        self._configure_socket(self._store_rep)
        self._store_rep.bind(f"tcp://0.0.0.0:{self.port + 1}")

        # Termination relay (REP)
        self._term_rep = self._ctx.socket(zmq.REP)
        self._configure_socket(self._term_rep)
        self._term_rep.bind(f"tcp://0.0.0.0:{self.port + 2}")

        # Broadcast (PUB)
        self._pub = self._ctx.socket(zmq.PUB)
        self._configure_socket(self._pub)
        self._pub.bind(f"tcp://0.0.0.0:{self.port + 3}")

        logger.debug(f"[Rank 0] Coordinator sockets bound on ports {self.port}-{self.port + 3}")

    async def _start_worker(self) -> None:
        """Start worker sockets (rank > 0)."""
        # Barrier (REQ)
        self._barrier_req = self._ctx.socket(zmq.REQ)
        self._configure_socket(self._barrier_req)
        self._barrier_req.connect(f"tcp://{self.host}:{self.port}")

        # Store (REQ)
        self._store_req = self._ctx.socket(zmq.REQ)
        self._configure_socket(self._store_req)
        self._store_req.connect(f"tcp://{self.host}:{self.port + 1}")

        # Termination (REQ)
        self._term_req = self._ctx.socket(zmq.REQ)
        self._configure_socket(self._term_req)
        self._term_req.connect(f"tcp://{self.host}:{self.port + 2}")

        logger.debug(f"[Rank {self.rank}] Worker sockets connected to {self.host}:{self.port}")

    async def _start_subscriber(self) -> None:
        """Start broadcast subscriber (all ranks)."""
        self._sub = self._ctx.socket(zmq.SUB)
        self._configure_socket(self._sub)

        # Subscribe to all messages
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")

        if self.rank == 0:
            # Rank 0 connects to its own publisher via localhost
            self._sub.connect(f"tcp://127.0.0.1:{self.port + 3}")
        else:
            self._sub.connect(f"tcp://{self.host}:{self.port + 3}")

        logger.debug(f"[Rank {self.rank}] Subscribed to broadcast")

    def _configure_socket(self, sock: zmq.asyncio.Socket) -> None:
        """Configure socket for low latency and graceful shutdown."""
        # Bounded graceful shutdown (100ms max wait)
        sock.setsockopt(zmq.LINGER, 100)

        # Small buffers for latency over throughput
        sock.setsockopt(zmq.SNDBUF, 4096)
        sock.setsockopt(zmq.RCVBUF, 4096)

    async def _startup_barrier(self) -> None:
        """Startup synchronization using store + barrier.

        Uses three-phase handshake to ensure all ranks complete before any returns:
        1. Phase 1: Each rank sets presence key, waits for all presence keys
        2. Phase 2: Each rank sets complete key, waits for all complete keys
        3. Phase 3: Actual barrier to sync before exit

        Phase 3 is critical because phases 1-2 have an asymmetry: Rank 0's store
        checks are local (instant) while workers go through sockets. Without
        phase 3, Rank 0 can exit and block the event loop (e.g., in
        mx.distributed.init) while workers are still waiting for socket responses.

        Phase 3 uses the barrier mechanism which keeps Rank 0 yielding while
        waiting for worker arrivals, allowing the store server to respond.
        """
        # Phase 1: Signal presence and wait for all ranks
        key = f"startup_rank{self.rank}"
        await self._store_set(key, "1")

        deadline = time.time() + self._timeout_sec
        while True:
            # Yield to let store server process requests from other ranks
            await asyncio.sleep(0.05)

            all_present = True
            for r in range(self.world_size):
                if not await self._store_exists(f"startup_rank{r}"):
                    all_present = False
                    break

            if all_present:
                break

            if time.time() >= deadline:
                raise PeerTimeoutError(
                    f"[Rank {self.rank}] Startup barrier timed out after {self._timeout_sec}s"
                )

        logger.debug(f"[Rank {self.rank}] Startup phase 1 complete")

        # Phase 2: Signal completion and wait for all ranks to complete
        # This ensures no rank returns until all have finished phase 1
        await self._store_set(f"startup_complete_rank{self.rank}", "1")

        deadline = time.time() + self._timeout_sec
        while True:
            # Yield to let store server process requests
            await asyncio.sleep(0.05)

            all_complete = True
            for r in range(self.world_size):
                if not await self._store_exists(f"startup_complete_rank{r}"):
                    all_complete = False
                    break

            if all_complete:
                break

            if time.time() >= deadline:
                raise PeerTimeoutError(
                    f"[Rank {self.rank}] Startup completion sync timed out"
                )

        logger.debug(f"[Rank {self.rank}] Startup phase 2 complete")

        # Phase 3: Use actual barrier mechanism for final sync
        # This is critical: the barrier coordinator (rank 0) yields while waiting
        # for worker arrivals, allowing the store server to respond to any
        # pending requests from workers still finishing phase 2.
        await self.barrier("startup_final", timeout=self._timeout_sec)
        logger.debug(f"[Rank {self.rank}] Startup phase 3 (barrier) complete")

    async def stop(self) -> None:
        """Stop the OOB coordinator gracefully."""
        logger.info(f"[Rank {self.rank}] Stopping OOB coordinator...")

        self._shutdown = True

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

        # Close sockets
        sockets = [
            self._barrier_rep,
            self._store_rep,
            self._term_rep,
            self._pub,
            self._barrier_req,
            self._store_req,
            self._term_req,
            self._sub,
        ]

        for sock in sockets:
            if sock is not None:
                sock.close()

        # Terminate context
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None

        logger.info(f"[Rank {self.rank}] OOB coordinator stopped")

    # --- Barrier (Count-based with broadcast release) ---

    async def barrier(self, name: str, timeout: Optional[float] = None) -> None:
        """Synchronize all ranks with count-based barrier.

        Unlike time-based barriers (pynng SURVEY), this releases immediately
        when all ranks arrive, providing ~1µs jitter instead of ~10µs.

        Args:
            name: Barrier name for debugging
            timeout: Timeout in seconds (default: self._timeout_sec)

        Raises:
            PeerTimeoutError: If timeout expires before all ranks arrive
            PeerTerminatedError: If peer signals termination
            AbortError: If abort signal received
        """
        if timeout is None:
            timeout = self._timeout_sec

        self._check_abort_or_termination(f"barrier {name}")

        logger.debug(f"[Rank {self.rank}] Entering barrier '{name}' (timeout={timeout}s)")

        if self.rank == 0:
            await self._barrier_coordinator(name, timeout)
        else:
            await self._barrier_worker(name, timeout)

        logger.debug(f"[Rank {self.rank}] Passed barrier '{name}'")

    async def _barrier_coordinator(self, name: str, timeout: float) -> None:
        """Coordinator side of barrier - collect arrivals and broadcast release."""
        # Coordinator counts as arrived
        arrived = 1
        needed = self.world_size
        deadline = time.time() + timeout

        # Wait for all workers to arrive
        while arrived < needed:
            self._check_abort_or_termination(f"barrier {name}")

            remaining = deadline - time.time()
            if remaining <= 0:
                raise PeerTimeoutError(
                    f"[Rank 0] Barrier '{name}' timed out after {timeout}s "
                    f"({arrived}/{needed} arrived)"
                )

            # The barrier server loop handles arrivals and signals us
            await asyncio.sleep(0.01)

            # Check how many arrived for this barrier
            arrived = 1 + len(self._barrier_waiters.get(name, []))

        # All arrived - broadcast release
        release_msg = f"RELEASE:{name}".encode()
        await self._pub.send(release_msg)
        logger.debug(f"[Rank 0] Barrier '{name}' released ({needed} ranks)")

        # Clean up
        self._barrier_waiters.pop(name, None)

    async def _barrier_worker(self, name: str, timeout: float) -> None:
        """Worker side of barrier - signal arrival and wait for release."""
        # Signal arrival to coordinator
        arrival_msg = f"ARRIVE:{name}:{self.rank}".encode()

        try:
            await asyncio.wait_for(self._barrier_req.send(arrival_msg), timeout=timeout)
            await asyncio.wait_for(self._barrier_req.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            raise PeerTimeoutError(
                f"[Rank {self.rank}] Barrier '{name}' arrival timeout after {timeout}s"
            ) from None

        # Wait for broadcast release
        deadline = time.time() + timeout
        release_msg = f"RELEASE:{name}".encode()

        while True:
            self._check_abort_or_termination(f"barrier {name}")

            remaining = deadline - time.time()
            if remaining <= 0:
                raise PeerTimeoutError(
                    f"[Rank {self.rank}] Barrier '{name}' release timeout after {timeout}s"
                )

            # Check if we received the release (set by broadcast listener)
            if name in self._barrier_events and self._barrier_events[name].is_set():
                self._barrier_events.pop(name, None)
                return

            await asyncio.sleep(0.001)

    async def _barrier_server_loop(self) -> None:
        """Coordinator: Handle barrier arrival requests."""
        logger.debug("[Rank 0] Barrier server loop starting")

        # Create poller for more reliable async receive
        poller = zmq.asyncio.Poller()
        poller.register(self._barrier_rep, zmq.POLLIN)

        while not self._shutdown:
            try:
                # Poll with 100ms timeout (returns list of (socket, event) tuples)
                events = dict(await poller.poll(timeout=100))

                if self._barrier_rep in events:
                    msg = await self._barrier_rep.recv(zmq.NOBLOCK)
                    parts = msg.decode().split(":")

                    if parts[0] == "ARRIVE" and len(parts) >= 3:
                        name = parts[1]
                        rank = int(parts[2])

                        if name not in self._barrier_waiters:
                            self._barrier_waiters[name] = []
                        self._barrier_waiters[name].append(rank)

                        # Acknowledge arrival
                        await self._barrier_rep.send(b"OK")
                        logger.debug(
                            f"[Rank 0] Barrier '{name}': rank {rank} arrived "
                            f"({len(self._barrier_waiters[name]) + 1}/{self.world_size})"
                        )
                    else:
                        await self._barrier_rep.send(b"ERR:UNKNOWN")

            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._shutdown:
                    logger.error(f"[Rank 0] Barrier server error: {e}", exc_info=True)

    # --- Store Operations (REQ/REP pattern) ---

    async def _store_server_loop(self) -> None:
        """Coordinator: Handle store requests from workers."""
        logger.debug("[Rank 0] Store server loop starting")
        msg_count = 0
        while not self._shutdown:
            try:
                msg = await asyncio.wait_for(self._store_rep.recv(), timeout=0.1)
                msg_count += 1
                parts = msg.decode().split(":", 2)
                cmd = parts[0]
                logger.info(f"[Rank 0] Store server received msg #{msg_count}: {cmd} (parts={len(parts)})")

                if cmd == "SET" and len(parts) >= 3:
                    key, value = parts[1], parts[2]
                    self._store[key] = value
                    logger.info(f"[Rank 0] Store server SET {key}={value}, sending OK")
                    await self._store_rep.send(b"OK")
                    logger.debug(f"[Rank 0] Store server SET {key} OK sent")

                elif cmd == "GET" and len(parts) >= 2:
                    key = parts[1]
                    value = self._store.get(key, "")
                    await self._store_rep.send(value.encode())

                elif cmd == "EXISTS" and len(parts) >= 2:
                    key = parts[1]
                    exists = b"1" if key in self._store else b"0"
                    await self._store_rep.send(exists)

                elif cmd == "DEL" and len(parts) >= 2:
                    key = parts[1]
                    self._store.pop(key, None)
                    await self._store_rep.send(b"OK")

                elif cmd == "KEYS":
                    # Return all keys (for debugging)
                    keys = "\n".join(sorted(self._store.keys()))
                    await self._store_rep.send(keys.encode())

                elif cmd == "DUMP":
                    # Dump all key-value pairs (for debugging)
                    lines = [f"{k}={v}" for k, v in sorted(self._store.items())]
                    await self._store_rep.send("\n".join(lines).encode())

                elif cmd == "STATUS":
                    # Return coordinator status for external debugging
                    status = (
                        f"rank=0\n"
                        f"world_size={self.world_size}\n"
                        f"shutdown={self._shutdown}\n"
                        f"abort={self._abort_flag}\n"
                        f"terminating_ranks={','.join(map(str, self._terminating_ranks))}\n"
                        f"pending_barriers={','.join(self._barrier_waiters.keys())}\n"
                        f"store_keys={len(self._store)}"
                    )
                    await self._store_rep.send(status.encode())

                else:
                    await self._store_rep.send(b"ERR:UNKNOWN")

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._shutdown:
                    logger.error(f"[Rank 0] Store server error: {e}")

    async def _store_set(self, key: str, value: str) -> None:
        """Set a key in the store."""
        if self.rank == 0:
            self._store[key] = value
            logger.debug(f"[Rank 0] Store SET local: {key}={value}")
        else:
            logger.info(f"[Rank {self.rank}] Store SET acquiring lock for {key}")
            async with self._store_lock:
                msg = f"SET:{key}:{value}".encode()
                logger.info(f"[Rank {self.rank}] Store SET sending: {key}")
                await self._store_req.send(msg)
                logger.info(f"[Rank {self.rank}] Store SET waiting for reply: {key}")
                await self._store_req.recv()
                logger.info(f"[Rank {self.rank}] Store SET complete: {key}")

    async def _store_get(self, key: str) -> str:
        """Get a value from the store."""
        if self.rank == 0:
            return self._store.get(key, "")
        else:
            async with self._store_lock:
                msg = f"GET:{key}".encode()
                await self._store_req.send(msg)
                data = await self._store_req.recv()
                return data.decode()

    async def _store_exists(self, key: str) -> bool:
        """Check if a key exists in the store."""
        if self.rank == 0:
            return key in self._store
        else:
            async with self._store_lock:
                msg = f"EXISTS:{key}".encode()
                await self._store_req.send(msg)
                data = await self._store_req.recv()
                return data == b"1"

    # --- Ready/Complete Signaling ---

    async def signal_ready(self, transfer_id: str) -> None:
        """Signal that this rank is ready to receive a transfer.

        Call this BEFORE posting the JACCL recv operation.

        Args:
            transfer_id: Unique identifier for this transfer
        """
        key = f"ready_{transfer_id}_rank{self.rank}"
        await self._store_set(key, "1")
        logger.debug(f"[Rank {self.rank}] Signaled ready for transfer {transfer_id}")

    async def wait_ready(
        self,
        transfer_id: str,
        receiver_rank: int,
        timeout: Optional[float] = None,
    ) -> None:
        """Wait for a receiver to signal ready.

        Call this BEFORE sending via JACCL to ensure receiver has posted recv.

        Args:
            transfer_id: Unique identifier for this transfer
            receiver_rank: Rank of the receiver to wait for
            timeout: Timeout in seconds (default: self._timeout_sec)

        Raises:
            PeerTimeoutError: If timeout expires before receiver signals ready
            PeerTerminatedError: If peer signals termination during wait
            AbortError: If abort signal received
        """
        if timeout is None:
            timeout = self._timeout_sec

        key = f"ready_{transfer_id}_rank{receiver_rank}"
        logger.info(
            f"[Rank {self.rank}] wait_ready: waiting for rank {receiver_rank} "
            f"ready signal for {transfer_id}"
        )

        deadline = time.time() + timeout
        poll_interval = 0.01  # 10ms between checks
        check_count = 0

        while not await self._store_exists(key):
            check_count += 1
            if check_count % 100 == 0:  # Log every 100 checks (~1 second)
                logger.info(f"[Rank {self.rank}] wait_ready: still waiting for {key} (check #{check_count})")
            self._check_abort_or_termination(f"wait_ready {transfer_id}")

            if time.time() >= deadline:
                raise PeerTimeoutError(
                    f"[Rank {self.rank}] Timeout ({timeout}s) waiting for "
                    f"rank {receiver_rank} ready signal for {transfer_id}"
                )

            await asyncio.sleep(poll_interval)

        logger.debug(
            f"[Rank {self.rank}] Rank {receiver_rank} is ready for transfer {transfer_id}"
        )

    async def signal_complete(self, transfer_id: str) -> None:
        """Signal that a transfer has completed.

        Optional - use if sender needs to know when receiver has finished.

        Args:
            transfer_id: Unique identifier for this transfer
        """
        key = f"complete_{transfer_id}_rank{self.rank}"
        await self._store_set(key, "1")
        logger.debug(f"[Rank {self.rank}] Signaled complete for transfer {transfer_id}")

    async def wait_complete(
        self,
        transfer_id: str,
        sender_rank: int,
        timeout: Optional[float] = None,
    ) -> None:
        """Wait for a transfer to complete.

        Args:
            transfer_id: Unique identifier for this transfer
            sender_rank: Rank that performed the send
            timeout: Timeout in seconds (default: self._timeout_sec)

        Raises:
            PeerTimeoutError: If timeout expires before sender signals complete
            PeerTerminatedError: If peer signals termination during wait
            AbortError: If abort signal received
        """
        if timeout is None:
            timeout = self._timeout_sec

        key = f"complete_{transfer_id}_rank{sender_rank}"
        logger.debug(
            f"[Rank {self.rank}] Waiting for rank {sender_rank} "
            f"to complete transfer {transfer_id} (timeout={timeout}s)"
        )

        deadline = time.time() + timeout
        poll_interval = 0.01

        while not await self._store_exists(key):
            self._check_abort_or_termination(f"wait_complete {transfer_id}")

            if time.time() >= deadline:
                raise PeerTimeoutError(
                    f"[Rank {self.rank}] Timeout ({timeout}s) waiting for "
                    f"rank {sender_rank} complete signal for {transfer_id}"
                )

            await asyncio.sleep(poll_interval)

        logger.debug(
            f"[Rank {self.rank}] Rank {sender_rank} completed transfer {transfer_id}"
        )

    # --- Termination/Abort (PUB/SUB pattern) ---

    async def _term_relay_loop(self) -> None:
        """Coordinator: Relay termination signals from workers to broadcast."""
        while not self._shutdown:
            try:
                msg = await asyncio.wait_for(self._term_rep.recv(), timeout=0.1)

                if msg.startswith(b"TERM:"):
                    # Relay to broadcast
                    await self._pub.send(msg)
                    await self._term_rep.send(b"OK")
                else:
                    await self._term_rep.send(b"ERR:UNKNOWN")

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._shutdown:
                    logger.error(f"[Rank 0] Term relay error: {e}")

    async def _broadcast_listener(self) -> None:
        """Listen for broadcast messages (barrier release, termination, abort)."""
        while not self._shutdown:
            try:
                msg = await asyncio.wait_for(self._sub.recv(), timeout=0.1)

                if msg.startswith(b"RELEASE:"):
                    # Barrier release
                    name = msg.decode().split(":", 1)[1]
                    if name not in self._barrier_events:
                        self._barrier_events[name] = asyncio.Event()
                    self._barrier_events[name].set()

                elif msg.startswith(b"TERM:"):
                    # Termination signal
                    rank = int(msg.decode().split(":")[1])
                    self._terminating_ranks.add(rank)
                    logger.info(f"[Rank {self.rank}] Received termination from rank {rank}")

                elif msg.startswith(b"ABORT"):
                    # Abort signal
                    self._abort_flag = True
                    logger.warning(f"[Rank {self.rank}] Received abort signal")

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._shutdown:
                    logger.debug(f"[Rank {self.rank}] Broadcast listener: {e}")

    async def signal_terminating(self) -> None:
        """Signal that this rank is terminating.

        Call this during shutdown to notify other ranks that they should stop
        RDMA operations.
        """
        msg = f"TERM:{self.rank}".encode()

        try:
            if self.rank == 0:
                # Coordinator broadcasts directly
                await self._pub.send(msg)
            else:
                # Workers send to coordinator for relay
                await self._term_req.send(msg)
                await self._term_req.recv()

            self._terminating_ranks.add(self.rank)
            logger.info(f"[Rank {self.rank}] Signaled termination to peers")

        except Exception as e:
            logger.debug(f"[Rank {self.rank}] Could not signal termination: {e}")

    def is_any_peer_terminating(self) -> bool:
        """Check if any peer has signaled termination (sync check).

        Returns:
            True if any other rank has signaled it is terminating
        """
        for r in self._terminating_ranks:
            if r != self.rank:
                return True
        return False

    async def abort(self) -> None:
        """Broadcast abort. All pending ops raise AbortError."""
        if self.rank == 0 and self._pub is not None:
            await self._pub.send(b"ABORT")
            self._abort_flag = True
            logger.warning(f"[Rank {self.rank}] Broadcast abort signal")

    def _check_abort_or_termination(self, operation: str) -> None:
        """Check for abort/termination and raise appropriate exception."""
        if self._abort_flag:
            raise AbortError(f"Abort during {operation}")
        if self.is_any_peer_terminating():
            raise PeerTerminatedError(f"Peer terminated during {operation}")

    def check_peers_alive(self) -> bool:
        """Check if OOB connection is still alive.

        Returns:
            True if connection appears healthy
        """
        return not self._shutdown and not self._abort_flag


# Global OOB coordinator instance (initialized lazily)
_oob_coordinator: Optional[OOBCoordinator] = None
_oob_lock = asyncio.Lock()


async def init_oob(
    rank: int,
    world_size: int,
    host: Optional[str] = None,
    port: int = 29400,
) -> OOBCoordinator:
    """Initialize the global OOB coordinator.

    This should be called once at startup, before MLX distributed init.

    Args:
        rank: This process's rank
        world_size: Total number of processes
        host: Coordinator IP/hostname. If None, reads from MLX_OOB_HOST
              or falls back to MLX_JACCL_COORDINATOR.
        port: Base TCP port (default: 29400)

    Returns:
        The initialized OOBCoordinator instance
    """
    global _oob_coordinator

    async with _oob_lock:
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
        await _oob_coordinator.start()
        return _oob_coordinator


async def shutdown_oob() -> None:
    """Shutdown the global OOB coordinator."""
    global _oob_coordinator

    async with _oob_lock:
        if _oob_coordinator is not None:
            await _oob_coordinator.stop()
            _oob_coordinator = None


def get_oob() -> Optional[OOBCoordinator]:
    """Get the global OOB coordinator instance.

    Returns:
        The OOBCoordinator instance, or None if not initialized
    """
    return _oob_coordinator


async def oob_barrier(name: Optional[str] = None, timeout: Optional[float] = None) -> None:
    """Convenience function for barrier synchronization.

    Args:
        name: Optional barrier name for debugging
        timeout: Optional timeout override
    """
    if _oob_coordinator is None:
        raise RuntimeError("OOB coordinator not initialized. Call init_oob() first.")
    await _oob_coordinator.barrier(name or "unnamed", timeout)


# Sync wrappers for compatibility with existing code paths
def oob_barrier_sync(name: Optional[str] = None, timeout: Optional[float] = None) -> None:
    """Synchronous barrier wrapper for non-async code paths.

    Creates a new event loop if needed. Prefer async version where possible.

    Args:
        name: Optional barrier name
        timeout: Optional timeout override
    """
    if _oob_coordinator is None:
        raise RuntimeError("OOB coordinator not initialized. Call init_oob() first.")

    try:
        loop = asyncio.get_running_loop()
        # Already in async context - this would deadlock
        raise RuntimeError(
            "oob_barrier_sync called from async context. Use 'await oob_barrier()' instead."
        )
    except RuntimeError as e:
        if "no running event loop" not in str(e).lower():
            raise
        # No running loop - safe to use run
        asyncio.run(_oob_coordinator.barrier(name or "unnamed", timeout))


def _run_sync(coro):
    """Run a coroutine synchronously, handling event loop presence."""
    try:
        loop = asyncio.get_running_loop()
        raise RuntimeError(
            "Sync wrapper called from async context. Use async version instead."
        )
    except RuntimeError as e:
        if "no running event loop" not in str(e).lower():
            raise
        return asyncio.run(coro)


def oob_signal_ready_sync(transfer_id: str) -> None:
    """Synchronous wrapper for signal_ready."""
    if _oob_coordinator is None:
        raise RuntimeError("OOB coordinator not initialized.")
    _run_sync(_oob_coordinator.signal_ready(transfer_id))


def oob_wait_ready_sync(transfer_id: str, receiver_rank: int, timeout: Optional[float] = None) -> None:
    """Synchronous wrapper for wait_ready."""
    if _oob_coordinator is None:
        raise RuntimeError("OOB coordinator not initialized.")
    _run_sync(_oob_coordinator.wait_ready(transfer_id, receiver_rank, timeout))


def oob_signal_complete_sync(transfer_id: str) -> None:
    """Synchronous wrapper for signal_complete."""
    if _oob_coordinator is None:
        raise RuntimeError("OOB coordinator not initialized.")
    _run_sync(_oob_coordinator.signal_complete(transfer_id))


def oob_wait_complete_sync(transfer_id: str, sender_rank: int, timeout: Optional[float] = None) -> None:
    """Synchronous wrapper for wait_complete."""
    if _oob_coordinator is None:
        raise RuntimeError("OOB coordinator not initialized.")
    _run_sync(_oob_coordinator.wait_complete(transfer_id, sender_rank, timeout))
