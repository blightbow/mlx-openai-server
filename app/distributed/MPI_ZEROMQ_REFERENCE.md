# MPI to ZeroMQ Pattern Reference

This document captures the mapping between MPI patterns and ZeroMQ implementations for JACCL OOB coordination. It serves as a reference for future development and troubleshooting.

## Current Implementation Status

### Implemented Patterns

| MPI Pattern | Our Implementation | ZeroMQ Equivalent |
|-------------|-------------------|-------------------|
| `MPI_Barrier` | `oob.barrier()` | REQ/REP arrivals + PUB/SUB release (count-based) |
| `MPI_Ibarrier` | `BarrierHandle` | Async with asyncio.Event |
| Receiver-initiated rendezvous | `signal_ready/wait_ready` | REQ/REP store |
| Completion signaling | `signal_complete/wait_complete` | REQ/REP store |
| Failure notification | `signal_terminating` | REQ/REP relay + PUB/SUB broadcast |
| Abort broadcast | `oob.abort()` | PUB/SUB broadcast |

### Not Yet Implemented (Future Work)

| MPI Pattern | Use Case | ZeroMQ Implementation | Build Effort |
|-------------|----------|----------------------|--------------|
| `MPI_Bcast` | Distribute config/metadata from rank 0 | PUB/SUB | Low |
| `MPI_Gather` | Collect outputs in tensor parallelism | REQ/REP collect | Low |
| `MPI_Scatter` | Distribute inputs in tensor parallelism | REQ/REP per rank | Medium |
| `MPI_Allreduce` | Aggregate loss/metrics across stages | REQ/REP + local reduce + Bcast | Medium |
| `MPI_Allgather` | All ranks get all data | REQ/REP + PUB/SUB | Medium |
| `MPI_Comm_split` | Subgroups for pipeline stages | Separate socket groups | Medium |
| `MPIX_Comm_revoke` (ULFM) | Clean abort on failure | PUB/SUB abort channel | Low |
| `MPIX_Comm_shrink` (ULFM) | Continue with surviving ranks | Reconnect with subset | High |
| Request/Test pattern | Poll without blocking | asyncio + poll | Low |

## Why ZeroMQ Over pynng

Research showed critical issues with pynng for JACCL coordination:

| Metric | ZeroMQ | pynng | Impact |
|--------|--------|-------|--------|
| Jitter | ~1 µs | ~8-10 µs | 8-10x worse barrier release spread |
| P99 latency | 45 µs | 62 µs | 38% worse worst-case |
| Max latency | 280 µs | 2,200 µs | 8x worse tail events |
| Barrier semantics | Count-based | Time-based (SURVEY) | pynng waits for timeout, not count |

Given JACCL's RDMA latency is 5-9 µs, pynng's ~10 µs jitter is comparable to RDMA operation time itself.

## ZeroMQ Pattern Details

### Count-Based Barrier (REQ/REP + PUB/SUB)

Unlike pynng's time-based SURVEY, ZeroMQ enables true count-based barriers:

```
Workers:                           Coordinator (rank 0):
    |                                    |
    |---[ARRIVE:{name}:{rank}]---------->|
    |<--[OK]-----------------------------| (count arrivals)
    |                                    |
    |     (wait for all N-1 arrivals)    |
    |                                    |
    |<--[RELEASE:{name}]-----------------| (PUB broadcast)
[proceed]                                 [proceed]
```

**Key properties:**
- Coordinator counts arrivals via REQ/REP
- Release broadcast is immediate when count reached (not timeout)
- ~1 µs jitter vs SURVEY's ~10 µs

### Key-Value Store (REQ/REP)

```
Worker:                            Coordinator:
    |                                   |
    |---[SET:key:value]---------------->|
    |<--[OK]----------------------------|
    |                                   |
    |---[GET:key]---------------------->|
    |<--[value]-------------------------|
```

**Key properties:**
- Request-response with guaranteed reply
- Stateful - must alternate req/rep
- Good for: key-value operations, signaling

### Termination Broadcast (PUB/SUB)

```
Worker:                            Coordinator:              All Ranks:
    |                                   |                        |
    |---[TERM:{rank}]------------------>| (REQ to relay)         |
    |                                   |---[TERM:{rank}]------->| (PUB)
    |                                   |                        |
```

**Key properties:**
- Workers send termination to coordinator via REQ
- Coordinator relays to all via PUB
- Fire-and-forget (no delivery guarantee needed for termination)

## MPI Concepts Reference

### Point-to-Point Communication

| MPI | Description | ZeroMQ |
|-----|-------------|--------|
| `MPI_Send` | Blocking send | `sock.send()` |
| `MPI_Recv` | Blocking receive | `sock.recv()` |
| `MPI_Isend` | Non-blocking send | `await sock.send()` |
| `MPI_Irecv` | Non-blocking receive | `await sock.recv()` |
| `MPI_Sendrecv` | Combined send+recv | Two sockets or DEALER/ROUTER |

### Collective Operations

| MPI | Description | ZeroMQ Pattern |
|-----|-------------|----------------|
| `MPI_Barrier` | All ranks synchronize | REQ/REP arrivals + PUB/SUB release |
| `MPI_Ibarrier` | Non-blocking barrier | Async barrier with asyncio.Event |
| `MPI_Bcast` | One-to-all | PUB/SUB |
| `MPI_Reduce` | All-to-one with operation | REQ/REP collect + local reduce |
| `MPI_Allreduce` | Reduce + broadcast | REQ/REP + reduce + PUB/SUB |
| `MPI_Gather` | Collect at root | REQ/REP collect |
| `MPI_Scatter` | Distribute from root | REQ/REP per rank |
| `MPI_Allgather` | Everyone gets everything | REQ/REP + PUB/SUB |

### Synchronization / Completion

| MPI | Description | ZeroMQ/asyncio |
|-----|-------------|----------------|
| `MPI_Request` | Handle for async op | `asyncio.Task` |
| `MPI_Test` | Non-blocking completion check | `task.done()` |
| `MPI_Wait` | Block until complete | `await task` |
| `MPI_Testall` | Test multiple | `all(t.done() for t in tasks)` |
| `MPI_Waitall` | Wait for all | `await asyncio.gather(*tasks)` |
| `MPI_Testany` | Test any one | `asyncio.wait(FIRST_COMPLETED)` |
| `MPI_Cancel` | Cancel operation | `task.cancel()` |

### Fault Tolerance (ULFM)

| MPI | Description | ZeroMQ |
|-----|-------------|--------|
| `MPIX_Comm_revoke` | Invalidate communicator | PUB/SUB abort broadcast |
| `MPIX_Comm_shrink` | Create new comm without failed | Reconnect subset |
| `MPIX_Comm_failure_ack` | Acknowledge failures | Track in local set |
| `MPI_ERR_PROC_FAILED` | Process failure error | Custom `PeerTerminatedError` |

## Key Design Insights

### Why REQ/REP + PUB/SUB for Barrier (Not SURVEY)

1. **Count-based, not time-based**: REQ/REP arrivals allow immediate release when count reached
2. **Lower jitter**: ~1 µs vs SURVEY's ~10 µs (critical for RDMA coordination)
3. **Async-native**: `await sock.recv()` integrates with asyncio cancellation
4. **Natural abort**: Set abort flag, pending waits raise AbortError

### Why PUB/SUB for Termination (Not Store Polling)

1. **Immediate delivery**: No polling delay
2. **Fire-and-forget**: Don't block waiting for ack during shutdown
3. **All ranks notified**: Single publish reaches everyone
4. **Decoupled**: Publisher doesn't need to know subscriber count

### Why REQ/REP for Store (Not Shared Memory)

1. **Cross-machine**: Works over TCP, not just same-host
2. **No locking complexity**: Request serialization handled by ZeroMQ
3. **Clear ownership**: Rank 0 owns store, workers are clients
4. **Debuggable**: Can log all store operations

## Socket Topology

```
Rank 0 (Coordinator):
  REP  tcp://0.0.0.0:29400  (bind) - barrier arrivals
  REP  tcp://0.0.0.0:29401  (bind) - store operations
  REP  tcp://0.0.0.0:29402  (bind) - termination relay
  PUB  tcp://0.0.0.0:29403  (bind) - broadcast

Workers (Rank 1+):
  REQ  tcp://{host}:29400  (connect) - barrier signal
  REQ  tcp://{host}:29401  (connect) - store operations
  REQ  tcp://{host}:29402  (connect) - termination signal
  SUB  tcp://{host}:29403  (connect) - broadcast receive
```

## Socket Configuration for Low Latency

```python
# Bounded graceful shutdown (100ms max wait)
sock.setsockopt(zmq.LINGER, 100)

# Small buffers for latency over throughput
sock.setsockopt(zmq.SNDBUF, 4096)
sock.setsockopt(zmq.RCVBUF, 4096)
```

## Timeout Handling

ZeroMQ socket timeouts don't work reliably with asyncio. Use asyncio timeouts:

```python
try:
    result = await asyncio.wait_for(sock.recv(), timeout=5.0)
except asyncio.TimeoutError:
    raise PeerTimeoutError("Operation timed out")
```

## Gotchas and Mitigations

| Issue | Description | Mitigation |
|-------|-------------|------------|
| No native socket timeout in asyncio | `ZMQ_RCVTIMEO` doesn't work | Use `asyncio.wait_for()` |
| PUB/SUB slow joiner | Subscribers may miss early messages | Workers connect SUB before signaling REQ |
| Socket lifecycle | Must close explicitly | Use `finally` blocks, set `LINGER=100` |
| Context termination | Can hang on close | Set LINGER on all sockets before `ctx.term()` |

## References

### ZeroMQ Documentation
- [PyZMQ Documentation](https://pyzmq.readthedocs.io/)
- [PyZMQ asyncio API](https://pyzmq.readthedocs.io/en/latest/api/zmq.asyncio.html)
- [ZeroMQ Guide Chapter 2 - Sockets](https://zguide.zeromq.org/docs/chapter2/)
- [ZeroMQ Guide Chapter 5 - Advanced Pub-Sub](https://zguide.zeromq.org/docs/chapter5/)

### MPI Documentation
- [MPI Forum Standard 4.1](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf)
- [Open MPI ULFM (Fault Tolerance)](https://docs.open-mpi.org/en/v5.0.x/features/ulfm.html)
- [MPI Collective Operations - OxRSE](https://train.rse.ox.ac.uk/material/HPCu/high_performance_computing/hpc_mpi/05_collective_communication)
- [MPI Reduce/Allreduce Tutorial](https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/)
- [MPI Groups and Communicators](https://mpitutorial.com/tutorials/introduction-to-groups-and-communicators/)
- [Non-blocking MPI - LLNL](https://hpc-tutorials.llnl.gov/mpi/non_blocking/)
- [MPI One-Sided Sync - ENCCS](https://enccs.github.io/intermediate-mpi/one-sided-sync/)

### MPI Progress and Async
- [MPI Progress For All (arXiv)](https://arxiv.org/html/2405.13807v1)
- [Intel MPI Async Progress](https://www.intel.com/content/www/us/en/docs/mpi-library/developer-guide-linux/2021-6/asynchronous-progress-control.html)
- [Open MPI Progress Threads](https://github.com/open-mpi/ompi/wiki/ProgressThreads)

### MPI Protocols
- [MPI Protocols Overview](https://pavanakumar.github.io/post/mpi-protocols/)
- [Avoiding Deadlocks - Cornell](https://cvw.cac.cornell.edu/mpip2p/p2p-usage-strategies/avoiding-deadlocks)

---

*Document created: 2026-01-30*
*Last updated: 2026-01-30*
*Replaces: MPI_PYNNG_REFERENCE.md (pynng branch)*
