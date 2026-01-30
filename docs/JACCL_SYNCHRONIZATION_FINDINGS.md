# JACCL Synchronization Findings

**Date:** 2026-01-30
**Authors:** slag, Claude Opus 4.5
**MLX Version:** HEAD (post-jaccl-large-buffer branch)
**macOS Version:** 26.2 (Thunderbolt 5 RDMA)

## Executive Summary

We discovered an **undocumented synchronization requirement** in MLX's JACCL backend: all ranks must enter collective operations (e.g., `all_sum`, `all_gather`) simultaneously. Without external synchronization, RDMA operations can receive data from subsequent operations, causing data corruption.

This issue likely went unnoticed because the commonly recommended "zero-contributing all_sum" pattern for pseudo-send/recv is naturally tolerant to timing errors.

## The Problem

### Symptoms Observed

1. **JACCL warmup mismatch**: Rank 0 received `97` ('a' in ASCII) instead of `1` from Rank 1
2. **Asymmetric corruption**: One rank got correct results while the other got garbage
3. **Timing-dependent**: Adding barriers made the problem disappear; removing them brought it back
4. **Post-crash corruption**: Ungraceful termination (Ctrl+C) exacerbated the issue on subsequent runs

### Root Cause

JACCL's collective operations immediately post RDMA send/recv work requests without waiting for all ranks to enter the operation:

```cpp
// From mlx/mlx/distributed/jaccl/jaccl.cpp, all_reduce implementation (lines 1066-1079)
// Prefill the pipeline
int buff = 0;
while (read_offset < total && buff < PIPELINE) {
    post_recv_all(buff);    // Posts receives immediately
    std::copy(...);          // Copies local data to send buffer
    post_send_all(buff);    // Posts sends immediately
    // ...
}
```

If Rank 1 enters `all_sum` and starts RDMA operations while Rank 0 hasn't entered yet:
- Rank 1's sends have no matching receives on Rank 0
- When Rank 0 eventually enters, it may receive data from Rank 1's *subsequent* operations
- Result: corrupted data that appears to be from unrelated operations

### Why It Went Unnoticed: The Cargo Cult Hypothesis

The MLX team recommends using `all_sum` with zero-contributing peers as a workaround for the lack of safe send/recv:

```python
# Recommended pattern for pseudo-broadcast
if rank == 0:
    data = mx.array([actual_data])
else:
    data = mx.zeros_like(template)
result = mx.distributed.all_sum(data)  # Everyone gets actual_data
```

**This pattern is timing-tolerant because:**
1. Only ONE rank contributes meaningful data
2. Other ranks contribute zeros
3. Even with timing issues: `0 + 0 + data + stale_zero = data`
4. The result appears correct despite underlying race conditions

**Our pattern (true collective) is timing-sensitive:**
```python
data = mx.array([rank])  # Rank 0: [0], Rank 1: [1]
result = mx.distributed.all_sum(data)  # Expected: [1]
```

If timing is off: `0 + garbage ≠ 1` - corruption is immediately visible.

The widespread use of the zero-contributing pattern may have masked this synchronization bug, leading to cargo cult propagation of a pattern that works *despite* the underlying issue.

## The Solution

### OOB (Out-of-Band) Synchronization

We implemented an OOB coordination layer using Python's `multiprocessing.managers` that provides:

1. **Barriers**: Synchronize all ranks before collective operations
2. **Termination signaling**: Notify peers when shutting down to prevent one-sided RDMA operations
3. **Ready/complete signaling**: Receiver-initiated rendezvous for safe send/recv

### Key Integration Points

```python
# Before JACCL all_sum (from app/main.py)
if oob is not None:
    oob.barrier("pre_warmup")  # Ensure all ranks enter together
warmup = mx.distributed.all_sum(warmup_input, group=group)
mx.eval(warmup)
if oob is not None:
    oob.barrier("post_warmup")  # Ensure all ranks passed before continuing
```

### All JACCL Collective Operations Audited

| Location | Operation | Mitigation |
|----------|-----------|------------|
| main.py:267 | warmup all_sum | Pre/post OOB barriers |
| file_sync.py:469+ | broadcast_manifest | Follows post-warmup barrier |
| file_sync.py:1769+ | weight transfers | Per-file OOB barriers |
| mlx_lm.py:313 | pipeline sync | Added OOB barrier |
| mlx_lm.py:330 | final sync | Added OOB barrier |
| coordinator.py | inference loop | Workers wait on coordinator (implicit sync) |

## JACCL Internals

### Unexposed Side Channel

JACCL has an internal `SideChannel` class (jaccl.cpp:445-555) that provides TCP-based coordination:

```cpp
class SideChannel {
    // ...
    template <typename T>
    std::vector<T> all_gather(const T& v);  // Used for barrier via all_gather<int>(0)
};
```

This is used once during initialization (`cm.barrier()` at line 1193) but is not exposed to users. We essentially reimplemented this functionality in Python for our OOB layer.

### RDMA Buffer Management

- `NUM_BUFFERS = 2` (double buffering)
- `BUFFER_SIZE = 4096` bytes per buffer
- `PIPELINE = 2` for send/recv pipelining
- No explicit synchronization between posting and completing work requests

## Recommendations

### For MLX Team

1. **Document the synchronization requirement**: Add a warning that ranks should enter collective operations together

2. **Expose the side channel**: Allow users to call `group.barrier()` for explicit synchronization

3. **Add internal barriers**: Consider adding automatic synchronization before collective operations (like MPI's eager protocol)

4. **Test with true collectives**: The current test patterns may all use zero-contributing all_sum, masking timing issues

### For Users of JACCL

1. **Use OOB synchronization**: Implement barriers before JACCL collective operations

2. **Handle termination gracefully**: Signal peers when shutting down to prevent RDMA corruption

3. **Verify with true collectives**: Test with operations where all ranks contribute meaningful data

4. **Fresh state after crashes**: Reboot or reset RDMA interfaces after ungraceful termination

## Code References

### OOB Implementation
- `app/distributed/oob.py`: OOBCoordinator class with barrier, signal_ready, signal_terminating

### Integration Points
- `app/main.py`: Warmup barriers and termination handling
- `app/distributed/file_sync.py`: Per-file barriers for weight transfer
- `app/models/mlx_lm.py`: Post-load synchronization barriers

### MLX Source Analysis
- `mlx/mlx/distributed/jaccl/jaccl.cpp`: JACCL implementation
- `mlx/docs/src/usage/distributed.rst`: Official documentation (no sync warnings)

## Appendix: Debugging Timeline

1. **Initial symptom**: Rank 0 received `97` ('a') instead of `1` in warmup all_sum
2. **Hypothesis 1**: RDMA buffer corruption - tried interface reset, reboots
3. **Discovery 1**: Corruption was asymmetric (one rank correct, other garbage)
4. **Hypothesis 2**: Timing issue - data from subsequent ops being misrouted
5. **Discovery 2**: Adding post-warmup barrier caused Rank 0 to hang on all_sum
6. **Key insight**: JACCL requires synchronized entry; barrier before all_sum fixed it
7. **Root cause confirmed**: JACCL posts RDMA ops immediately without coordination
8. **Cargo cult hypothesis**: Zero-contributing pattern masks the timing bug

## Upstream Research: Confirming Evidence

### GPU Timeout from Asymmetric Timing

**Source:** [ml-explore/mlx-examples#1185](https://github.com/ml-explore/mlx-examples/issues/1185)

When heterogeneous machines perform distributed operations, faster nodes finish first and wait. If the wait exceeds ~5 seconds, Metal times out:

```
libc++abi: terminating due to uncaught exception of type std::runtime_error:
[METAL] Command buffer execution failed: Caused GPU Timeout Error
(00000002:kIOGPUCommandBufferCallbackErrorTimeout)
```

**Key quote from @awni (MLX maintainer):**
> "This is a known issue with Metal that the GPU can only wait up to ~5 seconds after which it times out and you get an error. Your case especially exposes this since you are on heterogeneous machines the Ultra is much faster and finishes the validation early and just waits > 5 seconds for the other machines to finish."

**Workaround:** `mx.distributed.all_sum(data, stream=mx.cpu)` - running on CPU stream avoids Metal timeout.

### send/recv Consistently Fails, all_sum-with-zeros Works

**Source:** [ml-explore/mlx#1849](https://github.com/ml-explore/mlx/issues/1849)

This issue explicitly documents three approaches for sharing data:

1. **Option 1 (local generation):** Each node generates own data - WORKS
2. **Option 2 (send/recv):** Rank 0 sends slices to other ranks - **CONSISTENTLY FAILS** with GPU timeout
3. **Option 3 (all_sum with zeros):** All ranks contribute, non-owners contribute zeros - WORKS

**The failing code pattern:**
```python
def get_data_rank0_distributes(N_total, rank, world_size):
    for r in range(1, world_size):
        subarr = data_global[s:e, :]
        mx.eval(subarr)
        mx.distributed.send(subarr, dst=r)  # <-- FAILS with GPU timeout
```

**The working workaround:**
```python
def get_data_all_sum(N_total, rank, world_size):
    global_data = mx.zeros((N_total, 3), dtype=mx.float32)
    if rank == 0:
        global_data[:] = actual_data
    mx.eval(global_data)
    global_data_sum = mx.distributed.all_sum(global_data)  # <-- WORKS
```

### The Pattern Is Explicitly a Workaround

**Source:** [ml-explore/mlx#1220 (Discussion)](https://github.com/ml-explore/mlx/discussions/1220)

A user describes using the zero-contributing pattern:
> "I am trying to create the various dataset arrays on rank=0, along also empty (zero) counterpart arrays of the same shape on rank > 0 and then use mx.distributed.all_sum to get copies of the dataset on all ranks."

The user explicitly notes this is done **"in the absence of a distributed broadcast"** - confirming it's a workaround, not a recommended pattern.

### Lazy Evaluation Creates Synchronization Hazards

**Source:** [ml-explore/mlx#1849](https://github.com/ml-explore/mlx/issues/1849)

**Key quote from @awni:**
> "A `mx.distributed.send` is an operation in the graph like everything else. It has to be evaluated or it will never be realized. If you just call the send without evaling the output, then the receiving process is waiting on something to be sent which never happens. And you get the timeout."

This confirms that `mx.eval()` timing is critical and easy to get wrong with point-to-point operations.

### all_reduce_grads Failures with Multiple Nodes

**Source:** [ml-explore/mlx#1226](https://github.com/ml-explore/mlx/issues/1226)

> "When I try to use it with N>1, the code goes into a zombie state and the GPU stops being utilized effectively and the code never recovers from that state."

**Issue resolution from @angeloskath:**
> "I think this failure had to do with IP over thunderbolt across many machines so there is not something we can do from MLX."

### JACCL/RDMA Stability Issues

**Source:** [exo-explore/exo#1066](https://github.com/exo-explore/exo/issues/1066)

Critical crash report in EXO project:
> "RDMA-based strategies (Tensor/Pipeline MLX RDMA) fail to load models, causing EXO to crash with SIGSEGV. Only MLX Ring strategies succeed."

Crash occurs during RDMA buffer allocation in `ConnectionManager::initialize()`.

**Source:** [ml-explore/mlx#2944](https://github.com/ml-explore/mlx/issues/2944)

JACCL socket binding errors:
```
RuntimeError: [jaccl] Couldn't bind socket (error: 47)
```

### Summary: What We Discovered vs. What Was Known

| Aspect | Known Upstream | Our Discovery |
|--------|---------------|---------------|
| send/recv has timing issues | Yes (Issue #1849) | Confirmed |
| all_sum-with-zeros works as workaround | Yes (Issue #1849, #1220) | Confirmed |
| Metal has 5-second GPU timeout | Yes (Issue #1185) | Confirmed |
| `all_sum` itself requires sync | **NO** | **NEW** |
| Zero pattern masks timing bugs | **NO** | **NEW** |
| True collectives expose the bug | **NO** | **NEW** |
| OOB barriers fix the root cause | **NO** | **NEW** |

Our contribution is identifying that:
1. The synchronization requirement applies to ALL collective operations, not just send/recv
2. The zero-contributing pattern accidentally provides synchronization (idempotent to errors)
3. Using `all_sum` for actual collectives (where all ranks contribute meaningful data) exposes the bug
4. OOB barriers before JACCL ops fix the root cause

## Related Issues

- PyTorch TCPStore IPv6 issues on macOS (why we used multiprocessing.managers instead)
- MLX GitHub Issue #148440: macOS distributed support explicitly unmaintained
- JACCL send/recv SIGBUS crashes documented in `app/distributed/oob.py` docstring

## References

- [ml-explore/mlx#1849 - Issue with mx.distributed send and recv](https://github.com/ml-explore/mlx/issues/1849)
- [ml-explore/mlx#1226 - all_reduce_grads() fails with Transformer model for nodes > 1](https://github.com/ml-explore/mlx/issues/1226)
- [ml-explore/mlx#2944 - issues with RDMA and JACCL](https://github.com/ml-explore/mlx/issues/2944)
- [ml-explore/mlx#1220 - Memory usage when running in parallel with mx.distributed](https://github.com/ml-explore/mlx/discussions/1220)
- [ml-explore/mlx-examples#1185 - Distributed LORA training fails with METAL error](https://github.com/ml-explore/mlx-examples/issues/1185)
- [ml-explore/mlx#2808 - Thunderbolt RDMA communications backend (PR)](https://github.com/ml-explore/mlx/pull/2808)
- [exo-explore/exo#1066 - RDMA Fails to Load Models](https://github.com/exo-explore/exo/issues/1066)
- [MLX Distributed Documentation](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
