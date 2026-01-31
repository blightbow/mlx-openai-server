The Critical Metric: Barrier Release Tightness
Your JACCL problem is fundamentally about simultaneous entry into collective operations. The OOB barrier's job is to hold all ranks until everyone arrives, then release them as close to simultaneously as possible.
The research reveals a decisive difference:
MetricZeroMQNNG (pynng)Impact on JACCLJitter~1 µs~8-10 µs8-10x worse release spread with NNGP99 latency45 µs62 µs38% worse worst-caseMax latency280 µs2,200 µs8x worse tail → occasional large bubblesMax latency spikesRareCommon at 512KB+NNG has structural issues
What this means for JACCL: After your barrier releases all ranks, they enter the JACCL collective. With NNG's ~10 µs jitter, ranks could be staggered by 10+ µs entering all_sum. With ZeroMQ's ~1 µs jitter, the stagger is 1 µs.
Given JACCL's RDMA latency is 5-9 µs, NNG's jitter is on the same order as your RDMA latency itself—your barrier could introduce timing skew comparable to the RDMA operation time. ZeroMQ's jitter is an order of magnitude tighter.

SURVEY Pattern: Not Actually Ideal for Barriers
Your plan uses SURVEY for barriers, but there's a subtle mismatch:
PropertyMPI_BarrierNNG SURVEYCompletion conditionAll N ranks arrivedTimeout expiresSemanticsCount-basedTime-basedRelease triggerLast arrivalTimer
SURVEY doesn't naturally express "wait for exactly N respondents then release." You've worked around this by collecting responses until timeout, but this means:

Barrier latency = survey_time (you're always waiting for timeout, not early release on last arrival)
Or you set short survey_time and risk incomplete collection

ZeroMQ's REQ/REP coordinator pattern gives you count-based release:
python# Coordinator releases immediately when all N arrive
arrivals = 0
while arrivals < N:
    await sock.recv()  # Block until arrival
    arrivals += 1
# All arrived → immediate release (no timeout wait)
await pub_sock.send(b"GO")  # ~1 µs jitter on release
```

---

## TB5 Link Contention: Probably Not Your Problem

The research found **no documented cases** of TCP starvation on saturated Thunderbolt 5 RDMA links. More importantly:

**Barrier frequency in pipeline parallelism is low.** You need barriers at:
- Pipeline flush points (every N microbatches)
- Termination
- Possibly periodic sync points

You're not barriring every forward pass. If barriers occur every ~100ms of compute, then even a 100 µs barrier (degraded TCP) is 0.1% overhead—negligible.

**However**: If TCP latency *does* spike under RDMA saturation (uncharacterized), ZeroMQ's tighter baseline means you have more headroom before it becomes problematic.

---

## Recommendation: ZeroMQ for Tighter JACCL Synchronization

For your specific problem—ensuring synchronized entry into JACCL collectives on a saturated TB5 link—**PyZMQ is better aligned**:

| Factor | Winner | Why |
|--------|--------|-----|
| Barrier release tightness | **ZeroMQ** | 1 µs jitter vs 10 µs |
| Worst-case pipeline bubble | **ZeroMQ** | 280 µs max vs 2200 µs |
| Count-based barrier semantics | **ZeroMQ** | REQ/REP gives immediate release on last arrival |
| Headroom under contention | **ZeroMQ** | Tighter baseline = more margin |
| macOS ARM64 deployment | **ZeroMQ** | Native wheels vs CMake build |
| RDMA timing alignment | **ZeroMQ** | 1 µs jitter << 5-9 µs RDMA latency |

**The one advantage NNG has**—built-in SURVEY pattern—turns out to be a liability because SURVEY's time-based semantics don't match what you actually need (count-based barrier).

---

## If You Stick with pynng

Your current approach works, but consider these mitigations:

1. **Set `survey_time` aggressively low** (e.g., 50ms) and implement retry logic for missed responses
2. **Track response count explicitly** and break out of the receive loop early when all N responses arrive (don't wait for timeout)
3. **Monitor jitter** by timestamping barrier entry/exit across ranks

---

## One More Option: Hybrid Approach

Given the uncharacterized TB5 contention behavior, consider:
```
RDMA Data Plane:  Thunderbolt 5 (saturated, JACCL)
OOB Control Plane: Gigabit Ethernet (separate physical link)
A USB-C Ethernet adapter on each Mac Studio gives you:

Complete isolation from RDMA traffic
Characterized latency (~10-50 µs, not dependent on TB5 load)
Industry-standard HPC practice (separate control plane)

This eliminates the unknown of TCP/RDMA coexistence entirely.

Summary
QuestionAnswerWhich is better aligned with JACCL timing?ZeroMQ (8-10x tighter jitter)Is SURVEY ideal for barriers?No (time-based, not count-based)Will TB5 contention be a problem?Probably not (barriers are infrequent)Should you switch from pynng?Yes, if tighter synchronization mattersBest insurance policy?Separate Ethernet for OOB
