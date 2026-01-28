# Distributed KV Cache Architecture Analysis

This document analyzes caching strategies for distributed LLM inference, specifically comparing how different architectures and eviction policies perform under **tensor parallelism** vs **pipeline parallelism** modes.

## Background

MLX supports local (single-node) and two distributed inference modes with fundamentally different characteristics:

| Aspect | Local (Single Node) | Tensor Parallelism | Pipeline Parallelism |
|--------|---------------------|-------------------|----------------------|
| **Layer distribution** | All layers on one device | All ranks have all layers | Layers split across ranks |
| **Weight sharding** | None | Attention heads divided by N | Full heads, fewer layers |
| **Token visibility** | Full visibility | All ranks see all tokens | Only entry rank sees tokens |
| **Communication** | None | `all_sum()` per layer | `send()`/`recv()` between stages |
| **KV cache structure** | `(batch, heads, seq, dim)` | `(batch, heads/N, seq, dim)` per layer | `(batch, heads, seq, dim)` for local layers only |
| **Sync requirement** | None | Implicit (same tokens → same cache state) | Explicit (cache decisions must be coordinated) |

The choice of caching architecture must account for these differences.

---

## Part 1: Cache Architecture Research

### 1.1 LRU Prompt Cache (Current mlx-lm Implementation)

The current implementation uses a trie-based LRU cache where:
- Cache key = token sequence
- Cache value = KV states for all layers
- Eviction = Least Recently Used

**Reference**: `mlx_lm/server.py`, backported to `app/utils/prompt_cache.py`

### 1.2 Radix Tree / RadixAttention (SGLang)

SGLang's RadixAttention uses a radix tree (compressed trie) for automatic prefix discovery:

```
Radix Tree Example:
              [system_prompt]
              /             \
      [user_turn_1]      [user_turn_2]
           |
    [assistant_turn_1]
           |
      [user_turn_2]
```

**Key properties**:
- Edges can represent multiple tokens (compressed)
- Automatic sharing of common prefixes across requests
- No explicit cache keys needed - prefix matching is structural
- Production hit rates: 52-74% (Chatbot Arena deployment)

**Reference**: https://lmsys.org/blog/2024-01-17-sglang/

### 1.3 Coordinated Cache with Controller (LMCache)

LMCache introduces a two-layer coordination architecture:

```
┌─────────────────────────────────────┐
│     Controller Manager (Global)     │
│  - Cache index                      │
│  - Routing decisions                │
│  - Migration orchestration          │
└───────────────┬─────────────────────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐
│Worker 0│ │Worker 1│ │Worker 2│
│(local) │ │(local) │ │(local) │
└────────┘ └────────┘ └────────┘
```

**Key properties**:
- Centralized coordination, distributed execution
- Supports cross-node KV transfer (RDMA, NVLink)
- Hierarchical storage: GPU → CPU → disk → remote
- Up to 15× throughput improvement

**Reference**: https://docs.lmcache.ai/developer_guide/architecture.html

### 1.4 Prefill-Decode Disaggregation Pattern

Modern production systems separate prefill (compute-bound) from decode (memory-bound):

```
Prefill Node                    Decode Node
┌────────────────┐   KV xfer   ┌────────────────┐
│ Process prompt │ ─────────→  │ Generate tokens│
│ Build KV cache │    RDMA     │ Reuse KV cache │
└────────────────┘             └────────────────┘
```

**Key insight**: This pattern is architecturally similar to pipeline parallelism, where KV caches flow between stages.

**Reference**: https://hao-ai-lab.github.io/blogs/distserve/

---

## Part 2: Architecture Comparison Matrix

### 2.1 Local Operation Compatibility

| Architecture | Compatibility | Pros | Cons |
|--------------|---------------|------|------|
| **LRU Trie** | ✅ Optimal | Simple; proven; low overhead; current mlx-lm default | No automatic prefix discovery; exact key matching only |
| **Radix Tree** | ✅ Works | Automatic prefix sharing; memory efficient for multi-turn | Additional complexity for single-node; tree overhead may not be justified |
| **Controller-Based** | ❌ Unnecessary | - | Coordination overhead with no benefit; added latency and complexity |
| **P/D Disaggregation** | ❌ N/A | Not applicable - single node has no prefill/decode separation | - |

**Recommendation for Local Operation**: LRU Trie (current implementation). Simple, proven, minimal overhead. Radix Tree is acceptable if multi-turn/shared-prefix workloads dominate and justify the added complexity.

### 2.2 Tensor Parallelism Compatibility

| Architecture | Compatibility | Pros | Cons |
|--------------|---------------|------|------|
| **LRU Trie** | ✅ Native fit | Simple, no coordination needed; identical cache state across ranks guaranteed by identical token sequences | No automatic prefix discovery; cache keys must be exact matches |
| **Radix Tree** | ✅ Native fit | Automatic prefix sharing; memory efficient for multi-turn; no explicit key management | More complex implementation; tree operations add overhead |
| **Controller-Based** | ⚠️ Overkill | Could enable advanced features (cache-aware routing, migration) | Unnecessary complexity; adds latency for coordination that isn't needed |
| **P/D Disaggregation** | ❌ N/A | Not applicable - tensor parallelism doesn't separate prefill/decode by rank | - |

**Recommendation for Tensor Parallelism**: Radix Tree or LRU Trie. Both work naturally because all ranks see identical token sequences, guaranteeing implicit cache synchronization.

### 2.3 Pipeline Parallelism Compatibility

| Architecture | Compatibility | Pros | Cons |
|--------------|---------------|------|------|
| **LRU Trie** | ❌ Broken without coordination | - | Only entry rank knows tokens; other ranks can't perform lookups; cache state diverges |
| **Radix Tree** | ⚠️ Requires extension | Efficient prefix matching at entry rank | Must broadcast cache decisions to other pipeline stages; entry rank becomes bottleneck |
| **Controller-Based** | ✅ Good fit | Natural coordination point; entry rank queries controller, others follow | Adds complexity and latency; controller is single point of failure |
| **P/D Disaggregation** | ✅ Natural fit | Pipeline stages mirror P/D architecture; KV transfer patterns already established | Requires RDMA or high-bandwidth interconnect for efficiency |

**Recommendation for Pipeline Parallelism**: Controller-Based or P/D-style coordination. The fundamental asymmetry (only entry rank sees tokens) requires explicit coordination.

### 2.4 Summary Matrix

| Architecture | Local | Tensor Parallel | Pipeline Parallel | Implementation Complexity |
|--------------|-------|-----------------|-------------------|--------------------------|
| **LRU Trie** | ✅ Optimal | ✅ Works | ❌ Broken | Low |
| **Radix Tree** | ⚠️ Acceptable | ✅ Optimal | ⚠️ Needs coordination | Medium |
| **Controller-Based** | ❌ Overkill | ⚠️ Overkill | ✅ Good fit | High |
| **P/D Pattern** | ❌ N/A | ❌ N/A | ✅ Natural fit | High |

**Key observations**:
- **LRU Trie**: Best for local and tensor parallel; broken for pipeline without coordination
- **Radix Tree**: Good across local/tensor; requires extension for pipeline
- **Controller-Based**: Only justified for pipeline parallelism
- **P/D Pattern**: Pipeline-specific; not applicable elsewhere

---

## Part 3: Eviction Strategy Research

### 3.1 Least Recently Used (LRU)

Standard time-based eviction. Evicts entries not accessed recently.

**Reference**: Current mlx-lm implementation

### 3.2 FLOP-Aware Eviction (Marconi)

Prioritizes evicting entries that save fewer FLOPs when reused:

```
eviction_priority = tokens_cached × frequency_of_reuse
```

Entries with few cached tokens or low reuse frequency are evicted first.

**Result**: 19-220% improvement in token hit rate vs LRU.

**Reference**: https://arxiv.org/html/2411.19379v2

### 3.3 Workflow-Aware Eviction (KVFlow)

For agentic/multi-turn workloads, tracks "steps to next execution":

```
eviction_priority = estimated_steps_until_next_use
```

Avoids evicting caches for agents that will become active soon.

**Reference**: https://arxiv.org/pdf/2507.07400

### 3.4 Staleness-Aware Eviction (LLMCache)

Tracks match rate decay over time:

```
eviction_priority = current_match_rate × decay_factor(time_since_last_hit)
```

Entries with declining match rates are evicted even if recently used.

**Reference**: LLMCache paper

---

## Part 4: Eviction Strategy Comparison Matrix

### 4.1 Local Operation Compatibility

| Strategy | Compatibility | Pros | Cons |
|----------|---------------|------|------|
| **LRU** | ✅ Optimal | Simple; proven; low overhead; predictable behavior | Suboptimal for variable-length prompts; no workload awareness |
| **FLOP-Aware** | ✅ Works | Better memory efficiency for mixed prompt lengths | Tracking overhead; more complex implementation |
| **Workflow-Aware** | ✅ Works | Optimal for agentic/multi-turn patterns | Requires workflow metadata; highest complexity |
| **Staleness-Aware** | ✅ Works | Adapts to changing access patterns | Temporal tracking overhead; decay parameter tuning |

**Recommendation for Local Operation**: LRU for general workloads. Consider FLOP-aware or Workflow-aware only if benchmarks show clear benefit for your specific access patterns.

### 4.2 Tensor Parallelism Compatibility

| Strategy | Compatibility | Pros | Cons |
|----------|---------------|------|------|
| **LRU** | ✅ Works | Simple; deterministic across ranks | Suboptimal for variable-length prompts; no workload awareness |
| **FLOP-Aware** | ✅ Works | Better memory efficiency; prioritizes high-value entries | Requires tracking per-entry statistics; adds overhead |
| **Workflow-Aware** | ✅ Works | Optimal for agentic workloads | Requires workflow metadata; complex to implement |
| **Staleness-Aware** | ✅ Works | Adapts to changing access patterns | Requires temporal tracking; tuning decay parameters |

**Key constraint**: All strategies must be **deterministic** - given identical access patterns, all ranks must make identical eviction decisions to maintain cache coherence.

**Recommendation for Tensor Parallelism**: Same as local - LRU is sufficient for general workloads. The key requirement is determinism, which all strategies satisfy when operating on identical token sequences.

### 4.3 Pipeline Parallelism Compatibility

| Strategy | Compatibility | Pros | Cons |
|----------|---------------|------|------|
| **LRU** | ⚠️ Entry rank only | Simple at coordination point | Other ranks have no eviction context; must follow leader |
| **FLOP-Aware** | ⚠️ Entry rank only | Can optimize based on prompt lengths | Statistics only available at entry rank |
| **Workflow-Aware** | ⚠️ Entry rank only | Good for multi-agent scenarios | Workflow context only at entry rank |
| **Staleness-Aware** | ⚠️ Entry rank only | Adapts to access patterns | Access patterns only visible at entry rank |

**Key constraint**: In pipeline mode, only the entry rank has visibility into access patterns. All eviction decisions must be made at the entry rank and broadcast to other stages.

**Recommendation for Pipeline Parallelism**: Strategy choice is secondary to the coordination mechanism. Entry rank runs the eviction policy; other ranks follow. Start with LRU for simplicity; upgrade to FLOP-aware or Workflow-aware based on measured workload characteristics.

### 4.4 Summary Matrix

| Strategy | Local | Tensor Parallel | Pipeline Parallel | Workload Fit |
|----------|-------|-----------------|-------------------|--------------|
| **LRU** | ✅ Optimal | ✅ Native | ⚠️ Centralized | General purpose |
| **FLOP-Aware** | ⚠️ Consider | ✅ Native | ⚠️ Centralized | Variable prompt lengths |
| **Workflow-Aware** | ⚠️ Consider | ✅ Native | ⚠️ Centralized | Agentic/multi-turn |
| **Staleness-Aware** | ⚠️ Consider | ✅ Native | ⚠️ Centralized | High request churn |

**Key observations**:
- **Local**: LRU is sufficient for most workloads; advanced strategies add complexity without distributed coordination benefits
- **Tensor Parallel**: All strategies work natively due to deterministic operations on identical inputs
- **Pipeline Parallel**: All strategies require centralization at entry rank; choice depends on workload characteristics

---

## Part 5: Library Availability

### 5.1 Platform Constraints

**Critical finding**: Most production KV cache libraries target NVIDIA GPUs and are **incompatible with MLX/Apple Silicon**.

| Library | Platform | MLX Compatible | Notes |
|---------|----------|----------------|-------|
| [LMCache](https://github.com/LMCache/LMCache) | NVIDIA CUDA 12.1+, x86 only | ❌ No | Requires CUDA kernels; ARM64 not supported |
| [SGLang RadixAttention](https://github.com/sgl-project/sglang) | NVIDIA primarily | ❌ No | Embedded in SGLang runtime; not standalone |
| [vLLM Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/) | NVIDIA primarily | ❌ No | Embedded in vLLM; hash-based block caching |

### 5.2 MLX-Compatible Options

| Library | Type | Maturity | Install | Notes |
|---------|------|----------|---------|-------|
| [mlx-lm](https://github.com/ml-explore/mlx-lm) (built-in) | LRU Trie | ✅ Production | `pip install mlx-lm` | Current implementation; prompt cache files |
| [mlx-textgen](https://github.com/nath1295/MLX-Textgen) | Disk-based slots | ⚠️ Experimental | `pip install mlx-textgen` | Multiple cache slots; persists to disk |
| [pygtrie](https://github.com/google/pygtrie) | Radix/Prefix Tree | ✅ Production | `pip install pygtrie` | Google-maintained; pure Python |
| [datrie](https://github.com/pytries/datrie) | Trie (C-based) | ✅ Production | `pip install datrie` | Fast; uses libdatrie |
| [pypruningradixtrie](https://github.com/otto-de/PyPruningRadixTrie) | Radix Tree | ⚠️ Niche | `pip install pypruningradixtrie` | Optimized for autocomplete |

### 5.3 Library Evaluation Matrix

| Library | Local | Tensor Parallel | Pipeline Parallel | Effort | Risk |
|---------|-------|-----------------|-------------------|--------|------|
| **mlx-lm built-in** | ✅ Works | ⚠️ Needs sync verification | ❌ Needs coordination | Low | Low (Apple-maintained) |
| **mlx-textgen** | ✅ Works | ❌ Single-node design | ❌ Single-node design | Medium | Medium (community) |
| **pygtrie** | ✅ Works | ✅ Deterministic ops | ⚠️ Needs coordination | Medium | ⚠️ High (Google abandonment) |
| **datrie** | ✅ Works | ✅ Deterministic ops | ⚠️ Needs coordination | Medium | Medium (libdatrie dep) |
| **Custom P/D** | ✅ Works | ✅ Full control | ✅ Native fit | High | Low (we control it) |

### 5.4 Recommendations by Mode

**Local Operation:**
- **Use mlx-lm built-in** (LRUPromptCache) - already integrated, proven
- Consider mlx-textgen if disk persistence is needed

**Tensor Parallelism:**
- **Use mlx-lm built-in** with sync verification
- Avoid pygtrie dependency due to Google maintenance risk (see: kaniko, etc.)

**Pipeline Parallelism:**
- **Build custom P/D implementation** on existing OOB infrastructure
- Leverages JACCL RDMA with OOB rendezvous (already proven pattern)
- No external dependencies, full control over MLX-specific optimizations

### 5.5 Key Insight

The MLX ecosystem lacks the distributed KV cache infrastructure that NVIDIA platforms enjoy (LMCache, vLLM, SGLang). For distributed MLX inference:

1. **Tensor parallel**: Leverage existing mlx-lm cache with determinism guarantees
2. **Pipeline parallel**: Must build coordination layer; no off-the-shelf solution
3. **General trie operations**: pygtrie is mature but Google-maintained (abandonment risk)

### 5.6 Build vs Buy Recommendation

**Build our own P/D pattern implementation** for pipeline parallelism:

1. **We already have the foundation**: TCPStore-based OOB coordinator with receiver-initiated rendezvous (`app/distributed/oob.py`)

2. **JACCL send/recv CAN work** with OOB coordination:
   - The SIGBUS crashes happen due to timing asymmetry (receiver waiting, sender not ready)
   - Our OOB rendezvous pattern solves exactly this:
     ```python
     # Receiver signals ready via TCPStore
     oob.signal_ready(transfer_id)
     data = mx.distributed.recv_like(template, src=sender, group=jaccl_group)

     # Sender waits for ready signal, then sends
     oob.wait_ready(transfer_id, receiver_rank)
     mx.distributed.send(data, dst=receiver, group=jaccl_group)
     ```
   - This enables **direct RDMA KV cache transfer** over Thunderbolt 5

3. **Synergy with existing work**:
   - JACCL environment setup (`app/distributed/hostfile.py`)
   - Startup order independence (retry with backoff)
   - MLX_METAL_FAST_SYNCH optimization
   - Experience with JACCL timing issues and workarounds

4. **Avoids external dependencies**:
   - No Google maintenance risk (pygtrie)
   - No NVIDIA lock-in (LMCache)
   - Full control over MLX-specific optimizations

---

## Part 6: Custom P/D Implementation Design

Building on our existing JACCL OOB infrastructure (`app/distributed/oob.py`), we can implement a P/D-style cache coordination pattern for pipeline parallelism.

### 6.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TCPStore (OOB Layer)                     │
│  - Cache metadata coordination                              │
│  - Transfer rendezvous (signal_ready/wait_ready)            │
│  - Barrier synchronization                                  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   Entry Rank  │     │  Middle Rank  │     │   Exit Rank   │
│   (N-1)       │     │     ...       │     │     (0)       │
├───────────────┤     ├───────────────┤     ├───────────────┤
│ - Token input │     │ - Hidden recv │     │ - Hidden recv │
│ - Cache index │     │ - Local layers│     │ - Final layers│
│ - Cache lookup│     │ - Hidden send │     │ - Output      │
│ - KV for L0-k │     │ - KV for Lk-m │     │ - KV for Lm-L │
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                    JACCL RDMA Data Plane
              (send/recv with OOB rendezvous)
```

### 6.2 OOB Extensions for Cache Coordination

Extend `OOBCoordinator` with cache-specific operations:

```python
class OOBCoordinator:
    # ... existing methods ...

    def broadcast_cache_decision(
        self,
        request_id: str,
        cache_id: int,      # -1 = no cache hit
        prefix_len: int,    # tokens covered by cache
    ) -> None:
        """Entry rank broadcasts cache lookup result to all pipeline stages."""
        key = f"cache_decision_{request_id}"
        value = f"{cache_id}:{prefix_len}"
        self._store.set(key, value)

    def receive_cache_decision(self, request_id: str) -> tuple[int, int]:
        """Non-entry ranks receive cache decision from entry rank."""
        key = f"cache_decision_{request_id}"
        self._store.wait([key])
        value = self._store.get(key).decode()
        cache_id, prefix_len = value.split(":")
        return int(cache_id), int(prefix_len)

    def register_cache_entry(self, cache_id: int, token_hash: str) -> None:
        """Entry rank registers a new cache entry for future eviction coordination."""
        key = f"cache_registry_{cache_id}"
        self._store.set(key, token_hash)

    def broadcast_eviction(self, cache_id: int) -> None:
        """Entry rank signals all ranks to evict a cache entry."""
        key = f"cache_evict_{cache_id}"
        self._store.set(key, "1")
```

### 6.3 Cache Transfer via JACCL RDMA

For cache warming (prefill on one rank, transfer KV to others), use OOB-coordinated send/recv:

```python
async def transfer_kv_cache(
    oob: OOBCoordinator,
    group: mx.distributed.Group,
    cache_id: int,
    source_rank: int,
    kv_data: mx.array,
) -> mx.array | None:
    """Transfer KV cache between pipeline stages using RDMA with OOB rendezvous."""
    rank = group.rank()
    transfer_id = f"kv_cache_{cache_id}"

    if rank == source_rank:
        # Wait for all receivers to be ready
        for r in range(group.size()):
            if r != source_rank:
                oob.wait_ready(transfer_id, r)

        # Send to each receiver (could be optimized with broadcast)
        for r in range(group.size()):
            if r != source_rank:
                mx.distributed.send(kv_data, dst=r, group=group)
        mx.eval()  # Ensure sends complete

        oob.signal_complete(transfer_id)
        return None
    else:
        # Signal ready to receive
        oob.signal_ready(transfer_id)

        # Receive KV cache via RDMA
        received = mx.distributed.recv_like(kv_data, src=source_rank, group=group)
        mx.eval(received)

        # Wait for sender to confirm completion
        oob.wait_complete(transfer_id, source_rank)
        return received
```

### 6.4 Pipeline Cache Coordinator

```python
class PipelineCacheCoordinator:
    """Coordinates KV cache across pipeline stages using OOB + JACCL RDMA."""

    def __init__(
        self,
        oob: OOBCoordinator,
        group: mx.distributed.Group,
        max_entries: int = 10,
    ):
        self.oob = oob
        self.group = group
        self.rank = group.rank()
        self.world_size = group.size()
        self.is_entry_rank = (self.rank == self.world_size - 1)

        # Entry rank maintains the cache index (trie/LRU)
        if self.is_entry_rank:
            from app.utils.prompt_cache import LRUPromptCache
            self._cache_index = LRUPromptCache(max_size=max_entries)
            self._next_cache_id = 0

        # All ranks maintain local KV storage by cache_id
        self._local_kv: dict[int, list] = {}

    def lookup(self, request_id: str, tokens: list[int]) -> tuple[list | None, list[int]]:
        """
        Lookup cache entry for tokens.

        Entry rank performs lookup and broadcasts decision.
        Other ranks receive decision and return their local KV cache.
        """
        if self.is_entry_rank:
            cache, rest_tokens = self._cache_index.fetch_nearest_cache(tokens)

            if cache is not None:
                # Find cache_id for this entry (would need index tracking)
                cache_id = self._find_cache_id(tokens)
                prefix_len = len(tokens) - len(rest_tokens)
            else:
                cache_id = -1
                prefix_len = 0

            self.oob.broadcast_cache_decision(request_id, cache_id, prefix_len)

            if cache_id >= 0:
                return self._local_kv.get(cache_id), rest_tokens
            return None, tokens
        else:
            cache_id, prefix_len = self.oob.receive_cache_decision(request_id)

            if cache_id >= 0:
                return self._local_kv.get(cache_id), []  # rest_tokens not needed
            return None, []

    def insert(self, request_id: str, tokens: list[int], kv_cache: list) -> None:
        """
        Insert new cache entry after generation.

        Entry rank updates index and assigns cache_id.
        All ranks store their local KV portion.
        """
        if self.is_entry_rank:
            cache_id = self._next_cache_id
            self._next_cache_id += 1

            # Broadcast cache_id to other ranks
            self.oob.broadcast_cache_decision(f"{request_id}_insert", cache_id, len(tokens))

            # Insert into index (entry rank only)
            self._cache_index.insert_cache(tokens, kv_cache)
        else:
            cache_id, _ = self.oob.receive_cache_decision(f"{request_id}_insert")

        # All ranks store their local KV
        self._local_kv[cache_id] = kv_cache
```

---

## Part 7: Recommended Architectures

### 7.1 Local Operation: mlx-lm Built-in Cache

For single-node operation, the existing `LRUPromptCache` from mlx-lm is sufficient:

```python
from app.utils.prompt_cache import LRUPromptCache

cache = LRUPromptCache(max_size=10)
prompt_cache, rest_tokens = cache.fetch_nearest_cache(input_ids)
# ... generate ...
cache.insert_cache(cache_key, prompt_cache)
```

**Why this works**: No coordination needed. Simple, proven, already integrated.

### 7.2 Tensor Parallelism: Synchronized LRU Cache

```python
class TensorParallelCache:
    """
    Each rank maintains identical radix tree structure.
    Implicit sync via deterministic operations on identical token sequences.
    """
    def __init__(self, max_nodes: int = 100):
        self.tree = RadixTree()
        self.max_nodes = max_nodes

    def lookup(self, tokens: list[int]) -> tuple[CacheEntry | None, list[int]]:
        """Find longest matching prefix, return cache and remaining tokens."""
        match = self.tree.match_prefix(tokens)
        if match:
            self.tree.touch(match)  # Update LRU
            return match.cache, tokens[match.prefix_len:]
        return None, tokens

    def insert(self, tokens: list[int], cache: list[KVCache]) -> None:
        """Insert new cache entry, evict LRU if over capacity."""
        self.tree.insert(tokens, cache)
        while self.tree.size > self.max_nodes:
            self.tree.evict_lru()
```

**Why this works**: All ranks see identical tokens → identical tree operations → identical cache state. No coordination needed.

### 7.3 Pipeline Parallelism: Custom P/D Implementation

Use the `PipelineCacheCoordinator` from Part 6, which builds on our existing OOB infrastructure.

**Alternative (simpler, less optimal)**: Coordinated Cache with Entry-Rank Leader

```python
class PipelineParallelCache:
    """
    Entry rank (N-1) manages cache decisions.
    Other ranks receive cache metadata via broadcast.
    """
    def __init__(self, rank: int, world_size: int, group):
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.is_entry_rank = (rank == world_size - 1)

        if self.is_entry_rank:
            self.tree = RadixTree()  # Only entry rank has full tree
        self.local_cache = {}  # All ranks store local KV cache by ID

    def lookup(self, tokens: list[int]) -> tuple[CacheEntry | None, list[int], int]:
        """
        Entry rank: lookup and broadcast decision.
        Other ranks: receive decision and prepare local cache.
        Returns: (cache, remaining_tokens, cache_offset)
        """
        if self.is_entry_rank:
            match = self.tree.match_prefix(tokens)
            cache_id = match.id if match else -1
            prefix_len = match.prefix_len if match else 0

            # Broadcast: [cache_id, prefix_len]
            decision = mx.array([cache_id, prefix_len], dtype=mx.int32)
            mx.distributed.all_sum(decision, group=self.group)  # Entry contributes, others get

            if match:
                return self.local_cache[cache_id], tokens[prefix_len:], prefix_len
            return None, tokens, 0
        else:
            # Receive decision from entry rank
            decision = mx.zeros((2,), dtype=mx.int32)
            decision = mx.distributed.all_sum(decision, group=self.group)
            mx.eval(decision)

            cache_id, prefix_len = int(decision[0]), int(decision[1])

            if cache_id >= 0:
                return self.local_cache[cache_id], None, prefix_len  # tokens not needed
            return None, None, 0
```

**Why this works**: Entry rank is the single source of truth for cache decisions. Other ranks maintain local KV caches indexed by ID, following entry rank's lead.

### 7.4 Hybrid: Mode-Aware Cache Factory

```python
def create_distributed_cache(
    mode: Literal["tensor", "pipeline", "single"],
    rank: int,
    world_size: int,
    group: mx.distributed.Group | None,
    max_entries: int = 10,
) -> DistributedCache:
    """
    Factory function that creates appropriate cache for the parallelism mode.
    Avoids one-size-fits-all compromises.
    """
    if mode == "single" or world_size == 1:
        return LRUPromptCache(max_size=max_entries)

    elif mode == "tensor":
        # All ranks maintain identical state via deterministic operations
        return TensorParallelCache(max_nodes=max_entries)

    elif mode == "pipeline":
        # Entry rank leads, others follow
        return PipelineParallelCache(rank, world_size, group)

    else:
        raise ValueError(f"Unknown distributed mode: {mode}")
```

---

## Part 8: Implementation Priorities

### Phase 1: Correctness (Immediate)

1. **Disable cache in pipeline mode** until coordination is implemented
2. **Verify tensor mode** works with current LRU implementation
3. **Add mode detection** to cache initialization

### Phase 2: Tensor Parallelism Optimization (Short-term)

1. **Upgrade to Radix Tree** for automatic prefix discovery
2. **Ensure deterministic eviction** (same order across ranks)
3. **Benchmark** against current LRU implementation

### Phase 3: Pipeline Parallelism Support (Medium-term)

1. **Implement entry-rank coordination** protocol
2. **Add cache ID broadcast** mechanism
3. **Test with multi-turn conversations**

### Phase 4: Advanced Eviction (Long-term)

1. **FLOP-aware eviction** for variable prompt lengths
2. **Workflow-aware eviction** for agentic workloads
3. **Metrics collection** for eviction strategy tuning

---

## References

### Cache Architectures
- [SGLang RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)
- [LMCache Architecture](https://docs.lmcache.ai/developer_guide/architecture.html)
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)

### Eviction Strategies
- [Marconi: FLOP-aware Eviction](https://arxiv.org/html/2411.19379v2)
- [KVFlow: Workflow-aware Eviction](https://arxiv.org/pdf/2507.07400)

### Distributed Inference
- [TD-Pipe: Pipeline Parallelism Architecture](https://arxiv.org/html/2506.10470v1)
- [DistServe: P/D Disaggregation](https://hao-ai-lab.github.io/blogs/distserve/)
- [vLLM Parallelism Guide](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)

### KV Cache Management
- [Awesome KV Cache Management (Survey)](https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management)
- [LMCache Technical Report](https://lmcache.ai/tech_report.pdf)
