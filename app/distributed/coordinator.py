"""Coordination protocol for distributed inference.

This module handles synchronization between the HTTP-serving coordinator
(rank 0) and worker ranks that participate in distributed inference.

The core problem: sharded layers use all_sum() which blocks until ALL
ranks participate. Workers must call generate() at the same time as
the coordinator, or the coordinator hangs forever.

Solution: coordinator broadcasts tokens and parameters before each
generate() call. Workers loop waiting for broadcasts and participate
in inference, discarding their output.

CRITICAL IMPLEMENTATION NOTE - WHY WE USE all_sum() INSTEAD OF send()/recv_like():
================================================================================

We use all_sum() as a broadcast mechanism instead of point-to-point send/recv
operations. This is NOT a design choice - it's a WORKAROUND for a known issue.

THE PROBLEM:
    mx.distributed.send() and recv_like() crash with SIGBUS (signal 138) when
    there is asymmetric timing between sender and receiver over JACCL (Thunderbolt
    RDMA). In HTTP serving, workers block on recv_like() waiting for requests while
    the coordinator only sends when an HTTP request arrives - this timing mismatch
    triggers the crash.

    We experienced this firsthand: worker blocked ~17 seconds on recv_like(),
    coordinator called send() after HTTP request arrived, worker crashed with SIGBUS.

    Attempted fixes that DID NOT WORK:
    - Adding mx.eval() after send/recv operations
    - Using mx.synchronize()
    - Passing explicit group= parameter to send/recv
    - Using CPU stream

THIS IS A KNOWN ISSUE:
    - GitHub Issue #1849: "Issue with mx.distributed send and recv"
      https://github.com/ml-explore/mlx/issues/1849
      Reporter experienced identical symptoms - send/recv fails, all_sum works.

    - MLX documentation emphasizes all_sum/all_gather as primary operations.
      WWDC 2025 MLX session only demonstrates all_sum - doesn't mention send/recv.

    - Jeff Geerling reported RDMA crashes during testing:
      https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5

THE SOLUTION:
    Use all_sum() with a zero-contribution pattern:
    - Coordinator contributes actual data
    - Workers contribute zeros
    - Result: data + 0 + 0 + ... = data (everyone gets coordinator's data)

    This works because all_sum() is synchronous by design - all ranks must call
    it together, avoiding the "idle receiver" timing issue. Model sharding
    internally uses all_sum() and works reliably over JACCL.

DO NOT REFACTOR THIS TO USE send()/recv_like() WITHOUT VERIFYING THE UNDERLYING
ISSUE HAS BEEN FIXED IN MLX. The all_sum pattern is intentional and mission-critical.
"""

from typing import Generator

import mlx.core as mx
from loguru import logger
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler

# Maximum prompt tokens to support (padded for recv_like template)
MAX_PROMPT_LENGTH = 32768

# Parameters: [max_tokens, temp, top_p, top_k, min_p, seed, rep_penalty, rep_ctx]
PARAM_COUNT = 8


class DistributedCoordinator:
    """Broadcasts tokens and parameters from rank 0 to workers.

    This class is used by the coordinator (rank 0) to synchronize
    inference requests across all ranks. Before each generate() call,
    the coordinator broadcasts the tokenized prompt and generation
    parameters so workers can participate in the forward passes.

    Usage:
        coordinator = DistributedCoordinator(group)
        coordinator.broadcast_request(tokens, max_tokens=256, ...)
        # Then call generate() - workers are now synchronized
    """

    def __init__(self, group: mx.distributed.Group):
        """Initialize coordinator for the given distributed group.

        Args:
            group: MLX distributed group from mx.distributed.init()
        """
        self.group = group
        self.rank = group.rank()
        self.size = group.size()

        if self.rank != 0:
            raise ValueError("DistributedCoordinator should only be used on rank 0")

    def broadcast_request(
        self,
        tokens: list[int],
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        seed: int = 0,
        repetition_penalty: float = 1.0,
        repetition_context_size: int = 20,
    ) -> None:
        """Broadcast tokenized prompt and parameters to all workers.

        This must be called BEFORE generate() so workers can participate
        in the distributed forward passes.

        Args:
            tokens: List of token IDs (prompt)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            min_p: Min-p sampling parameter
            seed: Random seed for reproducibility
            repetition_penalty: Repetition penalty factor
            repetition_context_size: Context size for repetition penalty
        """
        # CRITICAL: We use all_sum as broadcast instead of send/recv.
        # send()/recv_like() crash with SIGBUS over JACCL when timing is asymmetric.
        # See module docstring for full explanation and issue references.
        #
        # Pattern: rank 0 contributes data, workers contribute zeros
        # Result: data + 0 + 0 = data (everyone gets coordinator's data)

        # Broadcast token length
        length = mx.array([len(tokens)], dtype=mx.int32)
        # all_sum: rank 0 contributes length, others contribute 0
        length_broadcast = mx.distributed.all_sum(length, group=self.group)
        mx.eval(length_broadcast)

        # Pad tokens to fixed size and broadcast
        padded = mx.zeros((MAX_PROMPT_LENGTH,), dtype=mx.int32)
        token_array = mx.array(tokens, dtype=mx.int32)
        if len(tokens) < MAX_PROMPT_LENGTH:
            padded = mx.concatenate([token_array, mx.zeros((MAX_PROMPT_LENGTH - len(tokens),), dtype=mx.int32)])
        else:
            padded = token_array[:MAX_PROMPT_LENGTH]

        tokens_broadcast = mx.distributed.all_sum(padded, group=self.group)
        mx.eval(tokens_broadcast)

        # Broadcast generation parameters
        params = mx.array(
            [
                float(max_tokens),
                temperature,
                top_p,
                float(top_k),
                min_p,
                float(seed),
                repetition_penalty,
                float(repetition_context_size),
            ],
            dtype=mx.float32,
        )
        params_broadcast = mx.distributed.all_sum(params, group=self.group)
        mx.eval(params_broadcast)

        logger.debug(
            f"[Rank 0] Broadcast {len(tokens)} tokens to {self.size - 1} workers"
        )


def run_worker_loop(
    model,
    tokenizer,
    group: mx.distributed.Group,
    max_kv_size: int = 32768,
) -> None:
    """Run the worker inference loop (blocking).

    Workers wait for token broadcasts from the coordinator, then
    participate in distributed inference. Output is discarded since
    only the coordinator returns HTTP responses.

    This function runs forever until the process is killed.

    Args:
        model: Loaded MLX model (sharded)
        tokenizer: Tokenizer for the model
        group: MLX distributed group
        max_kv_size: Maximum KV cache size
    """
    rank = group.rank()
    logger.info(f"[Rank {rank}] Starting worker loop, waiting for coordinator")

    # Zero templates for the all_sum broadcast pattern.
    # Workers contribute zeros; when summed with coordinator's data, result = coordinator's data.
    # DO NOT change to recv_like() - see module docstring for SIGBUS crash details.
    length_template = mx.zeros((1,), dtype=mx.int32)
    token_template = mx.zeros((MAX_PROMPT_LENGTH,), dtype=mx.int32)
    param_template = mx.zeros((PARAM_COUNT,), dtype=mx.float32)

    while True:
        try:
            # CRITICAL: We use all_sum as broadcast instead of recv_like.
            # recv_like() crashes with SIGBUS when we wait here for extended periods.
            # See module docstring for full explanation and GitHub issue references.
            #
            # Pattern: we contribute zeros, coordinator contributes data
            # Result: 0 + data = data (we receive coordinator's data)

            # Receive token length (all_sum with our zeros)
            logger.debug(f"[Rank {rank}] Waiting for token length (all_sum)...")
            length = mx.distributed.all_sum(length_template, group=group)
            mx.eval(length)
            actual_length = int(length[0].item())
            logger.debug(f"[Rank {rank}] Received length: {actual_length}")

            # Receive padded tokens
            logger.debug(f"[Rank {rank}] Waiting for tokens (all_sum)...")
            tokens = mx.distributed.all_sum(token_template, group=group)
            mx.eval(tokens)
            logger.debug(f"[Rank {rank}] Received tokens")

            # Receive generation parameters
            logger.debug(f"[Rank {rank}] Waiting for params (all_sum)...")
            params = mx.distributed.all_sum(param_template, group=group)
            mx.eval(params)
            logger.debug(f"[Rank {rank}] Received params")

            # Extract parameters
            max_tokens = int(params[0].item())
            temperature = float(params[1].item())
            top_p = float(params[2].item())
            top_k = int(params[3].item())
            min_p = float(params[4].item())
            seed = int(params[5].item())
            repetition_penalty = float(params[6].item())
            repetition_context_size = int(params[7].item())

            # Trim tokens to actual length
            input_tokens = tokens[:actual_length].tolist()

            logger.debug(
                f"[Rank {rank}] Received {actual_length} tokens, max_tokens={max_tokens}"
            )

            # Set random seed for deterministic sampling
            mx.random.seed(seed)

            # Create sampler and cache
            sampler = make_sampler(
                temp=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
            )
            prompt_cache = make_prompt_cache(model, max_kv_size)
            logits_processors = make_logits_processors(
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            )

            # Participate in distributed inference
            # Use stream_generate to stay synchronized with coordinator
            for _chunk in stream_generate(
                model,
                tokenizer,
                input_tokens,
                sampler=sampler,
                max_tokens=max_tokens,
                prompt_cache=prompt_cache,
                logits_processors=logits_processors,
            ):
                # Discard output - we're just participating in all_sum
                pass

            logger.debug(f"[Rank {rank}] Completed generation")

        except Exception as e:
            logger.error(f"[Rank {rank}] Error in worker loop: {e}")
            # Continue trying - don't crash on transient errors
            continue
