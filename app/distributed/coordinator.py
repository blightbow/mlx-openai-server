"""Coordination protocol for distributed inference.

This module handles synchronization between the HTTP-serving coordinator
(rank 0) and worker ranks that participate in distributed inference.

The core problem: sharded layers use all_sum() which blocks until ALL
ranks participate. Workers must call generate() at the same time as
the coordinator, or the coordinator hangs forever.

Solution: coordinator broadcasts tokens and parameters before each
generate() call. Workers loop waiting for broadcasts and participate
in inference, discarding their output.
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
        # Send token length first
        length = mx.array([len(tokens)], dtype=mx.int32)
        for dst in range(1, self.size):
            mx.distributed.send(length, dst=dst)
        mx.eval(length)  # Ensure length send completes before tokens

        # Pad tokens to fixed size and send
        padded = mx.zeros((MAX_PROMPT_LENGTH,), dtype=mx.int32)
        token_array = mx.array(tokens, dtype=mx.int32)
        # MLX doesn't support slice assignment, so we concatenate
        if len(tokens) < MAX_PROMPT_LENGTH:
            padded = mx.concatenate([token_array, mx.zeros((MAX_PROMPT_LENGTH - len(tokens),), dtype=mx.int32)])
        else:
            padded = token_array[:MAX_PROMPT_LENGTH]

        for dst in range(1, self.size):
            mx.distributed.send(padded, dst=dst)
        mx.eval(padded)  # Ensure token send completes before params

        # Send generation parameters
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
        for dst in range(1, self.size):
            mx.distributed.send(params, dst=dst)
        mx.eval(params)  # Ensure params send completes before generate()

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

    # Templates for recv_like (must match coordinator's send shapes)
    length_template = mx.zeros((1,), dtype=mx.int32)
    token_template = mx.zeros((MAX_PROMPT_LENGTH,), dtype=mx.int32)
    param_template = mx.zeros((PARAM_COUNT,), dtype=mx.float32)

    while True:
        try:
            # Receive token length
            length = mx.distributed.recv_like(length_template, src=0)
            mx.eval(length)
            actual_length = int(length[0].item())

            # Receive padded tokens
            tokens = mx.distributed.recv_like(token_template, src=0)
            mx.eval(tokens)

            # Receive generation parameters
            params = mx.distributed.recv_like(param_template, src=0)
            mx.eval(params)

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
