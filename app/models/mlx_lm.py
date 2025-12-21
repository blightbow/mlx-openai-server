import gc
import os
from typing import Any, Callable, Dict, Generator, List, Optional, Union

import mlx.core as mx
from loguru import logger
from mlx_lm.generate import generate, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import load, sharded_load
from outlines.processors import JSONLogitsProcessor

from ..utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer

DEFAULT_TEMPERATURE = os.getenv("DEFAULT_TEMPERATURE", 0.7)
DEFAULT_TOP_P = os.getenv("DEFAULT_TOP_P", 0.95)
DEFAULT_TOP_K = os.getenv("DEFAULT_TOP_K", 20)
DEFAULT_MIN_P = os.getenv("DEFAULT_MIN_P", 0.0)
DEFAULT_SEED = os.getenv("DEFAULT_SEED", 0)
DEFAULT_MAX_TOKENS = os.getenv("DEFAULT_MAX_TOKENS", 8192)
DEFAULT_BATCH_SIZE = os.getenv("DEFAULT_BATCH_SIZE", 32)

class MLX_LM:
    """
    A wrapper class for MLX Language Model that handles both streaming and non-streaming inference.

    This class provides a unified interface for generating text responses from text prompts,
    supporting both streaming and non-streaming modes.

    For distributed inference, the group is initialized BEFORE model loading so that
    sharded_load() can properly distribute weights across ranks.
    """

    def __init__(
        self,
        model_path: str,
        context_length: int = 32768,
        trust_remote_code: bool = False,
        chat_template_file: str = None,
        distributed: str = None,
        weight_loader: Optional[Callable[[str], Dict[str, Any]]] = None,
    ):
        try:
            self.distributed = distributed
            self.group = None
            self.rank = 0

            # CRITICAL: Initialize distributed group BEFORE model loading
            # sharded_load() needs the group to distribute weights correctly
            if self.distributed:
                self.group = mx.distributed.init()
                self.rank = self.group.rank()
                logger.info(
                    f"[Rank {self.rank}] Distributed mode: {self.distributed}, "
                    f"group size: {self.group.size()}"
                )

            self.model, self.tokenizer = self._initialize_model(
                model_path, trust_remote_code, weight_loader
            )
            self.pad_token_id = self.tokenizer.pad_token_id
            self.bos_token = self.tokenizer.bos_token
            self.model_type = self.model.model_type
            self.max_kv_size = context_length
            self.outlines_tokenizer = OutlinesTransformerTokenizer(self.tokenizer)
            if chat_template_file:
                if not os.path.exists(chat_template_file):
                    raise ValueError(f"Chat template file {chat_template_file} does not exist")
                with open(chat_template_file, "r") as f:
                    self.tokenizer.chat_template = f.read()
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")

    def _initialize_model(
        self,
        model_path: str,
        trust_remote_code: bool = False,
        weight_loader: Optional[Callable[[str], Dict[str, Any]]] = None,
    ):
        if self.distributed:
            # sharded_load(path, pipeline_group, tensor_group, weight_loader)
            # - tensor: shard weights within layers -> pass (None, group)
            # - pipeline: shard weights between layers -> pass (group, None)
            try:
                if self.distributed == "pipeline":
                    logger.info(f"[Rank {self.rank}] Using pipeline parallelism")
                    return sharded_load(model_path, self.group, None, weight_loader=weight_loader)
                else:  # tensor (default)
                    logger.info(f"[Rank {self.rank}] Using tensor parallelism")
                    return sharded_load(model_path, None, self.group, weight_loader=weight_loader)
            except ValueError as e:
                error_msg = str(e)
                if "does not support pipelining" in error_msg:
                    raise ValueError(
                        f"Model does not support pipeline parallelism. "
                        f"Try --distributed=tensor instead. See mlx-lm documentation "
                        f"for supported model architectures."
                    ) from e
                elif "does not support any sharding" in error_msg:
                    raise ValueError(
                        f"Model does not support distributed inference. "
                        f"See mlx-lm documentation for supported model architectures."
                    ) from e
                elif "does not support" in error_msg and "tensor" in error_msg.lower():
                    raise ValueError(
                        f"Model does not support tensor parallelism. "
                        f"Try --distributed=pipeline instead. See mlx-lm documentation "
                        f"for supported model architectures."
                    ) from e
                else:
                    raise
        return load(
            model_path,
            lazy=False,
            tokenizer_config={"trust_remote_code": trust_remote_code},
            weight_loader=weight_loader,
        )
        
    def _apply_pooling_strategy(self, embeddings: mx.array) -> mx.array:
        embeddings = mx.mean(embeddings, axis=1)
        return embeddings
    
    def _apply_l2_normalization(self, embeddings: mx.array) -> mx.array:
        l2_norms = mx.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (l2_norms +  1e-8)
        return embeddings
    
    def _batch_process(self, prompts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[List[int]]:
        """Process prompts in batches with optimized tokenization."""
        all_tokenized = []
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            tokenized_batch = []
            
            # Tokenize all prompts in batch
            for p in batch:
                add_special_tokens = self.bos_token is None or not p.startswith(self.bos_token)
                tokens = self.tokenizer.encode(p, add_special_tokens=add_special_tokens)
                tokenized_batch.append(tokens)
            
            # Find max length in batch
            max_length = max(len(tokens) for tokens in tokenized_batch)
            
            # Pad tokens in a vectorized way
            for tokens in tokenized_batch:
                padding = [self.pad_token_id] * (max_length - len(tokens))
                all_tokenized.append(tokens + padding)
        
        return all_tokenized

    def _preprocess_prompt(self, prompt: str) -> List[int]:
        """Tokenize a single prompt efficiently."""
        add_special_tokens = self.bos_token is None or not prompt.startswith(self.bos_token)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        return mx.array(tokens)
    
    def get_model_type(self) -> str:
        return self.model_type
    
    def get_embeddings(
        self, 
        prompts: List[str], 
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = True
    ) -> List[float]:
        """
        Get embeddings for a list of prompts efficiently.
        
        Args:
            prompts: List of text prompts
            batch_size: Size of batches for processing
            
        Returns:
            List of embeddings as float arrays
        """
        # Process in batches to optimize memory usage
        all_embeddings = []
        try:
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                tokenized_batch = self._batch_process(batch_prompts, batch_size)
                
                # Convert to MLX array for efficient computation
                tokenized_batch = mx.array(tokenized_batch)
                
                try:
                    # Compute embeddings for batch
                    batch_embeddings = self.model.model(tokenized_batch)
                    pooled_embedding = self._apply_pooling_strategy(batch_embeddings)
                    if normalize:
                        pooled_embedding = self._apply_l2_normalization(pooled_embedding)
                    all_embeddings.extend(pooled_embedding.tolist())
                finally:
                    # Explicitly free MLX arrays to prevent memory leaks
                    del tokenized_batch
                    if 'batch_embeddings' in locals():
                        del batch_embeddings
                    if 'pooled_embedding' in locals():
                        del pooled_embedding
                    # Force MLX garbage collection
                    mx.clear_cache()
                    gc.collect()
        except Exception as e:
            # Clean up on error
            mx.clear_cache()
            gc.collect()
            raise

        return all_embeddings
        
    def __call__(
        self, 
        messages: List[Dict[str, str]], 
        stream: bool = False, 
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate text response from the model.

        Args:
            messages (List[Dict[str, str]]): List of messages in the conversation.
            stream (bool): Whether to stream the response.
            **kwargs: Additional parameters for generation
                - temperature: Sampling temperature (default: 0.0)
                - top_p: Top-p sampling parameter (default: 1.0)
                - seed: Random seed (default: 0)
                - max_tokens: Maximum number of tokens to generate (default: 256)
        """
        # Set default parameters if not provided
        seed = kwargs.get("seed", DEFAULT_SEED)
        max_tokens = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)
        chat_template_kwargs = kwargs.get("chat_template_kwargs", {})

        sampler_kwargs = {
            "temp": kwargs.get("temperature", DEFAULT_TEMPERATURE),
            "top_p": kwargs.get("top_p", DEFAULT_TOP_P),
            "top_k": kwargs.get("top_k", DEFAULT_TOP_K),
            "min_p": kwargs.get("min_p", DEFAULT_MIN_P)
        }

        repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        repetition_context_size = kwargs.get("repetition_context_size", 20)
        logits_processors = make_logits_processors(repetition_penalty=repetition_penalty, repetition_context_size=repetition_context_size)
        json_schema = kwargs.get("schema", None)
        if json_schema:
            logits_processors.append(
                JSONLogitsProcessor(
                    schema = json_schema,
                    tokenizer = self.outlines_tokenizer,
                    tensor_library_name = "mlx"
                )
            )
        
        mx.random.seed(seed)
        prompt_cache = make_prompt_cache(self.model, self.max_kv_size)

        input_tokens = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )

        # In distributed mode (coordinator/rank 0), broadcast tokens to workers
        # Workers are running the inference loop and waiting for these tokens
        if self.distributed and self.rank == 0:
            from ..distributed import DistributedCoordinator

            coordinator = DistributedCoordinator(self.group)
            coordinator.broadcast_request(
                tokens=input_tokens,
                max_tokens=max_tokens,
                temperature=sampler_kwargs["temp"],
                top_p=sampler_kwargs["top_p"],
                top_k=sampler_kwargs["top_k"],
                min_p=sampler_kwargs["min_p"],
                seed=seed,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            )
            logger.debug(f"[Rank 0] Broadcast {len(input_tokens)} tokens to workers")

        sampler = make_sampler(**sampler_kwargs)

        prompt_tokens = len(input_tokens)

        if not stream:
            return generate(
                self.model,
                self.tokenizer,
                input_tokens,
                sampler=sampler,
                max_tokens=max_tokens,
                prompt_cache=prompt_cache,
                logits_processors=logits_processors
            ), prompt_tokens
        else:
            # Streaming mode: return generator of chunks
            return stream_generate(
                self.model,
                self.tokenizer,
                input_tokens,
                sampler=sampler,
                max_tokens=max_tokens,
                prompt_cache=prompt_cache,
                logits_processors=logits_processors
            ), prompt_tokens