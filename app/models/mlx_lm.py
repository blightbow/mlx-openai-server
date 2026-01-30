import gc
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Union

import mlx.core as mx
import mlx.utils
from dataclasses import dataclass
from loguru import logger
from mlx_lm.generate import GenerationResponse, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import load, load_config, load_tokenizer
from outlines.processors import JSONLogitsProcessor

from ..utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer
from ..distributed.helpers import synced_all_sum_sync

DEFAULT_TEMPERATURE = os.getenv("DEFAULT_TEMPERATURE", 0.7)
DEFAULT_TOP_P = os.getenv("DEFAULT_TOP_P", 0.95)
DEFAULT_TOP_K = os.getenv("DEFAULT_TOP_K", 20)
DEFAULT_MIN_P = os.getenv("DEFAULT_MIN_P", 0.0)
DEFAULT_XTC_PROBABILITY = os.getenv("DEFAULT_XTC_PROBABILITY", 0.0)
DEFAULT_XTC_THRESHOLD = os.getenv("DEFAULT_XTC_THRESHOLD", 0.0)
DEFAULT_SEED = os.getenv("DEFAULT_SEED", 0)
DEFAULT_MAX_TOKENS = os.getenv("DEFAULT_MAX_TOKENS", 8192)
DEFAULT_BATCH_SIZE = os.getenv("DEFAULT_BATCH_SIZE", 32)

@dataclass
class CompletionResponse:
    """
    The output of :func:`__call__` when stream is False.

    Args:
        text (str): The next segment of decoded text. This can be an empty string.
        tokens (List[int]): The list of tokens in the response.
        peak_memory (float): The peak memory used so far in GB.
        generation_tps (float): The tokens-per-second for generation.
        generation_tokens (int): The number of generated tokens.
        prompt_tps (float): The prompt processing tokens-per-second.
        prompt_tokens (int): The number of tokens in the prompt.
    """

    text: str = None
    tokens: List[int] = None
    peak_memory: float = None
    generation_tps: float = None
    prompt_tps: float = None
    prompt_tokens: int = None
    generation_tokens: int = None

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
        context_length: int | None = None,
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
            self.context_length = context_length
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
        if self.distributed and weight_loader is not None:
            # Distributed loading with custom weight_loader (e.g., streaming over TB5)
            # We handle this directly instead of using sharded_load because:
            # 1. Weight files don't exist locally on workers
            # 2. We control file selection via the weight_loader
            # 3. sharded_load assumes local files for its probe model
            return self._distributed_load_with_weight_loader(
                model_path, trust_remote_code, weight_loader
            )
        elif self.distributed:
            # Standard distributed loading (files exist locally)
            from mlx_lm.utils import sharded_load
            try:
                if self.distributed == "pipeline":
                    logger.info(f"[Rank {self.rank}] Using pipeline parallelism")
                    return sharded_load(model_path, self.group, None)
                else:  # tensor (default)
                    logger.info(f"[Rank {self.rank}] Using tensor parallelism")
                    return sharded_load(model_path, None, self.group)
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
        # Non-distributed mode: standard mlx_lm.load()
        return load(
            model_path,
            lazy=False,
            tokenizer_config={"trust_remote_code": trust_remote_code},
        )

    def _distributed_load_with_weight_loader(
        self,
        model_path: str,
        trust_remote_code: bool,
        weight_loader: Callable[[str], Dict[str, Any]],
    ):
        """Load model for distributed inference with custom weight_loader.

        This re-implements mlx_lm's load_model logic because the upstream version
        assumes weights exist on local disk (uses glob + mx.load). Our weight_loader
        streams weights via RDMA from rank 0, so workers don't have weight files.

        The approach:
        1. Use mlx_lm's load_config() - works because workers have config files
        2. Import model architecture dynamically (same as mlx_lm._get_classes)
        3. Call weight_loader for each file to collect weights via RDMA
        4. Instantiate model with config
        5. Handle sanitization and quantization
        6. Load weights into model
        7. Configure sharding (pipeline or tensor)
        """
        import importlib
        import mlx.nn as nn

        logger.info(f"[Rank {self.rank}] Using {self.distributed} parallelism with weight streaming")

        model_path = Path(model_path)

        # Get file list from weight_loader's manifest (already exchanged via OOB)
        from ..distributed.file_sync import get_weight_loader_file_list
        all_weight_files = get_weight_loader_file_list(weight_loader)

        if all_weight_files is None:
            # Fallback: read from index file (shouldn't happen with proper setup)
            index_path = model_path / "model.safetensors.index.json"
            if not index_path.exists():
                raise ValueError(
                    f"Distributed loading requires model.safetensors.index.json at {model_path}"
                )
            with open(index_path, "r") as f:
                weight_index = json.load(f)["weight_map"]
            all_weight_files = sorted(set(weight_index.values()))
            logger.warning(f"[Rank {self.rank}] Using fallback file list from index")

        weight_file_paths = [str(model_path / f) for f in all_weight_files]
        logger.info(f"[Rank {self.rank}] Processing {len(weight_file_paths)} weight files via streaming")

        # Load tokenizer (config files were synced to workers)
        tokenizer = load_tokenizer(
            model_path,
            {"trust_remote_code": trust_remote_code},
        )

        # Step 1: Load config (workers have config files via sync_metadata_to_workers)
        config = load_config(model_path)

        # Step 2: Get model architecture classes (same logic as mlx_lm._get_classes)
        MODEL_REMAPPING = {
            "mistral": "llama",
            "phi-msft": "phixtral",
        }
        model_type = config["model_type"]
        model_type = MODEL_REMAPPING.get(model_type, model_type)
        try:
            arch = importlib.import_module(f"mlx_lm.models.{model_type}")
        except ImportError:
            raise ValueError(f"Model type {model_type} not supported by mlx_lm")
        model_class = arch.Model
        model_args_class = arch.ModelArgs

        # Step 3: Load weights via weight_loader (streams from rank 0 via RDMA)
        logger.info(f"[Rank {self.rank}] Loading weights via distributed streaming...")
        weights = {}
        for i, wf in enumerate(weight_file_paths):
            logger.debug(f"[Rank {self.rank}] Loading weight file {i+1}/{len(weight_file_paths)}: {Path(wf).name}")
            file_weights = weight_loader(wf)
            weights.update(file_weights)
            # Clear intermediate to reduce memory pressure
            del file_weights

        logger.info(f"[Rank {self.rank}] Loaded {len(weights)} weight tensors")

        # Step 4: Instantiate model
        model_args = model_args_class.from_dict(config)
        model = model_class(model_args)

        # Step 5: Sanitize weights if model supports it
        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)

        # Step 6: Handle quantization (same logic as mlx_lm.load_model)
        if (quantization := config.get("quantization", None)) is not None:
            def class_predicate(p, m):
                if p in config["quantization"]:
                    return config["quantization"][p]
                if not hasattr(m, "to_quantized"):
                    return False
                return f"{p}.scales" in weights

            nn.quantize(
                model,
                group_size=quantization["group_size"],
                bits=quantization["bits"],
                mode=quantization.get("mode", "affine"),
                class_predicate=class_predicate,
            )

        # Step 7: Load weights into model
        model.load_weights(list(weights.items()), strict=False)
        del weights  # Free weight dict

        model.eval()

        # CHECKPOINT: After load_model, before sharding
        from ..distributed.file_sync import get_available_memory
        mem_after_load = get_available_memory()
        logger.info(f"[Rank {self.rank}] MEMORY after load_model: {mem_after_load / 1e9:.1f}GB available")

        # Configure sharding based on distributed mode
        if self.distributed == "tensor":
            if not hasattr(model, "shard"):
                raise ValueError(
                    "Model does not support tensor parallelism. "
                    "Try --distributed=pipeline instead."
                )
            model.shard(self.group)
            # CHECKPOINT
            mem_after_shard = get_available_memory()
            logger.info(f"[Rank {self.rank}] MEMORY after shard(): {mem_after_shard / 1e9:.1f}GB available")
        else:  # pipeline
            # Configure pipeline parallelism via mlx_lm's built-in method
            # This routes the forward pass so each rank only executes its layers
            if not hasattr(model, "model") or not hasattr(model.model, "pipeline"):
                raise ValueError(
                    "Model does not support pipeline parallelism. "
                    "Try --distributed=tensor instead."
                )
            model.model.pipeline(self.group)

            mem_after_pipeline = get_available_memory()
            logger.info(f"[Rank {self.rank}] MEMORY after pipeline(): {mem_after_pipeline / 1e9:.1f}GB available")

            # INCREMENTAL PARAMETER EVALUATION for pipeline mode.
            # Running model(warmup_tokens) + mx.eval() causes a memory spike
            # because ALL lazy tensors are evaluated at once. On rank 1
            # (which receives weights via RDMA), this can cause 2x memory:
            # lazy arrays + evaluated arrays coexisting temporarily.
            #
            # Solution: Evaluate parameters in small batches to spread out
            # memory allocation. This forces lazy arrays to materialize
            # incrementally before the first inference request.
            logger.info(f"[Rank {self.rank}] Starting incremental parameter evaluation...")
            self._incremental_eval_parameters(model)

            # Synchronize all ranks via OOB barrier + all_sum.
            # JACCL requires ranks to enter collective ops together.
            from ..distributed.oob import get_oob
            oob = get_oob()
            synced_all_sum_sync(mx.array(1.0), self.group, "pre_pipeline_sync", oob=oob)
            logger.info(f"[Rank {self.rank}] Pipeline setup complete, barrier passed")

            mem_after_eval = get_available_memory()
            logger.info(f"[Rank {self.rank}] MEMORY after incremental eval: {mem_after_eval / 1e9:.1f}GB available (delta: {(mem_after_pipeline - mem_after_eval) / 1e9:.1f}GB)")

        # Synchronize all ranks before returning.
        # OOB barrier first to ensure JACCL collective ops are entered together.
        from ..distributed.oob import get_oob
        oob = get_oob()
        synced_all_sum_sync(mx.array(1.0), self.group, "pre_final_sync", oob=oob)

        # Diagnostic: verify parameters are materialized (skip for pipeline to avoid memory spike)
        if self.distributed != "pipeline":
            params = [v for _, v in mlx.utils.tree_flatten(model.parameters())]
            param_count = sum(p.size for p in params)
            # Check if first param has actual data (not just lazy placeholder)
            if params:
                first_param = params[0]
                # Accessing .item() on first element forces evaluation if lazy
                try:
                    sample_val = first_param.flatten()[0].item()
                    logger.info(f"[Rank {self.rank}] Model loaded: {param_count:,} params, sample={sample_val:.6f}")
                except Exception as e:
                    logger.warning(f"[Rank {self.rank}] Model loaded: {param_count:,} params, but sample access failed: {e}")
        else:
            logger.info(f"[Rank {self.rank}] Model loaded (parameters evaluated incrementally)")

        logger.info(f"[Rank {self.rank}] Model loaded and sharded successfully")

        return model, tokenizer

    def _incremental_eval_parameters(
        self,
        model,
        batch_size_gb: float = 8.0,
    ) -> None:
        """Evaluate model parameters incrementally to avoid memory spikes.

        After load_weights(), model parameters may be lazy arrays. Evaluating
        them all at once (e.g., via warmup forward pass) causes 2x memory:
        lazy source + evaluated destination coexist during evaluation.

        This method evaluates parameters in small batches, allowing the lazy
        source to be freed before the next batch starts. This spreads memory
        allocation over time and prevents the spike.

        Args:
            model: The loaded model
            batch_size_gb: Target batch size in GB (default 8GB)
        """
        from ..distributed.file_sync import get_available_memory

        # Flatten all parameters (tree_flatten returns list of (key, value) tuples for dicts)
        params = [v for _, v in mlx.utils.tree_flatten(model.parameters())]
        if not params:
            logger.info(f"[Rank {self.rank}] No parameters to evaluate")
            return

        # Check how many parameters might be lazy (heuristic: check if they have data)
        # Note: This is diagnostic only - we evaluate all params regardless
        lazy_count = 0
        for p in params[:min(10, len(params))]:  # Sample first 10
            try:
                # Accessing data triggers evaluation if lazy
                _ = p.flatten()[0].item()
            except Exception:
                lazy_count += 1

        if lazy_count > 0:
            logger.info(
                f"[Rank {self.rank}] Detected potentially lazy params "
                f"({lazy_count}/{min(10, len(params))} sampled)"
            )

        # Calculate sizes and sort by size (evaluate largest first to catch issues early)
        param_sizes = [(p, p.nbytes) for p in params]
        param_sizes.sort(key=lambda x: x[1], reverse=True)

        total_bytes = sum(size for _, size in param_sizes)
        batch_target_bytes = int(batch_size_gb * 1e9)

        logger.info(
            f"[Rank {self.rank}] Incremental eval: {len(params)} params, "
            f"{total_bytes / 1e9:.1f}GB total, {batch_size_gb}GB batch target"
        )

        # Evaluate in batches
        batch = []
        batch_bytes = 0
        batches_evaluated = 0
        params_evaluated = 0
        mem_before = get_available_memory()

        for param, size in param_sizes:
            # Add to current batch
            batch.append(param)
            batch_bytes += size

            # Evaluate when batch is full enough
            if batch_bytes >= batch_target_bytes:
                mx.eval(*batch)
                mx.clear_cache()  # Free any intermediate allocations
                batches_evaluated += 1
                params_evaluated += len(batch)

                if batches_evaluated % 5 == 0:
                    mem_now = get_available_memory()
                    logger.debug(
                        f"[Rank {self.rank}] Evaluated batch {batches_evaluated}: "
                        f"{params_evaluated}/{len(params)} params, "
                        f"memory: {mem_now / 1e9:.1f}GB available"
                    )

                batch = []
                batch_bytes = 0

        # Evaluate remaining parameters
        if batch:
            mx.eval(*batch)
            batches_evaluated += 1
            params_evaluated += len(batch)

        mx.clear_cache()  # Final cleanup
        mem_after = get_available_memory()
        logger.info(
            f"[Rank {self.rank}] Incremental eval complete: {batches_evaluated} batches, "
            f"{params_evaluated} params, memory delta: {(mem_before - mem_after) / 1e9:.1f}GB"
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

    def create_prompt_cache(self) -> List[Any]:
        return make_prompt_cache(self.model, max_kv_size=self.context_length)
        
    def get_model_type(self) -> str:
        return self.model_type

    def create_input_prompt(self, messages: List[Dict[str, str]], chat_template_kwargs: Dict[str, Any]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )

    def encode_prompt(self, input_prompt: str) -> List[int]:
        add_special_tokens = self.tokenizer.bos_token is None or not input_prompt.startswith(
            self.tokenizer.bos_token
        )
        return self.tokenizer.encode(input_prompt, add_special_tokens=add_special_tokens)

    def __call__(
        self, 
        input_ids: List[int],
        prompt_cache: List[Any] = None,
        stream: bool = False, 
        **kwargs
    ) -> Union[CompletionResponse, Generator[GenerationResponse, None, None]]:
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
        seed = kwargs.get("seed")
        max_tokens = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)

        sampler_kwargs = {
            "temp": kwargs.get("temperature", DEFAULT_TEMPERATURE),
            "top_p": kwargs.get("top_p", DEFAULT_TOP_P),
            "top_k": kwargs.get("top_k", DEFAULT_TOP_K),
            "min_p": kwargs.get("min_p", DEFAULT_MIN_P),
            "xtc_probability": kwargs.get("xtc_probability", DEFAULT_XTC_PROBABILITY),
            "xtc_threshold": kwargs.get("xtc_threshold", DEFAULT_XTC_THRESHOLD),
        }

        # Add XTC special tokens (EOS and newline) when XTC is enabled
        if sampler_kwargs["xtc_probability"] > 0:
            sampler_kwargs["xtc_special_tokens"] = [
                self.tokenizer.eos_token_id
            ] + self.tokenizer.encode("\n")

        repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        repetition_context_size = kwargs.get("repetition_context_size", 20)
        logit_bias = kwargs.get("logit_bias", None)
        # Convert string keys to int if logit_bias is provided (OpenAI API uses string keys)
        if logit_bias is not None:
            logit_bias = {int(k): v for k, v in logit_bias.items()}
        logits_processors = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size
        )
        json_schema = kwargs.get("schema", None)
        if json_schema:
            logits_processors.append(
                JSONLogitsProcessor(
                    schema = json_schema,
                    tokenizer = self.outlines_tokenizer,
                    tensor_library_name = "mlx"
                )
            )

        # Only seed RNG when an explicit non-negative seed is provided
        # None or negative values (e.g., -1) result in non-deterministic generation
        if seed is not None and seed >= 0:
            mx.random.seed(seed)
        
        prompt_progress_callback = kwargs.get("prompt_progress_callback")

        # In distributed mode (coordinator/rank 0), broadcast tokens to workers
        # Workers are running the inference loop and waiting for these tokens
        if self.distributed and self.rank == 0:
            from ..distributed import DistributedCoordinator

            coordinator = DistributedCoordinator(self.group)
            coordinator.broadcast_request(
                tokens=input_ids,
                max_tokens=max_tokens,
                temperature=sampler_kwargs["temp"],
                top_p=sampler_kwargs["top_p"],
                top_k=sampler_kwargs["top_k"],
                min_p=sampler_kwargs["min_p"],
                seed=seed,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            )
            logger.debug(f"[Rank 0] Broadcast {len(input_ids)} tokens to workers")

        sampler = make_sampler(**sampler_kwargs)

        stream_response = stream_generate(
            self.model,
            self.tokenizer,
            input_ids,
            sampler=sampler,
            max_tokens=max_tokens,
            prompt_cache=prompt_cache,
            logits_processors=logits_processors,
            prompt_progress_callback=prompt_progress_callback
        )
        if stream:
            return stream_response

        text = ""
        tokens = []
        final_chunk = None
        for chunk in stream_response:
            text += chunk.text
            tokens.append(chunk.token)
            if chunk.finish_reason:
                final_chunk = chunk
        
        return CompletionResponse(
            text=text,
            tokens=tokens,
            peak_memory=final_chunk.peak_memory,
            generation_tps=final_chunk.generation_tps,
            prompt_tps=final_chunk.prompt_tps,
            prompt_tokens=final_chunk.prompt_tokens,
            generation_tokens=final_chunk.generation_tokens,
        )