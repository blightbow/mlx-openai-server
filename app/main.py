"""CLI entrypoint shim for the MLX OpenAI Server package.

This lightweight module allows running the CLI via ``python -m app.main``
while preserving the same behavior as the installed console script. It
normalizes ``sys.argv`` so a missing subcommand implicitly becomes
``launch`` (backwards compatibility) and delegates to the Click-based
``cli`` command group defined in :mod:`app.cli`.

Examples
--------
Run the default launch flow:

    python -m app.main

Forward explicit arguments to the CLI:

    python -m app.main launch --port 8000

Distributed inference (via mlx.launch):

    mlx.launch --backend jaccl --hostfile cluster.json -- \\
        python -m app.main launch --model-path <model> --distributed=tensor
"""

import os
import sys

import uvicorn
from loguru import logger

from .config import MLXServerConfig
from .server import setup_server
from .version import __version__


def print_startup_banner(config_args):
    """Log a compact startup banner describing the selected config.

    The function emits human-friendly log messages that summarize the
    runtime configuration (model path/type, host/port, concurrency,
    LoRA settings, and logging options). Intended for the user-facing
    startup output only.
    """
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"✨ MLX Server v{__version__} Starting ✨")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"🔮 Model Path: {config_args.model_path}")
    logger.info(f"🔮 Model Type: {config_args.model_type}")
    if config_args.context_length:
        logger.info(f"🔮 Context Length: {config_args.context_length}")
    logger.info(f"🌐 Host: {config_args.host}")
    logger.info(f"🔌 Port: {config_args.port}")
    logger.info(f"⚡ Max Concurrency: {config_args.max_concurrency}")
    logger.info(f"⏱️ Queue Timeout: {config_args.queue_timeout} seconds")
    logger.info(f"📊 Queue Size: {config_args.queue_size}")
    if config_args.model_type in ["image-generation", "image-edit"]:
        logger.info(f"🔮 Quantize: {config_args.quantize}")
        logger.info(f"🔮 Config Name: {config_args.config_name}")
        if config_args.lora_paths:
            logger.info(f"🔮 LoRA Paths: {config_args.lora_paths}")
        if config_args.lora_scales:
            logger.info(f"🔮 LoRA Scales: {config_args.lora_scales}")
    if (
        hasattr(config_args, "disable_auto_resize")
        and config_args.disable_auto_resize
        and config_args.model_type == "multimodal"
    ):
        logger.info("🖼️ Auto-resize: Disabled")
    if config_args.model_type in ["lm", "multimodal"]:
        if config_args.enable_auto_tool_choice:
            logger.info("🔧 Auto Tool Choice: Enabled")
        if config_args.tool_call_parser:
            logger.info(f"🔧 Tool Call Parser: {config_args.tool_call_parser}")
        if config_args.reasoning_parser:
            logger.info(f"🔧 Reasoning Parser: {config_args.reasoning_parser}")
        if config_args.message_converter:
            logger.info(f"🔧 Message Converter: {config_args.message_converter}")
    logger.info(f"📝 Log Level: {config_args.log_level}")
    if config_args.no_log_file:
        logger.info("📝 File Logging: Disabled")
    elif config_args.log_file:
        logger.info(f"📝 Log File: {config_args.log_file}")
    else:
        logger.info("📝 Log File: logs/app.log (default)")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


async def start(config: MLXServerConfig) -> None:
    """Run the ASGI server using the provided configuration.

    This coroutine wires the configuration into the server setup
    routine, logs progress, and starts the Uvicorn server. It handles
    KeyboardInterrupt and logs any startup failures before exiting the
    process with a non-zero code.

    In distributed mode:
    - Rank 0 (coordinator): Serves HTTP and broadcasts tokens to workers
    - Rank > 0 (workers): Load model shard, participate in inference via worker loop
    """
    try:
        # Handle distributed mode: workers run inference loop, not HTTP server
        if config.distributed:
            import mlx.core as mx

            from .distributed import (
                run_worker_loop,
                sync_model_to_workers,
                sync_metadata_to_workers,
                make_distributed_weight_loader,
                validate_memory_for_streaming,
                init_oob,
            )
            from .distributed.file_sync import detect_backend
            from .models.mlx_lm import MLX_LM

            # If hostfile provided, set up JACCL env vars BEFORE mx.distributed.init()
            # This enables direct execution without mlx.launch wrapper
            hosts = None
            explicit_rank = None
            if config.hostfile:
                from .distributed.hostfile import (
                    load_hostfile,
                    setup_jaccl_env,
                    get_oob_host_from_hostfile,
                )

                # Determine rank: CLI > env var
                explicit_rank = config.rank
                if explicit_rank is None:
                    env_rank = os.environ.get("MLX_RANK")
                    if env_rank is not None:
                        explicit_rank = int(env_rank)
                    else:
                        raise ValueError(
                            "--rank is required when using --hostfile "
                            "(or set MLX_RANK environment variable)"
                        )

                # Load hostfile and set up JACCL environment
                hosts = load_hostfile(config.hostfile)
                setup_jaccl_env(hosts, explicit_rank, config.jaccl_port)

                # Initialize OOB BEFORE distributed init when using JACCL backend.
                # OOB provides: 1) Startup gate (workers wait for rank 0)
                #               2) Rendezvous for send/recv during weight streaming
                # Only JACCL needs OOB - Ring has implicit sync, MPI has built-in rendezvous.
                backend = config.backend or detect_backend()
                if backend == "jaccl":
                    # Always use IP from hostfile for OOB - ensures TCPStore only
                    # traverses TB5 link, not public interfaces. Ignore hostname overrides.
                    oob_host = get_oob_host_from_hostfile(hosts)
                    oob_port = config.oob_port or int(os.environ.get("MLX_OOB_PORT", "29400"))
                    world_size = len(hosts)
                    oob = init_oob(explicit_rank, world_size, oob_host, oob_port)
                    logger.info(
                        f"[Rank {explicit_rank}] JACCL OOB initialized on TB5 -> "
                        f"{oob_host}:{oob_port}"
                    )
                else:
                    oob = None
                    logger.info(f"[Rank {explicit_rank}] Backend={backend}, OOB not needed")

            # Initialize distributed group (reads MLX_RANK, MLX_JACCL_COORDINATOR, etc.)
            group = mx.distributed.init()
            rank = group.rank()
            world_size = group.size()

            # Initialize OOB for mlx.launch mode (after distributed init, since
            # mlx.launch handles startup ordering). Skip if already initialized via hostfile.
            # Only JACCL needs OOB - Ring has implicit sync, MPI has built-in rendezvous.
            if not config.hostfile:
                backend = config.backend or detect_backend()
                if backend == "jaccl":
                    oob_host = config.oob_host or os.environ.get("MLX_OOB_HOST")
                    oob_port = config.oob_port or int(os.environ.get("MLX_OOB_PORT", "29400"))
                    if oob_host:
                        oob = init_oob(rank, world_size, oob_host, oob_port)
                        logger.info(f"[Rank {rank}] JACCL OOB initialized -> {oob_host}:{oob_port}")
                    else:
                        oob = None
                        logger.warning(f"[Rank {rank}] JACCL backend but no OOB host configured")
                else:
                    oob = None
                    logger.info(f"[Rank {rank}] Backend={backend}, OOB not needed")

            # Resolve sync mode: auto uses sharded for pipeline, full for tensor
            sync_mode = config.file_sync
            if sync_mode == "auto":
                sync_mode = "sharded" if config.distributed == "pipeline" else "full"

            # Configure logging early for workers so all distributed logs are captured
            # (rank 0 configures via setup_server later)
            if rank != 0:
                from .server import configure_logging

                configure_logging(
                    log_file=config.log_file,
                    no_log_file=config.no_log_file,
                    log_level=config.log_level,
                )

            # For memory mode, create distributed weight loader (no disk sync needed)
            # For disk modes (none/full/sharded), sync files first
            weight_loader = None
            if sync_mode == "memory":
                logger.info(f"[Rank {rank}] Using memory-based weight streaming")

                # Sync metadata files (configs, tokenizer) to workers first.
                # This ensures sharded_load's _download finds local files and
                # doesn't try to download from HuggingFace on workers.
                model_path = sync_metadata_to_workers(
                    config.model_path,
                    group,
                    worker_model_path=config.worker_model_path,
                )
                # Update config with resolved local path for workers
                if rank != 0:
                    config.model_path = str(model_path)

                # Validate memory before starting (fail early if insufficient)
                validate_memory_for_streaming(config.model_path, group)
                weight_loader = make_distributed_weight_loader(
                    group,
                    model_path=config.model_path,
                    distributed_mode=config.distributed,
                )
                # Store on config so setup_server can pass it to handlers
                # ALL ranks must use the same weight_loader for collective ops
                config.weight_loader = weight_loader
            else:
                logger.info(f"[Rank {rank}] Model sync check (mode={sync_mode})")
                try:
                    model_path = sync_model_to_workers(
                        config.model_path,
                        group,
                        mode=sync_mode,
                        worker_model_path=config.worker_model_path,
                    )
                    # Update config with resolved local path for workers
                    if rank != 0:
                        config.model_path = str(model_path)
                    logger.info(f"[Rank {rank}] Model path: {model_path}")
                except Exception as e:
                    logger.error(f"[Rank {rank}] Model sync failed: {e}")
                    raise

            if rank != 0:
                # Workers load model and run inference loop
                logger.info(f"[Rank {rank}] Worker mode - loading model shard")
                mlx_lm = MLX_LM(
                    model_path=config.model_path,
                    context_length=config.context_length,
                    trust_remote_code=config.trust_remote_code,
                    chat_template_file=config.chat_template_file,
                    distributed=config.distributed,
                    weight_loader=weight_loader,
                )
                logger.info(f"[Rank {rank}] Entering inference loop")
                # Worker loop is blocking (runs forever)
                run_worker_loop(
                    model=mlx_lm.model,
                    tokenizer=mlx_lm.tokenizer,
                    group=mlx_lm.group,
                    max_kv_size=mlx_lm.max_kv_size,
                )
                return  # Never reached, but explicit

            # Rank 0 continues to serve HTTP (model loaded in setup_server)
            logger.info(
                f"[Rank 0] Coordinator mode - starting HTTP server "
                f"(distributed={config.distributed})"
            )

        # Display startup information
        print_startup_banner(config)

        # Set up and start the server
        uvconfig = setup_server(config)
        logger.info("Server configuration complete.")
        logger.info("Starting Uvicorn server...")
        server = uvicorn.Server(uvconfig)
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user. Exiting...")
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        sys.exit(1)


def main():
    """Normalize process args and dispatch to the Click CLI.

    This helper gathers command-line arguments, inserts the "launch"
    subcommand when a subcommand is omitted for backwards compatibility,
    and delegates execution to :func:`app.cli.cli` through
    ``cli.main``.
    """
    from .cli import cli

    args = [str(x) for x in sys.argv[1:]]
    # Keep backwards compatibility: Add 'launch' subcommand if none is provided
    if not args or args[0].startswith("-"):
        args.insert(0, "launch")
    cli.main(args=args)


if __name__ == "__main__":
    main()
