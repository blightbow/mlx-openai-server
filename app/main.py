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

            from .distributed import run_worker_loop
            from .models.mlx_lm import MLX_LM

            # Check rank before loading model to avoid loading on coordinator
            # (coordinator will load via setup_server -> handler)
            group = mx.distributed.init()
            rank = group.rank()

            if rank != 0:
                # Workers load model and run inference loop
                logger.info(f"[Rank {rank}] Worker mode - loading model shard")
                mlx_lm = MLX_LM(
                    model_path=config.model_path,
                    context_length=config.context_length,
                    trust_remote_code=config.trust_remote_code,
                    chat_template_file=config.chat_template_file,
                    distributed=config.distributed,
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
