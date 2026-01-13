"""Command-line interface and helpers for the MLX server.

This module defines the Click command group used by the package and the
``launch`` command which constructs a server configuration and starts
the ASGI server.
"""

import asyncio
import sys

import click
from loguru import logger

from .config import MLXServerConfig
from .handler.parser.factory import PARSER_REGISTRY
from .main import start
from .version import __version__


class UpperChoice(click.Choice):
    """Case-insensitive choice type that returns uppercase values.

    This small convenience subclass normalizes user input in a
    case-insensitive way but returns the canonical uppercase option
    value to callers. It is useful for flags like ``--log-level``
    where the internal representation is uppercased.
    """

    def normalize_choice(self, choice, ctx):
        """Return the canonical uppercase choice or raise BadParameter.

        Parameters
        ----------
        choice:
            Raw value supplied by the user (may be ``None``).
        ctx:
            Click context object (unused here but part of the API).

        Returns
        -------
        str | None
            Uppercased canonical choice, or ``None`` if ``choice`` is
            ``None``.
        """
        if choice is None:
            return None
        upperchoice = choice.upper()
        for opt in self.choices:
            if opt.upper() == upperchoice:
                return upperchoice
        raise click.BadParameter(
            f"invalid choice: {choice!r}. (choose from {', '.join(map(repr, self.choices))})"
        )


# Configure basic logging for CLI (will be overridden by main.py)
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "✦ <level>{message}</level>",
    colorize=True,
    level="INFO",
)


@click.group()
@click.version_option(
    version=__version__,
    message="""
✨ %(prog)s - OpenAI Compatible API Server for MLX models ✨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 Version: %(version)s
""",
)
def cli():
    """Top-level Click command group for the MLX server CLI.

    Subcommands (such as ``launch``) are registered on this group and
    invoked by the console entry point.
    """
    pass


@cli.command()
@click.option(
    "--model-path",
    required=True,
    help="Path to the model (required for lm, multimodal, embeddings, image-generation, image-edit, whisper model types). With `image-generation` or `image-edit` model types, it should be the local path to the model.",
)
@click.option(
    "--model-type",
    default="lm",
    type=click.Choice(
        ["lm", "multimodal", "image-generation", "image-edit", "embeddings", "whisper"]
    ),
    help="Type of model to run (lm: text-only, multimodal: text+vision+audio, image-generation: flux image generation, image-edit: flux image edit, embeddings: text embeddings, whisper: audio transcription)",
)
@click.option(
    "--context-length",
    default=32768,
    type=int,
    help="Context length for language models. Only works with `lm` or `multimodal` model types.",
)
@click.option("--port", default=8000, type=int, help="Port to run the server on")
@click.option("--host", default="0.0.0.0", help="Host to run the server on")
@click.option(
    "--max-concurrency", default=1, type=int, help="Maximum number of concurrent requests"
)
@click.option("--queue-timeout", default=300, type=int, help="Request timeout in seconds")
@click.option("--queue-size", default=100, type=int, help="Maximum queue size for pending requests")
@click.option(
    "--quantize",
    default=8,
    type=int,
    help="Quantization level for the model. Only used for image-generation and image-edit Flux models.",
)
@click.option(
    "--config-name",
    default=None,
    type=click.Choice(["flux-schnell", "flux-dev", "flux-krea-dev", "flux-kontext-dev", "qwen-image", "qwen-image-edit", "z-image-turbo", "fibo"]),
    help="Config name of the model. Only used for image-generation and image-edit models.",
)
@click.option(
    "--lora-paths",
    default=None,
    type=str,
    help="Path to the LoRA file(s). Multiple paths should be separated by commas.",
)
@click.option(
    "--lora-scales",
    default=None,
    type=str,
    help="Scale factor for the LoRA file(s). Multiple scales should be separated by commas.",
)
@click.option(
    "--disable-auto-resize",
    is_flag=True,
    help="Disable automatic model resizing. Only work for Vision Language Models.",
)
@click.option(
    "--log-file",
    default=None,
    type=str,
    help="Path to log file. If not specified, logs will be written to 'logs/app.log' by default.",
)
@click.option(
    "--no-log-file",
    is_flag=True,
    help="Disable file logging entirely. Only console output will be shown.",
)
@click.option(
    "--log-level",
    default="INFO",
    type=UpperChoice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level. Default is INFO.",
)
@click.option(
    "--enable-auto-tool-choice",
    is_flag=True,
    help="Enable automatic tool choice. Only works with language models.",
)
@click.option(
    "--tool-call-parser",
    default=None,
    type=click.Choice(list(PARSER_REGISTRY.keys())),
    help="Specify tool call parser to use instead of auto-detection. Only works with language models.",
)
@click.option(
    "--reasoning-parser",
    default=None,
    type=click.Choice(list(PARSER_REGISTRY.keys())),
    help="Specify reasoning parser to use instead of auto-detection. Only works with language models.",
)
@click.option(
    "--trust-remote-code",
    is_flag=True,
    help="Enable trust_remote_code when loading models. This allows loading custom code from model repositories.",
)
@click.option(
    "--chat-template-file",
    default=None,
    type=str,
    help="Path to a custom chat template file. Only works with language models (lm) and multimodal models.",
)
@click.option(
    "--distributed",
    type=click.Choice(["tensor", "pipeline"]),
    default=None,
    help="Enable distributed inference. 'tensor' shards weights within layers (best for TB5 RDMA). 'pipeline' shards between layers. Requires mlx.launch wrapper.",
)
@click.option(
    "--file-sync",
    type=click.Choice(["none", "full", "sharded", "memory", "auto"]),
    default="none",
    help="Model file sync mode for distributed inference. 'none' (default) assumes paths aligned across ranks. 'full' syncs all files to disk. 'sharded' syncs only needed files to disk. 'memory' streams weights directly to RAM (no worker disk needed). 'auto' uses sharded for pipeline, full for tensor.",
)
@click.option(
    "--worker-model-path",
    default=None,
    type=str,
    help="Override model path for worker ranks. Use 'hf-cache' to store in HuggingFace cache structure. Default: same as --model-path.",
)
@click.option(
    "--backend",
    type=click.Choice(["ring", "mpi", "nccl", "jaccl"]),
    default=None,
    help="Distributed backend (same values as mlx.launch). Auto-detected if not specified. OOB coordination only enabled for 'jaccl'.",
)
@click.option(
    "--oob-host",
    default=None,
    type=str,
    help="Coordinator IP/hostname for JACCL OOB rendezvous (PyTorch TCPStore). Only used when --backend=jaccl. Falls back to MLX_OOB_HOST env var.",
)
@click.option(
    "--oob-port",
    default=29400,
    type=int,
    help="TCP port for JACCL OOB rendezvous. Default: 29400.",
)
@click.option(
    "--hostfile",
    default=None,
    type=str,
    help="Path to mlx.launch format hostfile (JSON). Enables direct execution without mlx.launch wrapper. Use with --rank to specify this node's rank.",
)
@click.option(
    "--rank",
    default=None,
    type=int,
    help="This node's rank for distributed inference. Overrides MLX_RANK env var. Required when using --hostfile.",
)
@click.option(
    "--jaccl-port",
    default=32323,
    type=int,
    help="JACCL coordinator port. Default: 32323.",
)
def launch(
    model_path,
    model_type,
    context_length,
    port,
    host,
    max_concurrency,
    queue_timeout,
    queue_size,
    quantize,
    config_name,
    lora_paths,
    lora_scales,
    disable_auto_resize,
    log_file,
    no_log_file,
    log_level,
    enable_auto_tool_choice,
    tool_call_parser,
    reasoning_parser,
    trust_remote_code,
    chat_template_file,
    distributed,
    file_sync,
    worker_model_path,
    backend,
    oob_host,
    oob_port,
    hostfile,
    rank,
    jaccl_port,
) -> None:
    """Start the FastAPI/Uvicorn server with the supplied flags.

    The command builds a server configuration object using
    ``MLXServerConfig`` and then calls the async ``start`` routine
    which handles the event loop and server lifecycle.
    """

    args = MLXServerConfig(
        model_path=model_path,
        model_type=model_type,
        context_length=context_length,
        port=port,
        host=host,
        max_concurrency=max_concurrency,
        queue_timeout=queue_timeout,
        queue_size=queue_size,
        quantize=quantize,
        config_name=config_name,
        lora_paths_str=lora_paths,
        lora_scales_str=lora_scales,
        disable_auto_resize=disable_auto_resize,
        log_file=log_file,
        no_log_file=no_log_file,
        log_level=log_level,
        enable_auto_tool_choice=enable_auto_tool_choice,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        trust_remote_code=trust_remote_code,
        chat_template_file=chat_template_file,
        distributed=distributed,
        file_sync=file_sync,
        worker_model_path=worker_model_path,
        backend=backend,
        oob_host=oob_host,
        oob_port=oob_port,
        hostfile=hostfile,
        rank=rank,
        jaccl_port=jaccl_port,
    )

    asyncio.run(start(args))
