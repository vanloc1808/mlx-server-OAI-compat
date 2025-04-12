import asyncio
import click
import uvicorn
from loguru import logger
import sys
import os
import warnings
from functools import lru_cache
from app.version import __version__
from app.main import setup_server

class Config:
    """Configuration container for server parameters."""
    def __init__(self, model_path, model_type, port, host, max_concurrency, queue_timeout, queue_size):
        self.model_path = model_path
        self.model_type = model_type
        self.port = port
        self.host = host
        self.max_concurrency = max_concurrency
        self.queue_timeout = queue_timeout
        self.queue_size = queue_size


# Configure Loguru once at module level
def configure_logging():
    """Set up optimized logging configuration."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr, 
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "✦ <level>{message}</level>",
        colorize=True,
        level="INFO"
    )

# Apply logging configuration
configure_logging()


@click.group()
@click.version_option(
    version=__version__, 
    message="""
✨ %(prog)s - OpenAI Compatible API Server ✨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 Version: %(version)s
"""
)
def cli():
    """MLX Server - OpenAI Compatible API for MLX models."""
    pass


@lru_cache(maxsize=1)
def get_server_config(model_path, model_type, port, host, max_concurrency, queue_timeout, queue_size):
    """Cache and return server configuration to avoid redundant processing."""
    return Config(
        model_path=model_path,
        model_type=model_type,
        port=port,
        host=host,
        max_concurrency=max_concurrency,
        queue_timeout=queue_timeout,
        queue_size=queue_size
    )


def print_startup_banner(args):
    """Display beautiful startup banner with configuration details."""
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"✨ MLX Server v{__version__} Starting ✨")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"🔮 Model: {args.model_path}")
    logger.info(f"🔮 Model Type: {args.model_type}")
    logger.info(f"🌐 Host: {args.host}")
    logger.info(f"🔌 Port: {args.port}")
    logger.info(f"⚡ Max Concurrency: {args.max_concurrency}")
    logger.info(f"⏱️ Queue Timeout: {args.queue_timeout} seconds")
    logger.info(f"📊 Queue Size: {args.queue_size}")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


@cli.command()
@click.option(
    "--model-path", 
    required=True, 
    help="Path to the model"
)
@click.option(
    "--model-type",
    default="lm",
    type=click.Choice(["lm", "vlm"]),
    help="Type of model to run"
)
@click.option(
    "--port", 
    default=8000, 
    type=int, 
    help="Port to run the server on"
)
@click.option(
    "--host", 
    default="0.0.0.0", 
    help="Host to run the server on"
)
@click.option(
    "--max-concurrency", 
    default=1, 
    type=int, 
    help="Maximum number of concurrent requests"
)
@click.option(
    "--queue-timeout", 
    default=300, 
    type=int, 
    help="Request timeout in seconds"
)
@click.option(
    "--queue-size", 
    default=100, 
    type=int, 
    help="Maximum queue size for pending requests"
)
def launch(model_path, model_type, port, host, max_concurrency, queue_timeout, queue_size):
    """Launch the MLX server with the specified model."""
    try:
        # Get optimized configuration
        args = get_server_config(model_path, model_type, port, host, max_concurrency, queue_timeout, queue_size)
        
        # Display startup information
        print_startup_banner(args)
        
        # Set up and start the server
        config = asyncio.run(setup_server(args))
        logger.info("Server configuration complete.")
        logger.info("Starting Uvicorn server...")
        uvicorn.Server(config).run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user. Exiting...")
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    cli()