import asyncio
import click
import uvicorn
from loguru import logger
import sys
from app.version import __version__
from app.main import setup_server

class Config:
    def __init__(self, model_path, port, host, max_concurrency, queue_timeout, queue_size):
        self.model_path = model_path
        self.port = port
        self.host = host
        self.max_concurrency = max_concurrency
        self.queue_timeout = queue_timeout
        self.queue_size = queue_size

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

# Configure Loguru
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

@cli.command()
@click.option(
    "--model-path", 
    required=True, 
    help="Path to the model"
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
def launch(model_path, port, host, max_concurrency, queue_timeout, queue_size):
    """Launch the MLX server with the specified model."""
    # Log a startup banner with configuration details
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"✨ MLX Server v{__version__} Starting ✨")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"🔮 Model: {model_path}")
    logger.info(f"🌐 Host: {host}")
    logger.info(f"🔌 Port: {port}")
    logger.info(f"⚡ Max Concurrency: {max_concurrency}")
    logger.info(f"⏱️ Queue Timeout: {queue_timeout} seconds")
    logger.info(f"📊 Queue Size: {queue_size}")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Set up the server configuration
    args = Config(
        model_path=model_path,
        port=port,
        host=host,
        max_concurrency=max_concurrency,
        queue_timeout=queue_timeout,
        queue_size=queue_size
    )
    
    config = asyncio.run(setup_server(args))
    logger.info("Server configuration complete.")
    logger.info("Starting Uvicorn server...")
    uvicorn.Server(config).run()

if __name__ == "__main__":
    cli()