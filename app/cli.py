import asyncio
import click
import uvicorn
from app.main import setup_server

# Use a hardcoded version to avoid import issues
__version__ = "1.0.1"

class Config:
    def __init__(self, model_path, port, host, max_concurrency, queue_timeout, queue_size):
        self.model_path = model_path
        self.port = port
        self.host = host
        self.max_concurrency = max_concurrency
        self.queue_timeout = queue_timeout
        self.queue_size = queue_size

@click.group()
@click.version_option(version=__version__)
def cli():
    """MLX Server - OpenAI Compatible API for MLX models."""
    pass

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
    click.echo(f"Starting MLX server with model: {model_path} on {host}:{port}")
    
    args = Config(
        model_path=model_path,
        port=port,
        host=host,
        max_concurrency=max_concurrency,
        queue_timeout=queue_timeout,
        queue_size=queue_size
    )
    
    config = asyncio.run(setup_server(args))
    uvicorn.Server(config).run()

if __name__ == "__main__":
    cli() 