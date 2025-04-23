import argparse
import asyncio
import gc
import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.handler.mlx_vlm import MLXVLMHandler
from app.handler.mlx_lm import MLXLMHandler 
from app.api.endpoints import router
from app.version import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="OAI-compatible proxy")
    parser.add_argument("--model-path", type=str, required=True, help="Huggingface model repo or local path")
    parser.add_argument("--model-type", type=str, default="lm", choices=["lm", "vlm"], help="Model type")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--max-concurrency", type=int, default=1, help="Maximum number of concurrent requests")
    parser.add_argument("--queue-timeout", type=int, default=300, help="Request timeout in seconds")
    parser.add_argument("--queue-size", type=int, default=100, help="Maximum queue size for pending requests")
    return parser.parse_args()


def create_lifespan(config_args):
    """Factory function to create a lifespan context manager with access to config args."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            logger.info(f"Initializing MLX handler with model path: {config_args.model_path}")
            if config_args.model_type == "vlm":
                handler = MLXVLMHandler(
                    model_path=config_args.model_path,
                    max_concurrency=config_args.max_concurrency
                )
            else:
                handler = MLXLMHandler(
                    model_path=config_args.model_path,
                    max_concurrency=config_args.max_concurrency
                )       
            # Initialize queue
            await handler.initialize({
                "max_concurrency": config_args.max_concurrency,
                "timeout": config_args.queue_timeout,
                "queue_size": config_args.queue_size
            })
            logger.info("MLX handler initialized successfully")
            app.state.handler = handler
        except Exception as e:
            logger.error(f"Failed to initialize MLX handler: {str(e)}")
            raise
        gc.collect()
        yield
        # Shutdown
        logger.info("Shutting down application")
        if hasattr(app.state, "handler") and app.state.handler:
            try:
                # Use the proper cleanup method which handles both request queue and image processor
                logger.info("Cleaning up resources")
                await app.state.handler.cleanup()
                logger.info("Resources cleaned up successfully")
            except Exception as e:
                logger.error(f"Error during shutdown: {str(e)}")
    
    return lifespan

# App instance will be created during setup with the correct lifespan
app = None

async def setup_server(args) -> uvicorn.Config:
    global app
    
    # Create FastAPI app with the configured lifespan
    app = FastAPI(
        title="OpenAI-compatible API",
        description="API for OpenAI-compatible chat completion and text embedding",
        version=__version__,
        lifespan=create_lifespan(args)
    )
    
    app.include_router(router)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Global exception handler caught: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "internal_error"}}
        )
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=True
    )
    return config

if __name__ == "__main__":
    args = parse_args()
    config = asyncio.run(setup_server(args))
    uvicorn.Server(config).run()