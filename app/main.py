from app import MLXHandler
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import argparse 
from app.handler.schema import ChatCompletionRequest
import uvicorn
import logging
import time
from contextlib import asynccontextmanager
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="OAI-compatible proxy")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--model-type", type=str, required=True, help="Type of the model")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    return parser.parse_args()

# Global handler instance
handler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global handler
    args = parse_args()
    try:
        logger.info(f"Initializing MLX handler with model path: {args.model_path}")
        if args.model_type == "mlx_vlm":
            handler = MLXHandler(args.model_path)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
        logger.info("MLX handler initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MLX handler: {str(e)}")
        raise
    yield
    # Shutdown
    logger.info("Shutting down application")

# Create FastAPI app
app = FastAPI(
    title="OpenAI-compatible API",
    description="API for OpenAI-compatible chat completion and text embedding",
    version="0.1",
    lifespan=lifespan
)

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

@app.post("/health")
async def health():
    """
    Health check endpoint.
    """
    try:
        # Add more health checks here if needed
        return {"status": "ok", "handler_initialized": handler is not None}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Handle chat completion requests.
    """
    if handler is None:
        raise HTTPException(status_code=503, detail="Model handler not initialized")
    
    try:
        if request.is_vision_request():
            logger.info("Processing vision request")
            
            if request.stream:
                # For streaming, get the generator and wrap it in OpenAI format
                async def stream_wrapper():
                    try:
                        async for chunk in handler.generate_vision_stream(request):
                            response_chunk = {
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": chunk},
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(response_chunk)}\n\n"
                    except Exception as e:
                        logger.error(f"Error in stream wrapper: {str(e)}")
                        error_chunk = {
                            "error": {
                                "message": str(e),
                                "type": "internal_error"
                            }
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                    finally:
                        yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    stream_wrapper(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"
                    }
                )
            else:
                # For non-streaming, get the complete response
                final_response = await handler.generate_vision_response(request)
                
                # Format the final response in OpenAI's format
                return {
                    "id": "chatcmpl-" + str(int(time.time())),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": final_response
                            },
                            "finish_reason": "stop"
                        }
                    ]
                }
        
        # TODO: handle text request
        raise HTTPException(status_code=501, detail="Text-only requests not yet implemented")
    except Exception as e:
        logger.error(f"Error processing chat completion request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=True
    )