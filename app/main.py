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
from app.handler.queue import RequestQueue

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
    parser.add_argument("--max-concurrency", type=int, default=1, help="Maximum number of concurrent requests")
    parser.add_argument("--queue-timeout", type=int, default=300, help="Request queue timeout in seconds")
    parser.add_argument("--queue-size", type=int, default=100, help="Maximum queue size for pending requests")
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
            handler = MLXHandler(
                model_path=args.model_path,
                max_concurrency=args.max_concurrency
            )
            # Initialize request queue with timeout and size
            await handler.vision_queue.stop()
            
            # Re-create queue with new parameters
            handler.vision_queue = RequestQueue(
                max_concurrency=args.max_concurrency,
                timeout=args.queue_timeout,
                queue_size=args.queue_size
            )
            
            # Initialize queue
            await handler.initialize_queues()
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
        logger.info("MLX handler initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MLX handler: {str(e)}")
        raise
    yield
    # Shutdown
    logger.info("Shutting down application")
    if handler:
        # Ensure queue is stopped
        await handler.vision_queue.stop()

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

@app.get("/v1/queue/stats")
async def queue_stats():
    """
    Get queue statistics.
    """
    if handler is None:
        raise HTTPException(status_code=503, detail="Model handler not initialized")
    
    try:
        stats = await handler.get_queue_stats()
        return {
            "status": "ok",
            "queue_stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get queue stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get queue stats")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Handle chat completion requests.
    """
    if handler is None:
        raise HTTPException(status_code=503, detail="Model handler not initialized")
    
    try:
        # Process the request to fix message order if needed    
        request.fix_message_order()
        # Check if this is a vision request
        is_vision_request = request.is_vision_request()
        
        if is_vision_request:
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
                        # Send final chunk with finish_reason
                        final_chunk = {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }
                            ]
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
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
        else:
            # Add support for text responses
            logger.info("Processing text-only request")
            
            if request.stream:
                # For streaming, get the generator and wrap it in OpenAI format
                async def stream_wrapper():
                    try:
                        async for chunk in handler.generate_text_stream(request):
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
                        # Send final chunk with finish_reason
                        final_chunk = {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }
                            ]
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
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
                final_response = await handler.generate_text_response(request)
                
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