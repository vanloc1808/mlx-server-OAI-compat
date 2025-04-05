import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.handler.schema import ChatCompletionRequest

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/health")
async def health(raw_request: Request):
    """
    Health check endpoint.
    """
    try:
        handler = raw_request.app.state.handler
        # Add more health checks here if needed
        return {"status": "ok", "handler_initialized": handler is not None}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/v1/queue/stats")
async def queue_stats(raw_request: Request):
    """
    Get queue statistics.
    """
    handler = raw_request.app.state.handler
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

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """
    Handle chat completion requests.
    """
    handler = raw_request.app.state.handler
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
