import json
import logging
import random
import time
from http import HTTPStatus

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.schemas.openai import ChatCompletionRequest
from app.utils.errors import create_error_response

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
    """Handle chat completion requests."""
    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(content=create_error_response("Model handler not initialized", "service_unavailable", 503), status_code=503)
    
    try:
        request.fix_message_order()
        return await process_vision_request(handler, request) if request.is_vision_request() \
               else await process_text_request(handler, request)
    except Exception as e:
        logger.error(f"Error processing chat completion request: {str(e)}", exc_info=True)
        return JSONResponse(content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

def create_response_chunk(content: str, model: str, is_final: bool = False):
    """Create a formatted response chunk for streaming."""
    return {
        "id": get_id(),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {} if is_final else {"content": content},
            "finish_reason": "stop" if is_final else None
        }]
    }


async def handle_stream_response(generator, model: str):
    """Handle streaming response generation."""
    try:
        async for chunk in generator:
            yield f"data: {json.dumps(create_response_chunk(chunk, model))}\n\n"
    except Exception as e:
        logger.error(f"Error in stream wrapper: {str(e)}")
        yield f"data: {json.dumps({'error': create_error_response(str(e))})}\n\n"
    finally:
        yield f"data: {json.dumps(create_response_chunk('', model, is_final=True))}\n\n"
        yield "data: [DONE]\n\n"

async def process_vision_request(handler, request: ChatCompletionRequest):
    """Process vision-specific requests."""
    if request.stream:
        return StreamingResponse(
            handle_stream_response(handler.generate_vision_stream(request), request.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )
    return format_final_response(await handler.generate_vision_response(request), request.model)

async def process_text_request(handler, request: ChatCompletionRequest):
    """Process text-only requests."""
    if request.stream:
        return StreamingResponse(
            handle_stream_response(handler.generate_text_stream(request), request.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )
    return format_final_response(await handler.generate_text_response(request), request.model)




def format_final_response(content: str, model: str):
    """Format the final non-streaming response."""
    return {
        "id": get_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }]
    }

def get_id():
    return f"chatcmpl-{int(time.time())}{random.randint(0, 10000)}"