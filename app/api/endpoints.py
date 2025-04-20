import json
import logging
import random
import time
from http import HTTPStatus

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.schemas.openai import (
    ChatCompletionRequest, EmbeddingRequest, Embedding,
    ChatCompletionResponse, ChatCompletionChunk, EmbeddingResponse,
    ChatCompletionChoice, Message
)
from app.utils.errors import create_error_response
from app.handler.mlx_lm import MLXLMHandler
from typing import List, Dict, Any, Optional

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
        return JSONResponse(content= create_error_response("Health check failed", "server_error", 500), status_code=500)

@router.get("/v1/queue/stats")
async def queue_stats(raw_request: Request):
    """
    Get queue statistics.
    """
    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(content= create_error_response("Model handler not initialized", "service_unavailable", 503), status_code=503)
    
    try:
        stats = await handler.get_queue_stats()
        return {
            "status": "ok",
            "queue_stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get queue stats: {str(e)}")
        return JSONResponse(content= create_error_response("Failed to get queue stats", "server_error", 500), status_code=500)
        

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """Handle chat completion requests."""
    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(content=create_error_response("Model handler not initialized", "service_unavailable", 503), status_code=503)
    
    try:
        
        # Check if this is a vision request
        is_vision_request = request.is_vision_request()
        
        # If it's a vision request but the handler is MLXLMHandler (text-only), reject it
        if is_vision_request and isinstance(handler, MLXLMHandler):
            return JSONResponse(
                content=create_error_response(
                    "Vision requests are not supported with text-only models. Use a VLM model type instead.", 
                    "unsupported_request", 
                    400
                ), 
                status_code=400
            )
        
        # Process the request based on type
        return await process_vision_request(handler, request) if is_vision_request \
               else await process_text_request(handler, request)
    except Exception as e:
        logger.error(f"Error processing chat completion request: {str(e)}", exc_info=True)
        return JSONResponse(content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
    
@router.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest, raw_request: Request):
    """Handle embedding requests."""
    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(content=create_error_response("Model handler not initialized", "service_unavailable", 503), status_code=503)

    if isinstance(handler, MLXLMHandler):
        try:
            embeddings = await handler.generate_embeddings_response(request)
            return create_response_embeddings(embeddings, request.model)
        except Exception as e:
            logger.error(f"Error processing embedding request: {str(e)}", exc_info=True)
            return JSONResponse(content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
    else:
        return JSONResponse(content= create_error_response("VLM has not supported embeddings yet", "unsupported_request", 400), status_code=400)
    
def create_response_embeddings(embeddings: List[float], model: str) -> EmbeddingResponse:
    embeddings_response = []
    for index, embedding in enumerate(embeddings):
        embeddings_response.append(Embedding(embedding=embedding, index=index))
    return EmbeddingResponse(data=embeddings_response, model=model)

def create_response_chunk(content: str, model: str, is_final: bool = False) -> ChatCompletionChunk:
    """Create a formatted response chunk for streaming."""
    return ChatCompletionChunk(
        id=get_id(),
        created=int(time.time()),
        model=model,
        choices=[ChatCompletionChoice(
            index=0,
            delta={} if is_final else {"content": content},
            finish_reason="stop" if is_final else None
        )]
    )


async def handle_stream_response(generator, model: str):
    """Handle streaming response generation."""
    try:
        async for chunk in generator:
            response_chunk = create_response_chunk(chunk, model)
            yield f"data: {json.dumps(response_chunk.model_dump())}\n\n"
    except Exception as e:
        logger.error(f"Error in stream wrapper: {str(e)}")
        error_response = {"error": {"message": str(e), "type": "server_error", "code": 500}}
        yield f"data: {json.dumps(error_response)}\n\n"
    finally:
        final_chunk = create_response_chunk('', model, is_final=True)
        yield f"data: {json.dumps(final_chunk.model_dump())}\n\n"
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

def format_final_response(content: str, model: str) -> ChatCompletionResponse:
    """Format the final non-streaming response."""
    return ChatCompletionResponse(
        id=get_id(),
        created=int(time.time()),
        model=model,
        choices=[ChatCompletionChoice(
            index=0,
            message=Message(role="assistant", content=content),
            finish_reason="stop"
        )]
    )

def get_id():
    """
    Generate a unique ID for chat completions with timestamp and random component.
    """
    timestamp = int(time.time())
    random_suffix = random.randint(0, 999999)
    return f"chatcmpl-{timestamp}{random_suffix:06d}"