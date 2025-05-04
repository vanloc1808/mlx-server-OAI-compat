import json
import logging
import random
import time
from http import HTTPStatus

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.schemas.openai import (
    EmbeddingRequest, Embedding, EmbeddingResponse, 
    ChatCompletionRequest, ChatCompletionChunk, Choice, Message, FunctionCall, StreamingChoice,
    ChatCompletionResponse, ChatCompletionMessageToolCall, ChoiceDeltaToolCall, ChoiceDeltaFunctionCall, Delta
)
from app.utils.errors import create_error_response
from app.handler.mlx_lm import MLXLMHandler
from typing import List, Dict, Any, Optional, Union

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

    try:
        embeddings = await handler.generate_embeddings_response(request)
        return create_response_embeddings(embeddings, request.model)
    except Exception as e:
        logger.error(f"Error processing embedding request: {str(e)}", exc_info=True)
        return JSONResponse(content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
    
def create_response_embeddings(embeddings: List[float], model: str) -> EmbeddingResponse:
    embeddings_response = []
    for index, embedding in enumerate(embeddings):
        embeddings_response.append(Embedding(embedding=embedding, index=index))
    return EmbeddingResponse(data=embeddings_response, model=model)

def create_response_chunk(chunk: Union[str, Dict[str, Any]], model: str, is_final: bool = False, finish_reason: Optional[str] = "stop") -> ChatCompletionChunk:
    """Create a formatted response chunk for streaming."""
    if isinstance(chunk, str):
        return ChatCompletionChunk(
            id=get_id(),
            object="chat.completion.chunk",
            created=int(time.time()),
            model=model,
            choices=[StreamingChoice(
                index=0,
                delta=Delta(content=chunk, role="assistant"),
                finish_reason=finish_reason if is_final else None
            )]
        )
    if "name" in chunk:
        tool_chunk = ChoiceDeltaToolCall(
            index=chunk["index"],
            type="function",
            id=get_tool_call_id(),
            function=ChoiceDeltaFunctionCall(
                name=chunk["name"],
                arguments=""
            )
        )
    else:
        tool_chunk = ChoiceDeltaToolCall(
            index=chunk["index"],
            function= ChoiceDeltaFunctionCall(
                arguments=chunk["arguments"]
            )
        )
    delta = Delta(
        content = None,
        function_call = tool_chunk.function,
        role = "assistant",
        tool_calls = [tool_chunk]
    )
    return ChatCompletionChunk(
        id=get_id(),
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=[StreamingChoice(index=0, delta=delta, finish_reason=None)]
    )


async def handle_stream_response(generator, model: str):
    """Handle streaming response generation."""
    try:
        finish_reason = "stop"
        index = 0   
        async for chunk in generator:
            if chunk:
                if isinstance(chunk, str):
                    response_chunk = create_response_chunk(chunk, model)
                    yield f"data: {json.dumps(response_chunk.model_dump())}\n\n"
                else:
                    finish_reason = "function_call"
                    function = {
                        "index": index,
                        "name": chunk["name"],
                    }
                    response_chunk = create_response_chunk(function, model)
                    yield f"data: {json.dumps(response_chunk.model_dump())}\n\n"
                    function_call = {
                        "index": index,
                        "arguments": json.dumps(chunk["arguments"])
                    }
                    response_chunk = create_response_chunk(function_call, model)
                    yield f"data: {json.dumps(response_chunk.model_dump())}\n\n"
                    index += 1

    except Exception as e:
        logger.error(f"Error in stream wrapper: {str(e)}")
        error_response = create_error_response(str(e), "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
        yield f"data: {json.dumps(error_response)}\n\n"
    finally:
        final_chunk = create_response_chunk('', model, is_final=True, finish_reason=finish_reason)
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

def get_id():
    """
    Generate a unique ID for chat completions with timestamp and random component.
    """
    timestamp = int(time.time())
    random_suffix = random.randint(0, 999999)
    return f"chatcmpl-{timestamp}{random_suffix:06d}"

def get_tool_call_id():
    """
    Generate a unique ID for tool calls with timestamp and random component.
    """
    timestamp = int(time.time())
    random_suffix = random.randint(0, 999999)
    return f"call-{timestamp}{random_suffix:06d}"

def format_final_response(response: Union[str, List[Dict[str, Any]]], model: str) -> ChatCompletionResponse:
    """Format the final non-streaming response."""
    
    if isinstance(response, str):
        return ChatCompletionResponse(
            id=get_id(),
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[Choice(
                index=0,
                message=Message(role="assistant", content=response),
                finish_reason="stop"
            )]
        )
    content = response.get("content", None)
    tool_calls = response.get("tool_calls", [])
    tool_call_responses = []
    for tool_call in tool_calls:
        function_call = FunctionCall(
            name=tool_call.get("name"),
            arguments=json.dumps(tool_call.get("arguments"))
        )
        tool_call_response = ChatCompletionMessageToolCall(
            id=get_tool_call_id(),
            type="function",
            function=function_call
        )
        tool_call_responses.append(tool_call_response)
    
    return ChatCompletionResponse(
        id=get_id(),
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[Choice(
            index=0,
            message=Message(role="assistant", content=content, tool_calls=tool_call_responses),
            finish_reason="function_call"
        )]
    )