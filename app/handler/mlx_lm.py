import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator
from http import HTTPStatus

from fastapi import HTTPException
from app.core.queue import RequestQueue
from app.models.mlx_lm import MLX_LM
from app.handler.parser import get_parser
from app.schemas.openai import ChatCompletionRequest, EmbeddingRequest
from app.utils.errors import create_error_response

# Configure logging
logger = logging.getLogger(__name__)

class MLXLMHandler:
    """
    Handler class for making requests to the underlying MLX text-only language model service.
    Provides request queuing, metrics tracking, and robust error handling.
    """

    def __init__(self, model_path: str, max_concurrency: int = 1):
        """
        Initialize the handler with the specified model path.
        
        Args:
            model_path (str): Path to the model directory.
            max_concurrency (int): Maximum number of concurrent model inference tasks.
        """
        self.model_path = model_path
        self.model = MLX_LM(model_path)
        self.model_type = self.model.get_model_type()
        self.tool_parser, self.thinking_parser = get_parser(self.model_type)
        
        # Initialize request queue for text tasks
        self.request_queue = RequestQueue(max_concurrency=max_concurrency)
        
        # Initialize metrics tracking
        # self.metrics = RequestMetrics()
        
        logger.info(f"Initialized MLXHandler with model path: {model_path}")
    
    async def initialize(self, queue_config: Optional[Dict[str, Any]] = None):
        """Initialize the handler and start the request queue."""
        if not queue_config:
            queue_config = {
                "max_concurrency": 1,
                "timeout": 300,
                "queue_size": 100
            }
        self.request_queue = RequestQueue(
            max_concurrency=queue_config.get("max_concurrency"),
            timeout=queue_config.get("timeout"),
            queue_size=queue_config.get("queue_size")
        )
        await self.request_queue.start(self._process_request)
        logger.info("Initialized MLXHandler and started request queue")

    async def generate_text_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response for text-only chat completion requests.
        Uses the request queue for handling concurrent requests.
        
        Args:
            request: ChatCompletionRequest object containing the messages.
        
        Yields:
            str: Response chunks.
        """
        request_id = f"text-{uuid.uuid4()}"
        
        try:
            chat_messages, model_params = await self._prepare_text_request(request)
            request_data = {
                "messages": chat_messages,
                "stream": True,
                **model_params
            }
            response_generator = await self.request_queue.submit(request_id, request_data)
            
            buffer = ""
            if model_params.get("tools", None) and self.tool_parser:
                for chunk in response_generator:
                    if chunk:
                        chunk = chunk.text
                        buffer += chunk
                        chunk, buffer = self.tool_parser.parse_stream(chunk, buffer)
                        yield chunk
            else:
                for chunk in response_generator:
                    if chunk:
                        yield chunk.text
            
        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response("Too many requests. Service is at capacity.", "rate_limit_exceeded", HTTPStatus.TOO_MANY_REQUESTS)
            raise HTTPException(status_code=429, detail=content)
        except Exception as e:
            logger.error(f"Error in text stream generation for request {request_id}: {str(e)}")
            content = create_error_response(f"Failed to generate text stream: {str(e)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            raise HTTPException(status_code=500, detail=content)

    async def generate_text_response(self, request: ChatCompletionRequest) -> str:
        """
        Generate a complete response for text-only chat completion requests.
        Uses the request queue for handling concurrent requests.
        
        Args:
            request: ChatCompletionRequest object containing the messages.
        
        Returns:
            str: Complete response.
        """
        request_id = f"text-{uuid.uuid4()}"
        
        try:
            chat_messages, model_params = await self._prepare_text_request(request)
            request_data = {
                "messages": chat_messages,
                "stream": False,
                **model_params
            }
            response = await self.request_queue.submit(request_id, request_data)

            if model_params.get("tools", None) and self.tool_parser:
                thinking_content = self.thinking_parser.parse(response)
                if thinking_content:
                    response = response[len(thinking_content):]
                parsed_response = self.tool_parser.parse(response)
                if isinstance(parsed_response, list):
                    return {
                        "content": thinking_content,
                        "tool_calls": parsed_response
                    }
                if thinking_content:
                    response = thinking_content + response
            return response
            
        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response("Too many requests. Service is at capacity.", "rate_limit_exceeded", HTTPStatus.TOO_MANY_REQUESTS)
            raise HTTPException(status_code=429, detail=content)
        except Exception as e:
            logger.error(f"Error in text response generation: {str(e)}")
            content = create_error_response(f"Failed to generate text response: {str(e)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            raise HTTPException(status_code=500, detail=content)
        
    async def generate_embeddings_response(self, request: EmbeddingRequest):
        """
        Generate embeddings for a given text input.
        
        Args:
            request: EmbeddingRequest object containing the text input.
        
        Returns:
            List[float]: Embeddings for the input text.
        """
        try:
            # Create a unique request ID
            request_id = f"embeddings-{uuid.uuid4()}"
            request_data = {
                "type": "embeddings",
                "input": request.input,
                "model": request.model
            }

            # Submit to the request queue
            response = await self.request_queue.submit(request_id, request_data)

            return response

        except Exception as e:
            logger.error(f"Error in embeddings generation: {str(e)}")
            content = create_error_response(f"Failed to generate embeddings: {str(e)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            raise HTTPException(status_code=500, detail=content)
        

    async def _process_request(self, request_data: Dict[str, Any]) -> str:
        """
        Process a text request. This is the worker function for the request queue.
        
        Args:
            request_data: Dictionary containing the request data.
            
        Returns:
            str: The model's response.
        """
        try:
            # Check if the request is for embeddings
            if request_data.get("type") == "embeddings":
                return self.model.get_embeddings(request_data["input"])

            # Extract request parameters
            messages = request_data.get("messages", [])
            stream = request_data.get("stream", False)
            
            # Remove these keys from model_params
            model_params = request_data.copy()
            model_params.pop("messages", None)
            model_params.pop("stream", None)
            
            # Start timing
            start_time = time.time()
            
            # Call the model
            response = self.model(
                messages=messages,
                stream=stream,
                **model_params
            )
           
            return response
            
        except Exception as e:
            logger.error(f"Error processing text request: {str(e)}")
            raise

    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the request queue and performance metrics.
        
        Returns:
            Dict with queue and performance statistics.
        """
        queue_stats = self.request_queue.get_queue_stats()
        
        return {
            "queue_stats": queue_stats,
            # "metrics": self.metrics.get_summary()
        }
        
    async def cleanup(self):
        """
        Cleanup resources and stop the request queue before shutdown.
        
        This method ensures all pending requests are properly cancelled
        and resources are released.
        """
        try:
            logger.info("Cleaning up MLXLMHandler resources")
            if hasattr(self, 'request_queue'):
                await self.request_queue.stop()
            logger.info("MLXLMHandler cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during MLXLMHandler cleanup: {str(e)}")
            raise

    async def _prepare_text_request(self, request: ChatCompletionRequest) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Prepare a text request by parsing model parameters and verifying the format of messages.
        
        Args:
            request: ChatCompletionRequest object containing the messages.
        
        Returns:
            Tuple containing the formatted chat messages and model parameters.
        """
        try:
            # Get model parameters from the request
            temperature = request.temperature or 0.7
            top_p = request.top_p or 1.0
            frequency_penalty = request.frequency_penalty or 0.0
            presence_penalty = request.presence_penalty or 0.0
            max_tokens = request.max_tokens or 1024
            enable_thinking = request.enable_thinking or False
            tools = request.tools or None
            tool_choice = request.tool_choice or None

            model_params = {
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "max_tokens": max_tokens,
                "tools": tools,
                "enable_thinking": enable_thinking,
                "tool_choice": tool_choice
            }
            
            # Format chat messages
            chat_messages = []
            for message in request.messages:
                chat_messages.append({
                    "role": message.role,
                    "content": message.content
                })
            
            return chat_messages, model_params
        
        except Exception as e:
            logger.error(f"Failed to prepare text request: {str(e)}")
            content = create_error_response(f"Failed to process request: {str(e)}", "bad_request", HTTPStatus.BAD_REQUEST)
            raise HTTPException(status_code=400, detail=content)