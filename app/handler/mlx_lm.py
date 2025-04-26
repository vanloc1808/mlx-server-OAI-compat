import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from app.core.metrics import RequestMetrics
from app.core.queue import RequestQueue
from app.models.mlx_lm import MLX_LM
from app.schemas.openai import ChatCompletionRequest, EmbeddingRequest

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
        
        # Initialize request queue for text tasks
        self.request_queue = RequestQueue(max_concurrency=max_concurrency)
        
        # Initialize metrics tracking
        self.metrics = RequestMetrics()
        
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

    async def generate_text_stream(self, request: ChatCompletionRequest):
        """
        Generate a streaming response for text-only chat completion requests.
        Uses the request queue for handling concurrent requests.
        
        Args:
            request: ChatCompletionRequest object containing the messages.
        
        Returns:
            AsyncGenerator: Yields response chunks.
        """
        # Create a unique request ID
        request_id = f"text-{uuid.uuid4()}"
        
        try:
            # Start timing
            start_time = time.time()
            total_tokens = 0
            total_words = 0
            total_chars = 0
            
            # Prepare the text request
            chat_messages, model_params = await self._prepare_text_request(request)
            
            # Create a request data object
            request_data = {
                "messages": chat_messages,
                "stream": True,
                **model_params
            }
            
            # Submit to the request queue and get the generator
            response_generator = await self.request_queue.submit(request_id, request_data)
            
            # Process and yield each chunk
            for chunk in response_generator:
                if chunk:
                    text_chunk = ""
                    if hasattr(chunk, 'text'):
                        text_chunk = chunk.text
                    elif isinstance(chunk, str):
                        text_chunk = chunk
                    else:
                        text_chunk = str(chunk)
                    
                    # Update token count
                    if text_chunk:
                        chunk_metrics = RequestMetrics.estimate_tokens(text_chunk)
                        total_tokens += chunk_metrics["estimated_tokens"]
                        total_words += chunk_metrics["word_count"]
                        total_chars += chunk_metrics["char_count"]
                    
                    yield text_chunk
            
            # Calculate and log TPS statistics
            elapsed_time = time.time() - start_time
            tps = total_tokens / elapsed_time if elapsed_time > 0 else 0
            
            # Update metrics
            metrics = {
                "token_count": total_tokens,
                "word_count": total_words,
                "char_count": total_chars,
                "elapsed_time": elapsed_time,
                "tps": tps
            }
            self.metrics.update("text_stream", metrics)
            
        except asyncio.QueueFull:
            self.metrics.increment_error_count()
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Service is at capacity."
            )
        except Exception as e:
            self.metrics.increment_error_count()
            logger.error(f"Error in text stream generation for request {request_id}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate text stream: {str(e)}"
            )

    async def generate_text_response(self, request: ChatCompletionRequest):
        """
        Generate a complete response for text-only chat completion requests.
        Uses the request queue for handling concurrent requests.
        
        Args:
            request: ChatCompletionRequest object containing the messages.
        
        Returns:
            str: Complete response.
        """
        try:
            # Create a unique request ID
            request_id = f"text-{uuid.uuid4()}"
            
            # Prepare the text request
            chat_messages, model_params = await self._prepare_text_request(request)
            
            # Create a request data object
            request_data = {
                "messages": chat_messages,
                "stream": False,
                **model_params
            }
            
            # Start timing
            start_time = time.time()
            
            # Submit to the request queue
            response = await self.request_queue.submit(request_id, request_data)
            
            # Calculate and log TPS statistics
            elapsed_time = time.time() - start_time
            metrics = RequestMetrics.estimate_tokens(response)
            tps = metrics["estimated_tokens"] / elapsed_time if elapsed_time > 0 else 0
            
            # Update metrics
            metrics.update({
                "elapsed_time": elapsed_time,
                "tps": tps,
                "token_count": metrics["estimated_tokens"]
            })
            self.metrics.update("text", metrics)
            
            return response
            
        except asyncio.QueueFull:
            self.metrics.increment_error_count()
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Service is at capacity."
            )
        except Exception as e:
            self.metrics.increment_error_count()
            logger.error(f"Error in text response generation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate text response: {str(e)}"
            )
        
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
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embeddings: {str(e)}"
            )
        

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
            
            # End timing and calculate metrics
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Calculate tokens in the response
            if isinstance(response, str):
                metrics = RequestMetrics.estimate_tokens(response)
                token_count = metrics["estimated_tokens"]
                tps = token_count / elapsed_time if elapsed_time > 0 else 0
                logger.info(f"Request completed: {token_count} tokens in {elapsed_time:.2f}s ({tps:.2f} tokens/sec)")
            
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
            "metrics": self.metrics.get_summary()
        }

    async def _prepare_text_request(self, request: ChatCompletionRequest) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Prepare a text-only request by processing messages.
        
        Args:
            request (ChatCompletionRequest): The incoming request containing messages and parameters.
            
        Returns:
            Tuple[List[Dict[str, str]], Dict[str, Any]]: A tuple containing:
                - List of processed chat messages
                - Dictionary of model parameters
                
        Raises:
            HTTPException: If message content is invalid.
        """
        chat_messages = []

        try:
            
            # Convert Message objects to dictionaries with 'role' and 'content' keys
            chat_messages = []
            for message in request.messages:
                # Only handle simple string content for text-only requests
                if not isinstance(message.content, str):
                    logger.warning(f"Non-string content in text request will be skipped: {message.role}")
                    continue
                
                chat_messages.append({
                    "role": message.role,
                    "content": message.content
                })

            # Extract model parameters, filtering out None values
            model_params = {
                k: v for k, v in {
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "frequency_penalty": request.frequency_penalty, 
                    "presence_penalty": request.presence_penalty,
                    "stop": request.stop,
                    "n": request.n,
                    "seed": request.seed
                }.items() if v is not None
            }

            # Handle response format
            if request.response_format and request.response_format.get("type") == "json_object":
                model_params["response_format"] = "json"

            # Handle tools and tool choice
            if request.tools:
                model_params["tools"] = request.tools
                if request.tool_choice:
                    model_params["tool_choice"] = request.tool_choice

            # Log processed data
            logger.debug(f"Processed text chat messages: {chat_messages}")
            logger.debug(f"Model parameters: {model_params}")

            return chat_messages, model_params

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to prepare text request: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to process request: {str(e)}")