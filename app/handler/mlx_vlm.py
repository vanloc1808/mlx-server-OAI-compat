import asyncio
import base64
import hashlib
import logging
import os
import tempfile
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import HTTPException
from PIL import Image

from app.core.image_processor import ImageProcessor
from app.core.metrics import RequestMetrics
from app.core.queue import RequestQueue
from app.models.mlx_vlm import MLX_VLM
from app.schemas.openai import ChatCompletionRequest

# Configure logging
logger = logging.getLogger(__name__)

class MLXVLMHandler:
    """
    Handler class for making requests to the underlying MLX vision-language model service.
    Provides caching, concurrent image processing, and robust error handling.
    """

    def __init__(self, model_path: str, max_workers: int = 4, max_concurrency: int = 1):
        """
        Initialize the handler with the specified model path.
        
        Args:
            model_path (str): Path to the model directory.
            max_workers (int): Maximum number of worker threads for image processing.
            max_concurrency (int): Maximum number of concurrent model inference tasks.
        """
        self.model_path = model_path
        self.model = MLX_VLM(model_path)
        self.temp_dir = tempfile.mkdtemp(prefix="mlx_vlm_")
        self.image_processor = ImageProcessor(self.temp_dir, max_workers)
        
        # Initialize request queue for vision and text tasks
        # We use the same queue for both vision and text tasks for simplicity
        # and to ensure we don't overload the model with too many concurrent requests
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

    async def generate_vision_stream(self, request: ChatCompletionRequest):
        """
        Generate a streaming response for vision-based chat completion requests.
        
        Args:
            request: ChatCompletionRequest object containing the messages.
        
        Returns:
            AsyncGenerator: Yields response chunks.
        """
        
        # Create a unique request ID
        request_id = f"vision-{uuid.uuid4()}"
        
        # Submit the vision request directly (not through queue for streaming)
        try:
            # Start timing
            start_time = time.time()
            total_tokens = 0
            total_words = 0
            total_chars = 0

            chat_messages, image_paths, model_params = await self._prepare_vision_request(request)
            
            # Create a request data object
            request_data = {
                "images": image_paths,
                "messages": chat_messages,
                "stream": True,
                **model_params
            }
            
            # Submit to the vision queue and get the generator
            response_generator = await self.request_queue.submit(request_id, request_data)
            
            # Process and yield each chunk asynchronously
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
            self.metrics.update("vision_stream", metrics)
        
        except asyncio.QueueFull:
            self.metrics.increment_error_count()
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Service is at capacity."
            )

        except Exception as e:
            self.metrics.increment_error_count()
            logger.error(f"Error in vision stream generation for request {request_id}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate vision stream: {str(e)}"
            )

    async def generate_vision_response(self, request: ChatCompletionRequest):
        """
        Generate a complete response for vision-based chat completion requests.
        Uses the request queue for handling concurrent requests.
        
        Args:
            request: ChatCompletionRequest object containing the messages.
        
        Returns:
            str: Complete response.
        """
        try:
            # Create a unique request ID
            request_id = f"vision-{uuid.uuid4()}"
            
            # Prepare the vision request
            chat_messages, image_paths, model_params = await self._prepare_vision_request(request)
            
            # Create a request data object
            request_data = {
                "images": image_paths,
                "messages": chat_messages,
                "stream": False,
                **model_params
            }
            
            # Start timing
            start_time = time.time()
            
            # Submit to the vision queue and wait for result
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
            self.metrics.update("vision", metrics)
            
            return response
            
        except asyncio.QueueFull:
            self.metrics.increment_error_count()
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Service is at capacity."
            )
        except Exception as e:
            self.metrics.increment_error_count()
            logger.error(f"Error in vision response generation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate vision response: {str(e)}"
            )

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
            
            # Submit to the vision queue (reusing the same queue for text requests)
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


    def __del__(self):
        """Cleanup resources on deletion."""

        if hasattr(self, 'image_processor'):
            self.image_processor.cleanup()

    async def _process_request(self, request_data: Dict[str, Any]) -> str:
        """
        Process a vision request. This is the worker function for the request queue.
        
        Args:
            request_data: Dictionary containing the request data.
            
        Returns:
            str: The model's response.
        """
        try:
            # Extract request parameters
            images = request_data.get("images", [])
            messages = request_data.get("messages", [])
            stream = request_data.get("stream", False)
            
            # Remove these keys from model_params
            model_params = request_data.copy()
            model_params.pop("images", None)
            model_params.pop("messages", None)
            model_params.pop("stream", None)
            
            # Start timing
            start_time = time.time()
            
            # Call the model
            response = self.model(
                images=images,
                messages=messages,
                stream=stream,
                **model_params
            )
            
            # End timing and calculate metrics
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Calculate tokens in the response
            # For simple text responses, approximating token count as words/1.3
            if isinstance(response, str):
                metrics = RequestMetrics.estimate_tokens(response)
                token_count = metrics["estimated_tokens"]
                tps = token_count / elapsed_time if elapsed_time > 0 else 0
                logger.info(f"Request completed: {token_count} tokens in {elapsed_time:.2f}s ({tps:.2f} tokens/sec)")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing vision request: {str(e)}")
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

    async def _prepare_vision_request(self, request: ChatCompletionRequest) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
        """
        Prepare the vision request by processing messages and images.
        
        This method:
        1. Extracts text messages and image URLs from the request
        2. Processes image URLs to get local file paths
        3. Prepares model parameters
        4. Returns processed data ready for model inference
        
        Args:
            request (ChatCompletionRequest): The incoming request containing messages and parameters.
            
        Returns:
            Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]: A tuple containing:
                - List of processed chat messages
                - List of processed image paths
                - Dictionary of model parameters
                
        Raises:
            HTTPException: If message content is invalid or image processing fails.
        """
        chat_messages = []
        image_urls = []

        try:
            # Process each message in the request
            for i, message in enumerate(request.messages):
                is_last_message = i == len(request.messages) - 1
                
                # Handle system and assistant messages (simple text content)
                if message.role in ["system", "assistant"]:
                    chat_messages.append({"role": message.role, "content": message.content})
                    continue

                # Handle user messages
                if message.role == "user":
                    # Case 1: Simple string content
                    if isinstance(message.content, str):
                        chat_messages.append({"role": "user", "content": message.content})
                        continue
                        
                    # Case 2: Content is a list of dictionaries or objects
                    if isinstance(message.content, list):
                        # Initialize containers for this message
                        texts = []
                        images = []
                        formatted_content = []
                        
                        # Process each content item in the list
                        for item in message.content:
                            # Case 2.1: Item is a dictionary (already formatted)
                            if isinstance(item, dict):
                                item_type = item.get("type")
                                
                                if item_type == "text":
                                    text = item.get("text", "").strip()
                                    if text:
                                        texts.append(text)
                                        formatted_content.append({"type": "text", "text": text})
                                        
                                elif item_type == "image_url":
                                    url = item.get("image_url", {}).get("url", "")
                                    if url:
                                        # Validate URL
                                        self._validate_image_url(url)
                                        images.append(url)
                                        formatted_content.append({"type": "image_url", "image_url": {"url": url}})
                                        
                            # Case 2.2: Item is an object with attributes
                            elif hasattr(item, "type"):
                                if item.type == "text":
                                    text = getattr(item, "text", "").strip()
                                    if text:
                                        texts.append(text)
                                        
                                elif item.type == "image_url":
                                    url = getattr(item, "image_url", None)
                                    if url and hasattr(url, "url"):
                                        url = url.url
                                        # Validate URL
                                        self._validate_image_url(url)
                                        images.append(url)
                                            
                                else:
                                    raise HTTPException(status_code=400, detail=f"Unsupported content type: {item.type}")
                        
                        # Add collected images to global list
                        if images:
                            image_urls.extend(images)
                            
                            # Validate constraints
                            if len(images) > 4:
                                raise HTTPException(status_code=400, detail="Too many images in a single message (max: 4)")
                        
                        # Determine how to format the final message
                        if formatted_content:
                            # Already properly formatted content
                            chat_messages.append({"role": "user", "content": formatted_content})
                        elif texts and images:
                            # Need to format text + images manually
                            prompt = " ".join(texts)
                            if not prompt:
                                raise HTTPException(status_code=400, detail="No text content provided")
                                
                            # Add text content
                            content = [{"type": "text", "text": prompt}]
                            
                            # Only add image tokens for non-final messages
                            if not is_last_message:
                                content.extend([{"type": "image"}] * len(images))
                                
                            chat_messages.append({"role": "user", "content": content})
                        elif texts:
                            # Text-only message
                            chat_messages.append({"role": "user", "content": " ".join(texts)})
                        else:
                            raise HTTPException(status_code=400, detail="Message contains no valid content")
                    else:
                        raise HTTPException(status_code=400, detail="Invalid message content format")

            # Process images and prepare model parameters
            image_paths = await self.image_processor.process_image_urls(image_urls)
            
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
            logger.debug(f"Processed chat messages: {chat_messages}")
            logger.debug(f"Processed image paths: {image_paths}")
            logger.debug(f"Model parameters: {model_params}")

            return chat_messages, image_paths, model_params

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to prepare vision request: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to process request: {str(e)}")
            
    def _validate_image_url(self, url: str) -> None:
        """
        Validate image URL format.
        
        Args:
            url: The image URL to validate
            
        Raises:
            HTTPException: If URL is invalid
        """
        if not url:
            raise HTTPException(status_code=400, detail="Empty image URL provided")
            
        # Validate base64 images
        if url.startswith("data:"):
            try:
                header, encoded = url.split(",", 1)
                if not header.startswith("data:image/"):
                    raise ValueError("Invalid image format")
                base64.b64decode(encoded)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")