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
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from fastapi import HTTPException
from PIL import Image

from app.core.queue import RequestQueue
from app.models.mlx_vlm import MLX_VLM
from app.schemas.openai import ChatCompletionRequest

# Configure logging
logger = logging.getLogger(__name__)

class MLXHandler:
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
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.temp_dir = tempfile.mkdtemp(prefix="mlx_vlm_")
        
        # Initialize request queue for vision and text tasks
        # We use the same queue for both vision and text tasks for simplicity
        # and to ensure we don't overload the model with too many concurrent requests
        self.vision_queue = RequestQueue(max_concurrency=max_concurrency)
        
        # Initialize metrics tracking
        self.metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time": 0,
            "request_types": defaultdict(int),
            "error_count": 0,
            "avg_tps": 0,
            "max_tps": 0,
            "min_tps": 0,
            "request_history": []
        }
        
        logger.info(f"Initialized MLXHandler with model path: {model_path}")
    
    async def initialize(self, queue_config: Optional[Dict[str, Any]] = None):
        """Initialize the handler and start the request queue."""
        if not queue_config:
            queue_config = {
                "max_concurrency": 1,
                "timeout": 300,
                "queue_size": 100
            }
        self.vision_queue = RequestQueue(
            max_concurrency=queue_config.get("max_concurrency"),
            timeout=queue_config.get("timeout"),
            queue_size=queue_config.get("queue_size")
        )
        await self.vision_queue.start(self._process_vision_request)
        logger.info("Initialized MLXHandler and started request queue")


    def __del__(self):
        """Cleanup resources on deletion."""
        self.executor.shutdown(wait=True)
        try:
            # Clean up temp directory
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {str(e)}")

    @lru_cache(maxsize=100)
    def _get_image_hash(self, image_url: str) -> str:
        """
        Generate a hash for an image URL or base64 string.
        
        Args:
            image_url (str): Image URL or base64-encoded string.
        
        Returns:
            str: Hash of the image content.
        """
        if image_url.startswith("data:"):
            _, encoded = image_url.split(",", 1)
            data = base64.b64decode(encoded)
        else:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            data = response.content
        return hashlib.md5(data).hexdigest()

    def _process_single_image(self, image_url: str) -> str:
        """
        Process a single image URL or base64 string.
        
        Args:
            image_url (str): Image URL or base64-encoded string.
        
        Returns:
            str: Path to the saved image file.
        
        Raises:
            ValueError: If image processing fails.
        """
        try:
            image_hash = self._get_image_hash(image_url)
            cached_path = os.path.join(self.temp_dir, f"{image_hash}.jpg")
            
            if os.path.exists(cached_path):
                logger.debug(f"Using cached image: {cached_path}")
                return cached_path

            if image_url.startswith("data:"):
                header, encoded = image_url.split(",", 1)
                data = base64.b64decode(encoded)
                image = Image.open(BytesIO(data))
            else:
                headers = {
                    "User-Agent": "proxy-OAI-compat/1.0 (https://github.com/vuonggiahuy/proxy-OAI-compat; your-email@example.com) Python-Requests"
                }
                response = requests.get(image_url, headers=headers, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))

            # Optimize image before saving
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')

            # Save with optimization
            image.save(cached_path, 'JPEG', quality=85, optimize=True)
            logger.debug(f"Saved processed image: {cached_path}")
            return cached_path

        except Exception as e:
            logger.error(f"Failed to process image: {str(e)}")
            raise ValueError(f"Failed to process image: {str(e)}")

    async def process_image_url(self, image_url: str) -> str:
        """
        Process an image URL or base64 string asynchronously.
        
        Args:
            image_url (str): Image URL or base64-encoded string.
        
        Returns:
            str: Path to the saved image file.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._process_single_image, image_url)

    async def process_image_urls(self, image_urls: List[str]) -> List[str]:
        """
        Process multiple image URLs concurrently.
        
        Args:
            image_urls (List[str]): List of image URLs or base64 strings.
        
        Returns:
            List[str]: List of paths to saved image files.
        """
        tasks = [self.process_image_url(url) for url in image_urls]
        return await asyncio.gather(*tasks)

    async def generate_vision_stream(self, request: ChatCompletionRequest):
        """
        Generate a streaming response for vision-based chat completion requests.
        
        Args:
            request: ChatCompletionRequest object containing the messages.
        
        Returns:
            AsyncGenerator: Yields response chunks.
        """
        chat_messages, image_paths, model_params = await self._prepare_vision_request(request)
        
        # Create a unique request ID
        request_id = f"vision-{uuid.uuid4()}"
        
        # Submit the vision request directly (not through queue for streaming)
        try:
            # Start timing
            start_time = time.time()
            total_tokens = 0
            total_words = 0
            total_chars = 0
            
            # Get the generator from the model
            response_generator = self.model(
                images=image_paths,
                messages=chat_messages,
                stream=True,
                **model_params
            )
            
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
                        chunk_metrics = self._estimate_tokens(text_chunk)
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
            self._update_metrics("vision_stream", metrics)
            
        except Exception as e:
            self.metrics["error_count"] += 1
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
            response = await self.vision_queue.submit(request_id, request_data)
            
            # Calculate and log TPS statistics
            elapsed_time = time.time() - start_time
            metrics = self._estimate_tokens(response)
            tps = metrics["estimated_tokens"] / elapsed_time if elapsed_time > 0 else 0
            
            # Update metrics
            metrics.update({
                "elapsed_time": elapsed_time,
                "tps": tps,
                "token_count": metrics["estimated_tokens"]
            })
            self._update_metrics("vision", metrics)
            
            return response
            
        except asyncio.QueueFull:
            self.metrics["error_count"] += 1
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Service is at capacity."
            )
        except Exception as e:
            self.metrics["error_count"] += 1
            logger.error(f"Error in vision response generation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate vision response: {str(e)}"
            )

    async def generate_text_stream(self, request: ChatCompletionRequest):
        """
        Generate a streaming response for text-only chat completion requests.
        
        Args:
            request: ChatCompletionRequest object containing the messages.
        
        Returns:
            AsyncGenerator: Yields response chunks.
        """
        chat_messages, model_params = await self._prepare_text_request(request)
        
        # Create a unique request ID
        request_id = f"text-{uuid.uuid4()}"
        
        try:
            # Start timing
            start_time = time.time()
            total_tokens = 0
            total_words = 0
            total_chars = 0
            
            # Get the generator from the model
            response_generator = self.model(
                messages=chat_messages,
                stream=True,
                **model_params
            )
            
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
                        chunk_metrics = self._estimate_tokens(text_chunk)
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
            self._update_metrics("text_stream", metrics)
            
        except Exception as e:
            self.metrics["error_count"] += 1
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
            response = await self.vision_queue.submit(request_id, request_data)
            
            # Calculate and log TPS statistics
            elapsed_time = time.time() - start_time
            metrics = self._estimate_tokens(response)
            tps = metrics["estimated_tokens"] / elapsed_time if elapsed_time > 0 else 0
            
            # Update metrics
            metrics.update({
                "elapsed_time": elapsed_time,
                "tps": tps,
                "token_count": metrics["estimated_tokens"]
            })
            self._update_metrics("text", metrics)
            
            return response
            
        except asyncio.QueueFull:
            self.metrics["error_count"] += 1
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Service is at capacity."
            )
        except Exception as e:
            self.metrics["error_count"] += 1
            logger.error(f"Error in text response generation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate text response: {str(e)}"
            )

    async def _process_vision_request(self, request_data: Dict[str, Any]) -> str:
        """
        Process a vision request. This is the worker function for the vision queue.
        
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
                metrics = self._estimate_tokens(response)
                token_count = metrics["estimated_tokens"]
                tps = token_count / elapsed_time if elapsed_time > 0 else 0
                logger.info(f"Request completed: {token_count} tokens in {elapsed_time:.2f}s ({tps:.2f} tokens/sec)")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing vision request: {str(e)}")
            raise
    
    def _estimate_tokens(self, text: str) -> Dict[str, int]:
        """
        Estimate the number of tokens in a text with more detailed metrics.
        
        Args:
            text (str): The text to estimate tokens for.
            
        Returns:
            Dict[str, int]: Dictionary containing token estimates and metrics.
        """
        # Split into words and count
        words = text.split()
        word_count = len(words)
        
        # Count characters
        char_count = len(text)
        
        # Estimate tokens using different methods
        # Method 1: words/1.3 (common approximation)
        tokens_by_words = int(word_count / 1.3)
        
        # Method 2: chars/4 (another common approximation)
        tokens_by_chars = int(char_count / 4)
        
        # Use the average of both methods for better accuracy
        estimated_tokens = (tokens_by_words + tokens_by_chars) // 2
        
        return {
            "estimated_tokens": estimated_tokens,
            "word_count": word_count,
            "char_count": char_count,
            "tokens_by_words": tokens_by_words,
            "tokens_by_chars": tokens_by_chars
        }

    def _update_metrics(self, request_type: str, metrics: Dict[str, Any]):
        """
        Update the global metrics with new request statistics.
        
        Args:
            request_type (str): Type of request (vision/text, streaming/non-streaming)
            metrics (Dict[str, Any]): Request-specific metrics
        """
        self.metrics["total_requests"] += 1
        self.metrics["request_types"][request_type] += 1
        
        # Update token and time metrics - use get() with defaults for safety
        self.metrics["total_tokens"] += metrics.get("token_count", metrics.get("estimated_tokens", 0))
        self.metrics["total_time"] += metrics.get("elapsed_time", 0)
        
        # Update TPS metrics
        current_tps = metrics.get("tps", 0)
        self.metrics["avg_tps"] = (self.metrics["avg_tps"] * (self.metrics["total_requests"] - 1) + current_tps) / self.metrics["total_requests"]
        self.metrics["max_tps"] = max(self.metrics["max_tps"], current_tps)
        if self.metrics["total_requests"] == 1 or current_tps < self.metrics["min_tps"]:
            self.metrics["min_tps"] = current_tps
        
        # Add to request history (keep last 100 requests)
        self.metrics["request_history"].append({
            "timestamp": time.time(),
            "request_type": request_type,
            **metrics
        })
        if len(self.metrics["request_history"]) > 100:
            self.metrics["request_history"].pop(0)
        
        # Log detailed metrics - use get() with defaults for safety
        token_count = metrics.get("token_count", metrics.get("estimated_tokens", 0))
        word_count = metrics.get("word_count", 0)
        char_count = metrics.get("char_count", 0)
        elapsed_time = metrics.get("elapsed_time", 0)
        
        logger.info(
            f"Request completed: {request_type}\n"
            f"Tokens: {token_count} (words: {word_count}, chars: {char_count})\n"
            f"Time: {elapsed_time:.2f}s\n"
            f"TPS: {current_tps:.2f}\n"
            f"Avg TPS: {self.metrics['avg_tps']:.2f}"
        )

    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the request queue and performance metrics.
        
        Returns:
            Dict with queue and performance statistics.
        """
        queue_stats = self.vision_queue.get_queue_stats()
        
        # Calculate additional metrics
        metrics_summary = {
            "total_requests": self.metrics["total_requests"],
            "total_tokens": self.metrics["total_tokens"],
            "total_time": self.metrics["total_time"],
            "request_types": dict(self.metrics["request_types"]),
            "error_count": self.metrics["error_count"],
            "performance": {
                "avg_tps": self.metrics["avg_tps"],
                "max_tps": self.metrics["max_tps"],
                "min_tps": self.metrics["min_tps"],
                "recent_requests": len(self.metrics["request_history"])
            }
        }
        
        return {
            "queue_stats": queue_stats,
            "metrics": metrics_summary
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

    async def _prepare_vision_request(self, request: ChatCompletionRequest) -> Tuple[List[Dict[str, str]], List[str], Dict[str, Any]]:
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
            Tuple[List[Dict[str, str]], List[str], Dict[str, Any]]: A tuple containing:
                - List of processed chat messages
                - List of processed image paths
                - Dictionary of model parameters
                
        Raises:
            HTTPException: If message content is invalid or image processing fails.
        """
        chat_messages = []
        image_urls = []

        def format_message(role, content):
            """Simple helper to format message with consistent structure"""
            return {"role": role, "content": content}

        def handle_list_with_image(prompt, role, num_images, skip_image_token=False):
            """Format message with proper image token handling"""
            content = [{"type": "text", "text": prompt}]
            if role == "user" and not skip_image_token:
                content.extend([{"type": "image"}] * num_images)
            return {"role": role, "content": content}

        try:
            for i, message in enumerate(request.messages):
                is_last_message = i == len(request.messages) - 1
                
                # Handle system and assistant messages (simple text content)
                if message.role in ["system", "assistant"]:
                    chat_messages.append(format_message(message.role, message.content))
                    continue

                # Handle user messages
                if message.role == "user":
                    # Simple string content
                    if isinstance(message.content, str):
                        chat_messages.append(format_message("user", message.content))
                        continue
                        
                    # Complex content (text + images)
                    elif isinstance(message.content, list):
                        texts = []
                        images = []
                        
                        # Process each content item
                        for item in message.content:
                            if item.type == "text":
                                text = item.text.strip() if item.text else ""
                                if text:
                                    texts.append(text)
                            elif item.type == "image_url":
                                url = item.image_url.url
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
                                
                                images.append(url)
                            else:
                                raise HTTPException(status_code=400, detail=f"Unsupported content type: {item.type}")
                        
                        # Validate constraints
                        if len(images) > 4:
                            raise HTTPException(status_code=400, detail="Too many images in a single message (max: 4)")
                            
                        if not texts:
                            raise HTTPException(status_code=400, detail="No text content provided")
                        
                        # Add collected images to global list
                        if images:
                            image_urls.extend(images)
                        
                        # Join text segments
                        prompt = " ".join(texts)
                        
                        # Format appropriately based on position in message sequence
                        skip_image_token = is_last_message
                        formatted_message = handle_list_with_image(
                            prompt=prompt,
                            role="user",
                            num_images=len(images),
                            skip_image_token=skip_image_token
                        )
                        chat_messages.append(formatted_message)
                    else:
                        raise HTTPException(status_code=400, detail="Invalid message content format")

            # Process images and prepare model parameters
            image_paths = await self.process_image_urls(image_urls)

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