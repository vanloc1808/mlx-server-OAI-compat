import os
import requests
import base64
import tempfile
import hashlib
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from PIL import Image
from io import BytesIO
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from app.handler.schema import ChatCompletionRequest
from app.models.mlx_vlm import MLX_VLM
from fastapi import HTTPException
from app.handler.queue import RequestQueue
import uuid

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
        
        # Initialize request queue for vision tasks
        self.vision_queue = RequestQueue(max_concurrency=max_concurrency)
        
        logger.info(f"Initialized MLXHandler with model path: {model_path}")
        
    async def initialize_queues(self):
        """Initialize and start the request queue."""
        await self.vision_queue.start(self._process_vision_request)
        logger.info("Started request queue for vision processing")

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
                response = requests.get(image_url, timeout=10)
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
                    if hasattr(chunk, 'text'):
                        yield chunk.text
                    elif isinstance(chunk, str):
                        yield chunk
                    else:
                        yield str(chunk)
        except Exception as e:
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
            
            # Submit to the vision queue and wait for result
            return await self.vision_queue.submit(request_id, request_data)
            
        except asyncio.QueueFull:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Service is at capacity."
            )
        except Exception as e:
            logger.error(f"Error in vision response generation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate vision response: {str(e)}"
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
            
            # Call the model
            response = self.model(
                images=images,
                messages=messages,
                stream=stream,
                **model_params
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing vision request: {str(e)}")
            raise
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the vision queue.
        
        Returns:
            Dict with queue statistics.
        """
        return {
            "vision_queue": self.vision_queue.get_queue_stats()
        }

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

        try:
            # Process each message in the request
            for message in request.messages:
                # Handle system messages
                if message.role == "system":
                    chat_messages.append({
                        "role": "system",
                        "content": message.content
                    })
                    continue

                # Handle assistant messages
                if message.role == "assistant":
                    chat_messages.append({
                        "role": "assistant",
                        "content": message.content
                    })
                    continue

                # Handle user messages
                if message.role == "user":
                    if isinstance(message.content, str):
                        # Simple text message
                        chat_messages.append({
                            "role": "user",
                            "content": message.content
                        })
                        continue
                        
                    elif isinstance(message.content, list):
                        # Message containing both text and images
                        texts = []
                        images = []
                        
                        for item in message.content:
                            if item.type == "text":
                                if not item.text or not item.text.strip():
                                    continue
                                texts.append(item.text.strip())
                            elif item.type == "image_url":
                                url = item.image_url.url
                                if not url:
                                    raise HTTPException(
                                        status_code=400,
                                        detail="Empty image URL provided"
                                    )
                                if url.startswith("data:"):
                                    # Validate base64 image format
                                    try:
                                        header, encoded = url.split(",", 1)
                                        if not header.startswith("data:image/"):
                                            raise ValueError("Invalid image format")
                                        base64.b64decode(encoded)
                                    except Exception as e:
                                        raise HTTPException(
                                            status_code=400,
                                            detail=f"Invalid base64 image: {str(e)}"
                                        )
                                images.append(url)
                            else:
                                raise HTTPException(
                                    status_code=400,
                                    detail=f"Unsupported content type: {item.type}"
                                )
                        
                        # Add text content if present
                        if texts:
                            chat_messages.append({
                                "role": "user",
                                "content": " ".join(texts)
                            })
                        
                        # Collect image URLs
                        if images:
                            # Check image count limit (OpenAI typically allows 1-4 images)
                            if len(images) > 4:
                                raise HTTPException(
                                    status_code=400,
                                    detail="Too many images in a single message"
                                )
                            image_urls.extend(images)
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid message content format"
                        )

            # Process image URLs to get local file paths
            image_paths = await self.process_image_urls(image_urls)

            # Prepare model parameters, excluding None values
            model_params = {
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stop": request.stop,
                "n": request.n,
                "seed": request.seed
            }
            model_params = {k: v for k, v in model_params.items() if v is not None}

            # Handle response format if specified
            if request.response_format:
                if request.response_format.get("type") == "json_object":
                    # Add JSON mode parameters if needed
                    model_params["response_format"] = "json"

            # Handle tool calls if specified
            if request.tools:
                model_params["tools"] = request.tools
                
                # Handle tool choice
                if request.tool_choice:
                    model_params["tool_choice"] = request.tool_choice

            # Log processed data for debugging
            logger.debug(f"Processed chat messages: {chat_messages}")
            logger.debug(f"Processed image paths: {image_paths}")
            logger.debug(f"Model parameters: {model_params}")

            return chat_messages, image_paths, model_params

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to prepare vision request: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process request: {str(e)}"
            )