import asyncio
import base64
import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from typing import List, Optional
from urllib3 import PoolManager
from PIL import Image
import aiohttp
import time

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, temp_dir: str, max_workers: int = 4, cache_size: int = 1000):
        self.temp_dir = temp_dir
        self._http_pool = PoolManager(maxsize=10, retries=3)
        self._session: Optional[aiohttp.ClientSession] = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache_size = cache_size
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # 1 hour
        
        # Ensure temp directory exists
        os.makedirs(temp_dir, exist_ok=True)
        
        # Configure PIL for better performance
        Image.MAX_IMAGE_PIXELS = None  # Disable maximum image size check
        Image.warnings.simplefilter('ignore', Image.DecompressionBombWarning)

    @lru_cache(maxsize=1000)
    def _get_image_hash(self, image_url: str) -> str:
        if image_url.startswith("data:"):
            _, encoded = image_url.split(",", 1)
            data = base64.b64decode(encoded)
        else:
            data = image_url.encode('utf-8')
        return hashlib.md5(data).hexdigest()
    
    def _resize_image_keep_aspect_ratio(self, image: Image.Image, max_size: int = 768) -> Image.Image:
        width, height = image.size
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    "User-Agent": "mlx-server-OAI-compat/1.0 (https://github.com/cubist38/mlx-server-OAI-compat; cubist38@gmail.com) Python-Requests"
                }
            )
        return self._session

    def _cleanup_old_files(self):
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            try:
                for file in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, file)
                    if os.path.getmtime(file_path) < current_time - self._cleanup_interval:
                        os.remove(file_path)
                self._last_cleanup = current_time
            except Exception as e:
                logger.warning(f"Failed to clean up old files: {str(e)}")

    async def _process_single_image(self, image_url: str) -> str:
        try:

            image_hash = self._get_image_hash(image_url)
            cached_path = os.path.join(self.temp_dir, f"{image_hash}.jpg")

            if os.path.exists(cached_path):
                logger.debug(f"Using cached image: {cached_path}")
                return cached_path
            
            # check if image_url is a local file
            if os.path.exists(image_url):
                image = Image.open(image_url)
                image = self._resize_image_keep_aspect_ratio(image)
                image.save(cached_path, 'JPEG', quality=100, optimize=True)
                return cached_path

            elif image_url.startswith("data:"):
                try:
                    header, encoded = image_url.split(",", 1)
                    data = base64.b64decode(encoded)
                    
                    try:
                        image = Image.open(BytesIO(data))
                    except Exception as img_error:
                        logger.error(f"Failed to open image from base64 data: {str(img_error)}")
                        if "image/jpeg" in header:
                            from PIL import ImageFile
                            ImageFile.LOAD_TRUNCATED_IMAGES = True
                            image = Image.open(BytesIO(data))
                        elif "image/png" in header:
                            image = Image.open(BytesIO(data))
                        else:
                            raise ValueError(f"Unsupported image format: {header}")
                except Exception as base64_error:
                    raise ValueError(f"Invalid base64 image data: {str(base64_error)}")
            else:
                session = await self._get_session()
                async with session.get(image_url) as response:
                    response.raise_for_status()
                    data = await response.read()
                    image = Image.open(BytesIO(data))

            image = self._resize_image_keep_aspect_ratio(image)
            image.save(cached_path, 'JPEG', quality=100, optimize=True)
            
            # Cleanup old files periodically
            self._cleanup_old_files()
            
            return cached_path

        except Exception as e:
            logger.error(f"Failed to process image: {str(e)}")
            raise ValueError(f"Failed to process image: {str(e)}")

    async def process_image_url(self, image_url: str) -> str:
        return await self._process_single_image(image_url)

    async def process_image_urls(self, image_urls: List[str]) -> List[str]:
        tasks = [self.process_image_url(url) for url in image_urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def cleanup(self):
        """Cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
        self.executor.shutdown(wait=True)
        self._http_pool.clear()
        
        try:
            # Clean up temp directory
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {str(e)}")
