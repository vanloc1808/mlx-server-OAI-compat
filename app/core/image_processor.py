import asyncio
import base64
import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from typing import List

import requests
from PIL import Image

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, temp_dir: str, max_workers: int = 4):
        self.temp_dir = temp_dir
        self._connection = requests.Session()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    @lru_cache(maxsize=100)
    def _get_image_hash(self, image_url: str) -> str:
        if image_url.startswith("data:"):
            _, encoded = image_url.split(",", 1)
            data = base64.b64decode(encoded)
        else:
            # since image_url is a unique identifier, we can use it as the hash
            data = image_url.encode('utf-8')
        return hashlib.md5(data).hexdigest()

    def _process_single_image(self, image_url: str) -> str:
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
                response = self._connection.get(image_url, headers=headers, timeout=10)
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
            image.save(cached_path, 'JPEG', quality=100, optimize=True)
            logger.debug(f"Saved processed image: {cached_path}")
            return cached_path

        except Exception as e:
            logger.error(f"Failed to process image: {str(e)}")
            raise ValueError(f"Failed to process image: {str(e)}")

    async def process_image_url(self, image_url: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._process_single_image, image_url)

    async def process_image_urls(self, image_urls: List[str]) -> List[str]:
        tasks = [self.process_image_url(url) for url in image_urls]
        return await asyncio.gather(*tasks)

    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        self._connection.close()
        try:
            # Clean up temp directory
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {str(e)}")
