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
                try:
                    header, encoded = image_url.split(",", 1)
                    data = base64.b64decode(encoded)
                    
                    # Additional logging for debugging
                    logger.debug(f"Base64 image header: {header}")
                    logger.debug(f"Decoded data length: {len(data)} bytes")
                    
                    try:
                        image = Image.open(BytesIO(data))
                    except Exception as img_error:
                        logger.error(f"Failed to open image from base64 data: {str(img_error)}")
                        # Try to infer format from header
                        if "image/jpeg" in header:
                            logger.info("Trying to force JPEG format")
                            # Ensure data is properly formatted for JPEG
                            from PIL import ImageFile
                            ImageFile.LOAD_TRUNCATED_IMAGES = True
                            image = Image.open(BytesIO(data))
                        elif "image/png" in header:
                            logger.info("Trying to force PNG format")
                            image = Image.open(BytesIO(data))
                        else:
                            raise ValueError(f"Unsupported or invalid image format in base64 data: {header}")
                except Exception as base64_error:
                    logger.error(f"Base64 decoding error: {str(base64_error)}")
                    raise ValueError(f"Invalid base64 image data: {str(base64_error)}")
            else:
                headers = {
                    "User-Agent": "mlx-server-OAI-compat/1.0 (https://github.com/cubist38/mlx-server-OAI-compat; cubist38@gmail.com) Python-Requests"
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
