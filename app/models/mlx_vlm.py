from typing import List, Dict, Union, Generator
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, generate, stream_generate, load_image_processor


# Default model parameters
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0

class MLX_VLM:
    """
    A wrapper class for MLX Vision Language Model that handles both streaming and non-streaming inference.
    
    This class provides a unified interface for generating text responses from images and text prompts,
    supporting both streaming and non-streaming modes.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the MLX_VLM model.
        
        Args:
            model_path (str): Path to the model directory containing model weights and configuration.
            
        Raises:
            ValueError: If model loading fails.
        """
        try:
            self.model, self.processor = load(model_path, lazy=False, trust_remote_code=True)
            self.config = load_config(model_path, trust_remote_code=True)
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
        
    def __call__(
        self, 
        messages: List[Dict[str, str]], 
        images: List[str] = None,
        stream: bool = False, 
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate text response from images and messages.
        
        Args:
            images (List[str]): List of image paths to process.
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content' keys.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            **kwargs: Additional model parameters (temperature, max_tokens, etc.)
            
        Returns:
            Union[str, Generator[str, None, None]]: 
                - If stream=False: Complete response as string
                - If stream=True: Generator yielding response chunks
        """
        # Prepare the prompt using the chat template
        prompt = apply_chat_template(
            self.processor, 
            self.config, 
            messages, 
            add_generation_prompt=True,
            num_images=len(images) if images else 0
        )       
        print("PROMPT", prompt)
        # Set default parameters if not provided
        model_params = {
            "temperature": kwargs.get("temperature", DEFAULT_TEMPERATURE),
            "max_tokens": kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            **kwargs
        }
        
        if not stream:
            # Non-streaming mode: return complete response
            return generate(
                self.model,
                self.processor,
                prompt,
                image=images,
                **model_params
            )
        else:
            # Streaming mode: return generator of chunks
            return stream_generate(
                self.model,
                self.processor,
                prompt,
                images,
                **model_params
            )