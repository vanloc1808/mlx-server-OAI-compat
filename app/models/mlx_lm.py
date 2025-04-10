from mlx_lm.utils import load
from mlx_lm.generate import (
    generate,
    stream_generate,
)
from typing import List, Dict, Union, Generator


class MLX_LM:
    """
    A wrapper class for MLX Language Model that handles both streaming and non-streaming inference.
    
    This class provides a unified interface for generating text responses from text prompts,
    supporting both streaming and non-streaming modes.
    """

    def __init__(self, model_path: str):
        try:
            self.model, self.tokenizer = load(model_path)
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
        
    def __call__(
        self, 
        messages: List[Dict[str, str]], 
        stream: bool = False, 
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate text response from the model.

        Args:
            messages (List[Dict[str, str]]): List of messages in the conversation.
            stream (bool): Whether to stream the response.
        """
        try:
            if not messages:
                raise ValueError("Messages must be provided")
            
            # Prepare input tokens
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,                
            )
            if not stream:
                # Non-streaming mode: return complete response
                return generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    **kwargs
                )
            else:
                # Streaming mode: return generator of chunks
                return stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    **kwargs
                )
        except Exception as e:
            raise ValueError(f"Error generating text: {str(e)}")

