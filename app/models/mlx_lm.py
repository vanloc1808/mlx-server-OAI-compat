import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import (
    generate,
    stream_generate,
)
from mlx_lm.sample_utils import make_sampler
from typing import List, Dict, Union, Generator

DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0
DEFAULT_MAX_TOKENS = 256

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
        
    def _apply_pooling_strategy(self, embeddings: mx.array, pooling_strategy: str = "cls") -> mx.array:
        if pooling_strategy == "cls":
            embeddings = embeddings[:, 0, :]
        elif pooling_strategy == "mean":
            embeddings = embeddings.mean(axis=1)
        elif pooling_strategy == "max":
            embeddings = embeddings.max(axis=1)
        return embeddings
    
    def get_embeddings(self, prompt: List[str], pooling_strategy: str = "cls") -> List[float]:
        batch_prompt = []
        for p in prompt:
            add_special_tokens = self.tokenizer.bos_token is None or not p.startswith(
                self.tokenizer.bos_token
            )
            p = self.tokenizer.encode(p, add_special_tokens=add_special_tokens)
            batch_prompt.append(p)
        batch_prompt = mx.array(batch_prompt)
        embeddings = self.model.model(batch_prompt)
        embeddings = self._apply_pooling_strategy(embeddings, pooling_strategy)
        return embeddings.tolist()
        
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
        # Set default parameters if not provided
        temperature = kwargs.get("temperature", DEFAULT_TEMPERATURE)
        top_p = kwargs.get("top_p", DEFAULT_TOP_P)
        seed = kwargs.get("seed", DEFAULT_SEED)

        mx.random.seed(seed)

        # Prepare input tokens
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,                
        )
        if not stream:
            # Non-streaming mode: return complete response
            return generate(
                self.model,
                self.tokenizer,
                prompt,
                sampler = make_sampler(temperature, top_p)
            )
        else:
            # Streaming mode: return generator of chunks
            return stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                sampler = make_sampler(temperature, top_p)
            )