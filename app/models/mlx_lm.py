import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import (
    generate,
    stream_generate,
)
from mlx_lm.sample_utils import make_sampler
from typing import List, Dict, Union, Generator, Optional, Tuple

DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0
DEFAULT_MAX_TOKENS = 512
DEFAULT_BATCH_SIZE = 32

class MLX_LM:
    """
    A wrapper class for MLX Language Model that handles both streaming and non-streaming inference.
    
    This class provides a unified interface for generating text responses from text prompts,
    supporting both streaming and non-streaming modes.
    """

    def __init__(self, model_path: str):
        try:
            self.model, self.tokenizer = load(model_path)
            self.pad_token_id = self.tokenizer.pad_token_id
            self.bos_token = self.tokenizer.bos_token
            self.model_type = self.model.model_type
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
        
    def _apply_pooling_strategy(self, embeddings: mx.array) -> mx.array:
        embeddings = mx.mean(embeddings, axis=1)
        return embeddings
    
    def _apply_l2_normalization(self, embeddings: mx.array) -> mx.array:
        l2_norms = mx.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (l2_norms +  1e-8)
        return embeddings
    
    def _batch_process(self, prompts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[List[int]]:
        """Process prompts in batches with optimized tokenization."""
        all_tokenized = []
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            tokenized_batch = []
            
            # Tokenize all prompts in batch
            for p in batch:
                add_special_tokens = self.bos_token is None or not p.startswith(self.bos_token)
                tokens = self.tokenizer.encode(p, add_special_tokens=add_special_tokens)
                tokenized_batch.append(tokens)
            
            # Find max length in batch
            max_length = max(len(tokens) for tokens in tokenized_batch)
            
            # Pad tokens in a vectorized way
            for tokens in tokenized_batch:
                padding = [self.pad_token_id] * (max_length - len(tokens))
                all_tokenized.append(tokens + padding)
        
        return all_tokenized

    def _preprocess_prompt(self, prompt: str) -> List[int]:
        """Tokenize a single prompt efficiently."""
        add_special_tokens = self.bos_token is None or not prompt.startswith(self.bos_token)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        return mx.array(tokens)
    
    def get_model_type(self) -> str:
        return self.model_type
    
    def get_embeddings(
        self, 
        prompts: List[str], 
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = True
    ) -> List[float]:
        """
        Get embeddings for a list of prompts efficiently.
        
        Args:
            prompts: List of text prompts
            batch_size: Size of batches for processing
            
        Returns:
            List of embeddings as float arrays
        """
        # Process in batches to optimize memory usage
        all_embeddings = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            tokenized_batch = self._batch_process(batch_prompts, batch_size)
            
            # Convert to MLX array for efficient computation
            tokenized_batch = mx.array(tokenized_batch)
            
            # Compute embeddings for batch
            batch_embeddings = self.model.model(tokenized_batch)
            pooled_embedding = self._apply_pooling_strategy(batch_embeddings)
            if normalize:
                pooled_embedding = self._apply_l2_normalization(pooled_embedding)
            all_embeddings.extend(pooled_embedding.tolist())

        return all_embeddings
        
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
            **kwargs: Additional parameters for generation
                - temperature: Sampling temperature (default: 0.0)
                - top_p: Top-p sampling parameter (default: 1.0)
                - seed: Random seed (default: 0)
                - max_tokens: Maximum number of tokens to generate (default: 256)
        """
        # Set default parameters if not provided
        temperature = kwargs.get("temperature", DEFAULT_TEMPERATURE)
        top_p = kwargs.get("top_p", DEFAULT_TOP_P)
        seed = kwargs.get("seed", DEFAULT_SEED)
        max_tokens = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)

        mx.random.seed(seed)

        chat_template_kwargs = {
            "add_generation_prompt": True,
            "tools": kwargs.get("tools", None),
            "enable_thinking": kwargs.get("enable_thinking", False)
        }

        # Prepare input tokens
        prompt = self.tokenizer.apply_chat_template(
            messages,
            **chat_template_kwargs
        )
        
        sampler = make_sampler(temperature, top_p)
        
        if not stream:
            return generate(
                self.model,
                self.tokenizer,
                prompt,
                sampler=sampler,
                max_tokens=max_tokens
            )
        else:
            # Streaming mode: return generator of chunks
            return stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                sampler=sampler,
                max_tokens=max_tokens
            )