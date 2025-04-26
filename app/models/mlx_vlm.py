from typing import List, Dict, Union, Generator, Optional
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.models.cache import KVCache, SimpleKVCache
from mlx_vlm.utils import load_config, generate, stream_generate, prepare_inputs
import mlx.core as mx


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
        
    def get_embeddings(
        self,
        prompts: List[str],
        images: Optional[List[str]] = None,
        batch_size: int = 1,
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Get embeddings for a list of prompts and optional images, supporting batch processing.
        Args:
            prompts: List of text prompts
            images: Optional list of image paths (must be same length as prompts if provided)
            batch_size: Size of batches for processing
            normalize: Whether to apply L2 normalization to embeddings
        Returns:
            List of embeddings as float arrays
        """
        if images is None:
            images = []

        # Text-only batch
        if not images:
            # Batch tokenize and pad
            tokenized = [self.processor.tokenizer.encode(self._format_prompt(p, 0), add_special_tokens=True) for p in prompts]
            max_len = max(len(t) for t in tokenized)
            pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
            batch_input_ids = [t + [pad_id] * (max_len - len(t)) for t in tokenized]
            batch_input_ids = mx.array(batch_input_ids)

            # Run in batches
            all_embeddings = []
            for i in range(0, len(prompts), batch_size):
                batch_ids = batch_input_ids[i:i+batch_size]
                embeddings = self.model.language_model.model(batch_ids)
                pooled = self._apply_pooling_strategy(embeddings)
                if normalize:
                    pooled = self._apply_l2_normalization(pooled)
                all_embeddings.extend(pooled.tolist())
            return all_embeddings

        # Image+prompt batch
        if len(images) != len(prompts):
            raise ValueError("If images are provided, must be same length as prompts (one image per prompt)")

        all_embeddings = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_images = images[i:i+batch_size]
            formatted_prompts = [self._format_prompt(p, 1) for p in batch_prompts]
            inputs = prepare_inputs(
                self.processor,
                batch_images,
                formatted_prompts,
                getattr(self.model.config, "image_token_index", None)
            )
            input_ids = inputs["input_ids"]
            pixel_values = inputs.get("pixel_values", None)
            image_grid_thw = inputs.get("image_grid_thw", None)
            inputs_embeds = self.model.get_input_embeddings(input_ids, pixel_values, image_grid_thw)
            embeddings = self.model.language_model.model(None, inputs_embeds=inputs_embeds)
            pooled = self._apply_pooling_strategy(embeddings)
            if normalize:
                pooled = self._apply_l2_normalization(pooled)
            all_embeddings.extend(pooled.tolist())
        return all_embeddings

    def _format_prompt(self, prompt: str, n_images: int) -> str:
        """Format a single prompt using the chat template."""
        return apply_chat_template(
            self.processor,
            self.config,
            prompt,
            add_generation_prompt=True,
            num_images=n_images
        )

    def _prepare_single_input(self, formatted_prompt: str, images: List[str]) -> Dict:
        """Prepare inputs for a single prompt-image pair."""
        return prepare_inputs(
            self.processor,
            images,
            formatted_prompt,
            getattr(self.model.config, "image_token_index", None)
        )

    def _get_single_embedding(
        self,
        inputs: Dict,
        normalize: bool = True
    ) -> List[float]:
        """Get embedding for a single processed input."""
        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values", None)
        
        # Extract additional kwargs
        data_kwargs = {
            k: v for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        image_grid_thw = data_kwargs.pop("image_grid_thw", None)
        
        inputs_embeds = self.model.get_input_embeddings(input_ids, pixel_values, image_grid_thw)
        embeddings = self.model.language_model.model(None, inputs_embeds=inputs_embeds)
        
        # Apply pooling
        pooled_embedding = self._apply_pooling_strategy(embeddings)
        
        # Apply normalization if requested
        if normalize:
            pooled_embedding = self._apply_l2_normalization(pooled_embedding)
        
        return pooled_embedding.tolist()
   
    def _apply_pooling_strategy(self, embeddings: mx.array) -> mx.array:
        """Apply mean pooling to embeddings."""
        return mx.mean(embeddings, axis=1)

    def _apply_l2_normalization(self, embeddings: mx.array) -> mx.array:
        """Apply L2 normalization to embeddings."""
        l2_norms = mx.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (l2_norms + 1e-8)
        return embeddings