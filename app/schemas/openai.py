from typing import Any, Dict, List, Optional, Union
from typing_extensions import Literal

from pydantic import BaseModel, Field, validator


# Configuration
class Config:
    """
    Configuration class holding the default model names for different types of requests.
    """
    TEXT_MODEL = "gpt-4-turbo"          # Default model for text-based chat completions
    VISION_MODEL = "gpt-4-vision-preview"  # Model used for vision-based requests
    EMBEDDING_MODEL = "text-embedding-ada-002"  # Model used for generating embeddings

class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int

# Common models used in both streaming and non-streaming contexts
class ImageUrl(BaseModel):
    """
    Represents an image URL in a message.
    """
    url: str

class VisionContentItem(BaseModel):
    """
    Represents a single content item in a message (text or image).
    """
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class FunctionCall(BaseModel):
    """
    Represents a function call in a message.
    """
    arguments: str
    name: str

class ChatCompletionMessageToolCall(BaseModel):
    """
    Represents a tool call in a message.
    """
    id: str
    function: FunctionCall
    type: Literal["function"]

class Message(BaseModel):
    """
    Represents a message in a chat completion.
    """
    content: Union[str, List[VisionContentItem]]
    refusal: Optional[str] = None
    role: Literal["system", "user", "assistant", "tool"]
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None

# Common request base for both streaming and non-streaming
class ChatCompletionRequestBase(BaseModel):
    """
    Base model for chat completion requests.
    """
    model: str = Config.TEXT_MODEL
    messages: List[Message]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None
    n: Optional[int] = 1
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    user: Optional[str] = None
    enable_thinking: Optional[bool] = False

    @validator("messages")
    def check_messages_not_empty(cls, v):
        """
        Ensure that the messages list is not empty and validate message structure.
        """
        if not v:
            raise ValueError("messages cannot be empty")

        # Validate message history length
        if len(v) > 100:  # OpenAI's limit is typically around 100 messages
            raise ValueError("message history too long")

        # Validate message roles
        valid_roles = {"user", "assistant", "system", "tool"}
        for msg in v:
            if msg.role not in valid_roles:
                raise ValueError(f"invalid role: {msg.role}")

        return v

    @validator("temperature")
    def check_temperature(cls, v):
        """
        Validate temperature is between 0 and 2.
        """
        if v is not None and (v < 0 or v > 2):
            raise ValueError("temperature must be between 0 and 2")
        return v

    @validator("max_tokens")
    def check_max_tokens(cls, v):
        """
        Validate max_tokens is positive and within reasonable limits.
        """
        if v is not None:
            if v <= 0:
                raise ValueError("max_tokens must be positive")
            if v > 4096:  # Typical limit for GPT-4
                raise ValueError("max_tokens too high")
        return v

    def is_vision_request(self) -> bool:
        """
        Check if the request includes image content, indicating a vision-based request.
        """
        import logging
        logger = logging.getLogger(__name__)

        for message in self.messages:
            content = message.content
            if isinstance(content, list):
                for item in content:
                    if hasattr(item, 'type') and item.type == "image_url":
                        if hasattr(item, 'image_url') and item.image_url and item.image_url.url:
                            logger.debug(f"Detected vision request with image: {item.image_url.url[:30]}...")
                            return True

        logger.debug(f"No images detected, treating as text-only request")
        return False

# Non-streaming request and response
class ChatCompletionRequest(ChatCompletionRequestBase):
    """
    Model for non-streaming chat completion requests.
    """
    stream: bool = False

class Choice(BaseModel):
    """
    Represents a choice in a chat completion response.
    """
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    index: int
    message: Message

class ChatCompletionResponse(BaseModel):
    """
    Represents a complete chat completion response.
    """
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[Choice]

# Streaming request and response
class StreamChatCompletionRequest(ChatCompletionRequestBase):
    """
    Model for streaming chat completion requests.
    """
    stream: bool = True

class ChoiceDeltaFunctionCall(BaseModel):
    """
    Represents a function call delta in a streaming response.
    """
    arguments: Optional[str] = None
    name: Optional[str] = None

class ChoiceDeltaToolCall(BaseModel):
    """
    Represents a tool call delta in a streaming response.
    """
    index: Optional[int] = None
    id: Optional[str] = None
    function: Optional[ChoiceDeltaFunctionCall] = None
    type: Optional[str] = None

class Delta(BaseModel):
    """
    Represents a delta in a streaming response.
    """
    content: Optional[str] = None
    function_call: Optional[ChoiceDeltaFunctionCall] = None
    refusal: Optional[str] = None
    role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    tool_calls: Optional[List[ChoiceDeltaToolCall]] = None

class StreamingChoice(BaseModel):
    """
    Represents a choice in a streaming response.
    """
    delta: Delta
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None
    index: int

class ChatCompletionChunk(BaseModel):
    """
    Represents a chunk in a streaming chat completion response.
    """
    id: str
    choices: List[StreamingChoice]
    created: int
    model: str
    object: Literal["chat.completion.chunk"]

# Embedding models
class EmbeddingRequest(BaseModel):
    """
    Model for embedding requests.
    """
    model: str = Config.EMBEDDING_MODEL
    input: List[str] = Field(..., description="List of text inputs for embedding")
    image_url: Optional[str] = Field(default=None, description="Image URL to embed")

class Embedding(BaseModel):
    """
    Represents an embedding object in an embedding response.
    """
    embedding: List[float] = Field(..., description="The embedding vector")
    index: int = Field(..., description="The index of the embedding in the list")
    object: str = Field(default="embedding", description="The object type")

class EmbeddingResponse(BaseModel):
    """
    Represents an embedding response.
    """
    object: str = "list"
    data: List[Embedding]
    model: str

# TTS models
class TTSRequest(BaseModel):
    """
    Model for text-to-speech requests.
    """
    model: str = "gpt-4o-mini-tts"
    input: str = Field(..., description="The text to convert to speech")
    voice: str = Field(..., description="The voice to use for speech synthesis")
    response_format: Optional[str] = Field(default="mp3", description="The format of the audio output")

class TTSResponse(BaseModel):
    """
    Represents a text-to-speech response.
    """
    audio: bytes
    format: str