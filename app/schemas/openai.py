from typing import Any, Dict, List, Optional, Union

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

# Data Models
class Message(BaseModel):
    """
    Represents a single message in a chat completion request.
    """
    role: str      # The role of the message sender (e.g., 'user', 'assistant')
    content: Union[str, List[VisionContentItem]]  # The content of the message

class ToolCall(BaseModel):
    """
    Represents a tool call within a chat completion request.
    """
    type: str             # The type of tool call
    function: Dict[str, str]  # Details of the function to be called

class EmbeddingRequest(BaseModel):
    """
    Model for embedding requests.
    """
    model: str = Config.EMBEDDING_MODEL     # Model to use, defaults to embedding model
    input: List[str] = Field(..., description="List of text inputs for embedding")  # Text inputs to embed

class ChatCompletionRequest(BaseModel):
    """
    Model for chat completion requests, including messages, streaming option, and tools.
    """
    model: str = Config.TEXT_MODEL          # Model to use, defaults to text model
    messages: List[Message]                 # List of messages in the chat
    stream: Optional[bool] = False          # Whether to stream the response
    tools: Optional[List[Dict[str, Any]]] = None  # Optional list of tools to use
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None  # Tool selection behavior
    max_tokens: Optional[int] = None        # Maximum number of tokens to generate
    temperature: Optional[float] = 0.7      # Sampling temperature
    top_p: Optional[float] = 1.0           # Nucleus sampling parameter
    frequency_penalty: Optional[float] = 0.0  # Frequency penalty
    presence_penalty: Optional[float] = 0.0   # Presence penalty
    stop: Optional[List[str]] = None        # Stop sequences
    n: Optional[int] = 1                    # Number of completions to generate
    response_format: Optional[Dict[str, str]] = None  # Response format specification
    seed: Optional[int] = None              # Random seed for deterministic outputs
    user: Optional[str] = None              # User identifier for tracking

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
        valid_roles = {"user", "assistant", "system"}
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
        If so, switch the model to the vision model.
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
                            self.model = Config.VISION_MODEL    
                            return True
        
        logger.debug(f"No images detected, treating as text-only request")
        return False

    def fix_message_order(self) -> None:
        """
        Ensure that messages alternate between 'user' and 'assistant' roles.
        If consecutive messages have the same role, insert a dummy message with the opposite role.
        """
        if not self.messages:
            return
            
        fixed_messages = []
        last_role = None
        
        for msg in self.messages:
            if isinstance(msg.content, list):
                # Nothing to do for text + image messages
                fixed_messages.append(msg)
                continue
        
            role = msg.role.strip().lower()
            content = msg.content.strip()
            
            # Insert opposite role if needed
            if (last_role in ("user", "assistant")) and role == last_role:
                fixed_messages.append(Message(
                    role="assistant" if last_role == "user" else "user",
                    content=""
                ))
            
            fixed_messages.append(Message(role=role, content=content))
            last_role = role
            
        self.messages = fixed_messages