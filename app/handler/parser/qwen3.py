from app.handler.parser.base import BaseToolParser

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"

class Qwen3Parser(BaseToolParser):
    """Parser for Qwen3 model's tool response format."""
    
    def __init__(self):
        super().__init__(
            tool_open=TOOL_OPEN,
            tool_close=TOOL_CLOSE   
        )