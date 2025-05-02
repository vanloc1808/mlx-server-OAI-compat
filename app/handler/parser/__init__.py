from app.handler.parser.base import BaseToolParser, BaseThinkingParser
from app.handler.parser.qwen3 import Qwen3ToolParser, Qwen3ThinkingParser
from typing import Tuple
__all__ = ['BaseToolParser', 'BaseThinkingParser', 'Qwen3ToolParser', 'Qwen3ThinkingParser']

parser_map = {
    'qwen3': {
        "tool_parser": Qwen3ToolParser,
        "thinking_parser": Qwen3ThinkingParser
    }
}

def get_parser(model_name: str) -> Tuple[BaseToolParser, BaseThinkingParser]:
    if model_name not in parser_map:
        return None, None
        
    model_parsers = parser_map[model_name]
    tool_parser = model_parsers.get("tool_parser")
    thinking_parser = model_parsers.get("thinking_parser")
    
    return (tool_parser() if tool_parser else None, 
            thinking_parser() if thinking_parser else None)