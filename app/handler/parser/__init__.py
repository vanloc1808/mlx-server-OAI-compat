from app.handler.parser.base import BaseToolParser
from app.handler.parser.qwen3 import Qwen3Parser

__all__ = ['BaseToolParser', 'Qwen3Parser']

parser_map = {
    'qwen3': Qwen3Parser
}