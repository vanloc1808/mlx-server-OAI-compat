import json
import uuid
from typing import List, Dict, Any, Tuple

class BaseThinkingParser:
    def __init__(self, thinking_open: str, thinking_close: str):
        self.thinking_open = thinking_open
        self.thinking_close = thinking_close

    def parse(self, content: str) -> str:
        if self.thinking_open in content:
            end_thinking = content.find(self.thinking_close)
            if end_thinking != -1:
                return content[:end_thinking + len(self.thinking_close)]
        return None
    
    def parse_stream(self, chunk: str, buffer: str = "") -> Tuple[str, bool]:
        if chunk == self.thinking_open:
            return None, True
        if chunk == self.thinking_close:
            return None, False
        if self.thinking_open in buffer:
            return chunk, True
        end_thinking = chunk.find(self.thinking_close)
        if end_thinking != -1:
            return chunk[end_thinking + len(self.thinking_close):], False
        return chunk, True

class BaseToolParser:
    def __init__(self, tool_open: str, tool_close: str):
        self.tool_open = tool_open
        self.tool_close = tool_close

    def get_tool_open(self):
        return self.tool_open
    
    def get_tool_close(self):
        return self.tool_close
    
    def parse(self, content: str) -> Tuple[List[Dict[str, Any]], str]:
        if self.tool_open not in content:
            return content
        res = []
        while True:
            start_tool = content.find(self.tool_open)
            end_tool = content.find(self.tool_close, len(self.tool_open))
            if end_tool == -1:
                break
            else:
                tool_content = content[start_tool + len(self.tool_open):end_tool].strip()

                try:
                    json_output = json.loads(tool_content)  
                    res.append(json_output)
                except json.JSONDecodeError:
                    print("Error parsing tool call: ", tool_content)
                    break
                content = content[end_tool + len(self.tool_close):]
        return res
    
    def parse_stream(self, chunk: str, buffer: str = ""):
        if self.tool_open not in buffer:
            return chunk, buffer
        while True:
            start_tool = buffer.find(self.tool_open)
            end_tool = buffer.find(self.tool_close, len(self.tool_open))
            if end_tool == -1:
                break
            try:
                json_output = json.loads(buffer[start_tool + len(self.tool_open):end_tool].strip())
                return json_output, buffer[end_tool + len(self.tool_close):]
            except json.JSONDecodeError:
                print("Error parsing tool call: ", buffer[start_tool + len(self.tool_open):end_tool].strip())
                break
        return None, buffer