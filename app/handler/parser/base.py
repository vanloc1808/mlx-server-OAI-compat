import json
import uuid
from typing import List, Dict, Any, Tuple

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
            end_tool = content.find(self.tool_close, len(self.tool_open))
            if end_tool == -1:
                break
            else:
                tool_content = content[len(self.tool_open):end_tool].strip()

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
            end_tool = buffer.find(self.tool_close, len(self.tool_open))
            if end_tool == -1:
                break
            try:
                json_output = json.loads(buffer[len(self.tool_open):end_tool].strip())
                return json_output, buffer[end_tool + len(self.tool_close):]
            except json.JSONDecodeError:
                print("Error parsing tool call: ", buffer[len(self.tool_open):end_tool].strip())
                break
        return None, buffer