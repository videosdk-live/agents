from __future__ import annotations

import base64
import time
from enum import Enum
from typing import List, Optional, Union, Literal
import av
from pydantic import BaseModel, Field

from .. import images
from ..images import EncodeOptions, ResizeOptions
from ..utils import FunctionTool, is_function_tool, get_tool_info


class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ImageContent(BaseModel):
    id: str = Field(default_factory=lambda: f"img_{int(time.time())}")
    type: Literal["image"] = "image"
    image: Union[av.VideoFrame, str]
    inference_detail: Literal["auto", "high", "low"] = "auto"
    encode_options: EncodeOptions = Field(
        default_factory=lambda: EncodeOptions(
            format="JPEG",
            quality=90,
            resize_options=ResizeOptions(
                width=320, height=240
            ),
        )
    )
    class Config:
        arbitrary_types_allowed = True

    def to_data_url(self) -> str:
        if isinstance(self.image, str):
            return self.image

        encoded_image = images.encode(self.image, self.encode_options)
        b64_image = base64.b64encode(encoded_image).decode("utf-8")
        return f"data:image/{self.encode_options.format.lower()};base64,{b64_image}"


ChatContent = Union[str, ImageContent]


class FunctionCall(BaseModel):
    """Represents a function call in the chat context"""
    id: str = Field(default_factory=lambda: f"call_{int(time.time())}")
    type: Literal["function_call"] = "function_call"
    name: str
    arguments: str
    call_id: str


class FunctionCallOutput(BaseModel):
    """Represents the output of a function call"""
    id: str = Field(default_factory=lambda: f"output_{int(time.time())}")
    type: Literal["function_call_output"] = "function_call_output"
    name: str
    call_id: str
    output: str
    is_error: bool = False


class ChatMessage(BaseModel):
    """Represents a single message in the chat context"""
    role: ChatRole
    content: Union[str, List[ChatContent]]
    id: str = Field(default_factory=lambda: f"msg_{int(time.time())}")
    type: Literal["message"] = "message"
    created_at: float = Field(default_factory=time.time)
    interrupted: bool = False


ChatItem = Union[ChatMessage, FunctionCall, FunctionCallOutput]


class ChatContext:
    """Manages a conversation context for LLM interactions"""
    
    def __init__(self, items: Optional[List[ChatItem]] = None):
        self._items: List[ChatItem] = items or []

    @classmethod
    def empty(cls) -> ChatContext:
        """Create an empty chat context"""
        return cls([])

    @property
    def items(self) -> List[ChatItem]:
        """Get all items in the context"""
        return self._items

    def add_message(
        self,
        role: ChatRole,
        content: Union[str, List[ChatContent]],
        message_id: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> ChatMessage:
        """Add a new message to the context"""
        if isinstance(content, str):
            content = [content]

        message = ChatMessage(
            role=role,
            content=content,
            id=message_id or f"msg_{int(time.time())}",
            created_at=created_at or time.time(),
        )
        self._items.append(message)
        return message

    def add_function_call(
        self,
        name: str,
        arguments: str,
        call_id: Optional[str] = None
    ) -> FunctionCall:
        """Add a function call to the context"""
        call = FunctionCall(
            name=name,
            arguments=arguments,
            call_id=call_id or f"call_{int(time.time())}"
        )
        self._items.append(call)
        return call

    def add_function_output(
        self,
        name: str,
        output: str,
        call_id: str,
        is_error: bool = False
    ) -> FunctionCallOutput:
        """Add a function output to the context"""
        function_output = FunctionCallOutput(
            name=name,
            output=output,
            call_id=call_id,
            is_error=is_error
        )
        self._items.append(function_output)
        return function_output

    def get_by_id(self, item_id: str) -> Optional[ChatItem]:
        """Find an item by its ID"""
        return next(
            (item for item in self._items if item.id == item_id),
            None
        )

    def copy(
        self,
        *,
        exclude_function_calls: bool = False,
        exclude_system_messages: bool = False,
        tools: Optional[List[FunctionTool]] = None,
    ) -> ChatContext:
        """Create a filtered copy of the chat context"""
        items = []
        valid_tool_names = {get_tool_info(tool).name for tool in (tools or []) if is_function_tool(tool)}

        for item in self._items:
            # Skip function calls if excluded
            if exclude_function_calls and isinstance(item, (FunctionCall, FunctionCallOutput)):
                continue

            # Skip system messages if excluded
            if exclude_system_messages and isinstance(item, ChatMessage) and item.role == ChatRole.SYSTEM:
                continue

            # Filter by valid tools if tools are provided
            if tools and isinstance(item, (FunctionCall, FunctionCallOutput)):
                if item.name not in valid_tool_names:
                    continue

            items.append(item)

        return ChatContext(items)

    def truncate(self, max_items: int) -> ChatContext:
        """Truncate the context to the last N items while preserving system message"""
        system_msg = next(
            (item for item in self._items 
             if isinstance(item, ChatMessage) and item.role == ChatRole.SYSTEM),
            None
        )
        
        new_items = self._items[-max_items:]
        
        while new_items and isinstance(new_items[0], (FunctionCall, FunctionCallOutput)):
            new_items.pop(0)
            
        if system_msg and system_msg not in new_items:
            new_items.insert(0, system_msg)
            
        self._items = new_items
        return self

    def to_dict(self) -> dict:
        """Convert the context to a dictionary representation"""
        return {
            "items": [
                {
                    "type": item.type,
                    "id": item.id,
                    **({"role": item.role.value, "content": item.content} 
                       if isinstance(item, ChatMessage) else {}),
                    **({"name": item.name, "arguments": item.arguments, "call_id": item.call_id}
                       if isinstance(item, FunctionCall) else {}),
                    **({"name": item.name, "output": item.output, "call_id": item.call_id, "is_error": item.is_error}
                       if isinstance(item, FunctionCallOutput) else {})
                }
                for item in self._items
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChatContext:
        """Create a ChatContext from a dictionary representation"""
        items = []
        for item_data in data["items"]:
            if item_data["type"] == "message":
                items.append(ChatMessage(
                    role=ChatRole(item_data["role"]),
                    content=item_data["content"],
                    id=item_data["id"]
                ))
            elif item_data["type"] == "function_call":
                items.append(FunctionCall(
                    name=item_data["name"],
                    arguments=item_data["arguments"],
                    call_id=item_data["call_id"],
                    id=item_data["id"]
                ))
            elif item_data["type"] == "function_call_output":
                items.append(FunctionCallOutput(
                    name=item_data["name"],
                    output=item_data["output"],
                    call_id=item_data["call_id"],
                    is_error=item_data.get("is_error", False),
                    id=item_data["id"]
                ))
        return cls(items)
