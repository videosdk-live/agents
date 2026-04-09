from __future__ import annotations

import base64
import time
from enum import Enum
from typing import Any, List, Optional, Union, Literal
import av
from pydantic import BaseModel, Field

from .. import images
from ..images import EncodeOptions, ResizeOptions
from ..utils import FunctionTool, is_function_tool, get_tool_info
import logging
logger = logging.getLogger(__name__)

class ChatRole(str, Enum):
    """
    Enumeration of chat roles for message classification.

    Defines the three standard roles used in chat conversations:
    - SYSTEM: Instructions and context for the AI assistant
    - USER: Messages from the human user
    - ASSISTANT: Responses from the AI assistant
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ImageContent(BaseModel):
    """
    Represents image content in chat messages.

    Attributes:
        id (str): Unique identifier for the image. Auto-generated if not provided.
        type (Literal["image"]): Type identifier, always "image".
        image (Union[av.VideoFrame, str]): The image data as VideoFrame or URL string.
        inference_detail (Literal["auto", "high", "low"]): Detail level for LLM image analysis.
        encode_options (EncodeOptions): Configuration for image encoding and resizing.
    """
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
        """
        Convert the image to a data URL format.

        Returns:
            str: A data URL string representing the image.
        """
        if isinstance(self.image, str):
            return self.image

        encoded_image = images.encode(self.image, self.encode_options)
        b64_image = base64.b64encode(encoded_image).decode("utf-8")
        return f"data:image/{self.encode_options.format.lower()};base64,{b64_image}"


ChatContent = Union[str, ImageContent]


class FunctionCall(BaseModel):
    """
    Represents a function call in the chat context.


    Attributes:
        id (str): Unique identifier for the function call. Auto-generated if not provided.
        type (Literal["function_call"]): Type identifier, always "function_call".
        name (str): Name of the function to be called.
        arguments (str): JSON string containing the function arguments.
        call_id (str): Unique identifier linking this call to its output.
        metadata (Optional[dict]): Provider-specific metadata, e.g. Gemini thought_signature bytes
            stored as base64 string, or Anthropic cache control markers.
    """
    id: str = Field(default_factory=lambda: f"call_{int(time.time())}")
    type: Literal["function_call"] = "function_call"
    name: str
    arguments: str
    call_id: str
    metadata: Optional[dict] = None


class FunctionCallOutput(BaseModel):
    """
    Represents the output of a function call.

    Attributes:
        id (str): Unique identifier for the function output. Auto-generated if not provided.
        type (Literal["function_call_output"]): Type identifier, always "function_call_output".
        name (str): Name of the function that was called.
        call_id (str): Identifier linking this output to the original function call.
        output (str): The result or output from the function execution.
        is_error (bool): Flag indicating if the function execution failed.
    """
    id: str = Field(default_factory=lambda: f"output_{int(time.time())}")
    type: Literal["function_call_output"] = "function_call_output"
    name: str
    call_id: str
    output: str
    is_error: bool = False


class ChatMessage(BaseModel):
    """
    Represents a single message in the chat context.

    Attributes:
        role (ChatRole): The role of the message sender (system, user, or assistant).
        content (Union[str, List[ChatContent]]): The message content as text or list of content items.
        id (str): Unique identifier for the message. Auto-generated if not provided.
        type (Literal["message"]): Type identifier, always "message".
        created_at (float): Unix timestamp when the message was created.
        interrupted (bool): Flag indicating if the message was interrupted during generation.
        extra (dict): Flexible metadata (e.g. {"summary": True} for compression-generated messages).
    """
    role: ChatRole
    content: Union[str, List[ChatContent]]
    id: str = Field(default_factory=lambda: f"msg_{int(time.time())}")
    type: Literal["message"] = "message"
    created_at: float = Field(default_factory=time.time)
    interrupted: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


ChatItem = Union[ChatMessage, FunctionCall, FunctionCallOutput]


class ChatContext:
    """
    Manages a conversation context for LLM interactions.
    """

    def __init__(self, items: Optional[List[ChatItem]] = None):
        """
        Initialize the chat context.

        Args:
            items (Optional[List[ChatItem]]): Initial list of chat items. If None, starts with empty context.
        """
        self._items: List[ChatItem] = items or []

    @classmethod
    def empty(cls) -> ChatContext:
        """
        Create an empty chat context.

        Returns:
            ChatContext: A new empty chat context instance.
        """
        return cls([])

    @property
    def items(self) -> List[ChatItem]:
        """
        Get all items in the context.

        Returns:
            List[ChatItem]: List of all conversation items (messages, function calls, outputs).
        """
        return self._items

    def messages(self) -> List[ChatMessage]:
        """
        Return only ChatMessage items, filtering out function calls and outputs.

        Returns:
            List[ChatMessage]: List of all chat messages in the context.
        """
        return [item for item in self._items if isinstance(item, ChatMessage)]

    def turn_count(self) -> int:
        """
        Count the number of user turns (user-assistant exchange pairs).

        Returns:
            int: Number of user messages in the context.
        """
        return sum(
            1 for item in self._items
            if isinstance(item, ChatMessage) and item.role == ChatRole.USER
        )

    def estimated_tokens(self) -> int:
        """
        Rough token estimate for the current context using a ~4 chars per token heuristic.
        Good enough for budget decisions — not a replacement for provider-reported usage.

        Returns:
            int: Estimated token count.
        """
        total = 0
        for item in self._items:
            total += self._estimate_item_tokens(item)
        return total

    def _estimate_item_tokens(self, item: ChatItem) -> int:
        """Estimate tokens for a single chat item."""
        tokens = 4 
        if isinstance(item, ChatMessage):
            parts = item.content if isinstance(item.content, list) else [item.content]
            for part in parts:
                if part is None:
                    continue
                if isinstance(part, str):
                    tokens += len(part) // 4
                elif isinstance(part, ImageContent):
                    tokens += 300  
        elif isinstance(item, FunctionCall):
            tokens += len(item.name) // 4 + 5
            if item.arguments:
                tokens += len(item.arguments) // 4
        elif isinstance(item, FunctionCallOutput):
            tokens += len(item.name) // 4 + 5
            if item.output:
                tokens += len(item.output) // 4
        return tokens

    def add_message(
        self,
        role: ChatRole,
        content: Union[str, List[ChatContent]],
        message_id: Optional[str] = None,
        created_at: Optional[float] = None,
        replace: bool = False,
    ) -> ChatMessage:
        """
        Add a new message to the context.

        Args:
            role (ChatRole): The role of the message sender.
            content (Union[str, List[ChatContent]]): The message content as text or content items.
            message_id (Optional[str], optional): Custom message ID. Auto-generated if not provided.
            created_at (Optional[float], optional): Custom creation timestamp. Uses current time if not provided.
            replace (bool, optional): If True and role is SYSTEM, replaces the existing system message. Defaults to False.

        Returns:
            ChatMessage: The newly created and added message.
        """
        if replace and role == ChatRole.SYSTEM:
            self._items = [
                item for item in self._items
                if not (isinstance(item, ChatMessage) and item.role == ChatRole.SYSTEM)
            ]

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
        """
        Add a function call to the context.

        Args:
            name (str): Name of the function to be called.
            arguments (str): JSON string containing the function arguments.
            call_id (Optional[str], optional): Custom call ID. Auto-generated if not provided.

        Returns:
            FunctionCall: The newly created and added function call.
        """
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
        """
        Add a function output to the context.

        Args:
            name (str): Name of the function that was executed.
            output (str): The result or output from the function execution.
            call_id (str): ID linking this output to the original function call.
            is_error (bool, optional): Flag indicating if the function execution failed. Defaults to False.

        Returns:
            FunctionCallOutput: The newly created and added function output.
        """
        function_output = FunctionCallOutput(
            name=name,
            output=output,
            call_id=call_id,
            is_error=is_error
        )
        self._items.append(function_output)
        return function_output

    def get_by_id(self, item_id: str) -> Optional[ChatItem]:
        """
        Find an item by its ID.

        Args:
            item_id (str): The ID of the item to find.

        Returns:
            Optional[ChatItem]: The found item or None if not found.
        """
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
        """
        Create a filtered copy of the chat context.

        Args:
            exclude_function_calls (bool, optional): Whether to exclude function calls and outputs. Defaults to False.
            exclude_system_messages (bool, optional): Whether to exclude system messages. Defaults to False.
            tools (Optional[List[FunctionTool]], optional): List of available tools to filter function calls by. Defaults to None.

        Returns:
            ChatContext: A new ChatContext instance with the filtered items.
        """
        items = []
        valid_tool_names = {get_tool_info(tool).name for tool in (
            tools or []) if is_function_tool(tool)}

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

    def truncate(
        self,
        max_items: int | None = None,
        max_tokens: int | None = None,
    ) -> ChatContext:
        """
        Truncate the context while preserving system message and summary messages.

        Removes oldest non-system items until both constraints are satisfied.
        Keeps function call/output pairs together to avoid orphaned tool calls.

        Args:
            max_items: Maximum number of items to keep. None means no item limit.
            max_tokens: Maximum estimated token budget. None means no token limit.

        Returns:
            ChatContext: The current context instance after truncation.
        """
        if max_items is None and max_tokens is None:
            return self

        logger.debug(f"Truncating context: {len(self._items)} items, {self.estimated_tokens()} tokens")

        # Identify protected items that must never be removed:
        # - System message (agent instructions)
        # - Summary message (compressed history)
        # - Last user message (LLMs require conversation to end with user turn)
        system_msg = next(
            (item for item in self._items
             if isinstance(item, ChatMessage) and item.role == ChatRole.SYSTEM),
            None
        )
        summary_msg = next(
            (item for item in self._items
             if isinstance(item, ChatMessage) and item.extra.get("summary")),
            None
        )
        last_user_msg = next(
            (item for item in reversed(self._items)
             if isinstance(item, ChatMessage) and item.role == ChatRole.USER),
            None
        )
        protected = {id(m) for m in (system_msg, summary_msg, last_user_msg) if m is not None}

        # Start with all items; remove oldest non-protected until constraints met
        new_items = list(self._items)

        def _needs_trim() -> bool:
            if max_items is not None and len(new_items) > max_items:
                return True
            if max_tokens is not None:
                token_est = sum(self._estimate_item_tokens(it) for it in new_items)
                if token_est > max_tokens:
                    return True
            return False

        while _needs_trim():
            removed = False
            for i, item in enumerate(new_items):
                # Skip protected items
                if id(item) in protected:
                    continue
                # Don't orphan function call pairs — remove them together
                if isinstance(item, FunctionCall):
                    output_idx = next(
                        (j for j in range(i + 1, len(new_items))
                         if isinstance(new_items[j], FunctionCallOutput) and new_items[j].call_id == item.call_id),
                        None
                    )
                    if output_idx is not None:
                        new_items.pop(output_idx)
                        new_items.pop(i)
                    else:
                        new_items.pop(i)
                    removed = True
                    break
                elif isinstance(item, FunctionCallOutput):
                    new_items.pop(i)
                    removed = True
                    break
                else:
                    new_items.pop(i)
                    removed = True
                    break
            if not removed:
                break  # Only protected items remain — stop even if over budget

        # Clean up ALL orphaned function items (call without output, or output without call)
        call_ids_in_list = {item.call_id for item in new_items if isinstance(item, FunctionCall)}
        output_ids_in_list = {item.call_id for item in new_items if isinstance(item, FunctionCallOutput)}
        new_items = [
            item for item in new_items
            if not (
                (isinstance(item, FunctionCall) and item.call_id not in output_ids_in_list)
                or
                (isinstance(item, FunctionCallOutput) and item.call_id not in call_ids_in_list)
            )
        ]

        # Re-insert protected items if they were accidentally removed by orphan cleanup
        if system_msg and system_msg not in new_items:
            new_items.insert(0, system_msg)
        if summary_msg and summary_msg not in new_items:
            insert_pos = 1 if system_msg in new_items else 0
            new_items.insert(insert_pos, summary_msg)
        if last_user_msg and last_user_msg not in new_items:
            new_items.append(last_user_msg)

        self._items = new_items
        logger.debug(f"Truncation complete: {len(self._items)} items, {self.estimated_tokens()} tokens")
        return self

    # ── Provider format conversions ────────────────────────────────────
    # Actual logic lives in llm/format_converters.py. These methods
    # delegate to keep the public API on ChatContext unchanged.

    def to_openai_messages(self, *, reasoning_model: bool = False) -> list[dict]:
        """Convert context to OpenAI chat completion messages format."""
        from .format_converters import to_openai_messages
        return to_openai_messages(self, reasoning_model=reasoning_model)

    def to_anthropic_messages(self, *, caching: bool = False) -> tuple[list[dict], Optional[str]]:
        """Convert context to Anthropic messages format with role alternation enforced."""
        from .format_converters import to_anthropic_messages
        return to_anthropic_messages(self, caching=caching)

    async def to_google_contents(self, *, thought_signatures: dict | None = None) -> tuple[list, Optional[str]]:
        """Convert context to Google Gemini contents format."""
        from .format_converters import to_google_contents
        return await to_google_contents(self, thought_signatures=thought_signatures)

        return contents, system_instruction

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Convert the context to a dictionary representation.

        Returns:
            dict: Dictionary representation of the chat context.
        """
        return {
            "items": [
                {
                    "type": item.type,
                    "id": item.id,
                    **({"role": item.role.value, "content": item.content}
                       if isinstance(item, ChatMessage) else {}),
                    **({"name": item.name, "arguments": item.arguments, "call_id": item.call_id,
                        "metadata": item.metadata}
                       if isinstance(item, FunctionCall) else {}),
                    **({"name": item.name, "output": item.output, "call_id": item.call_id, "is_error": item.is_error}
                       if isinstance(item, FunctionCallOutput) else {})
                }
                for item in self._items
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChatContext:
        """
        Create a ChatContext from a dictionary representation.

        Args:
            data (dict): Dictionary containing the serialized chat context data.

        Returns:
            ChatContext: A new ChatContext instance reconstructed from the data.
        """
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
                    id=item_data["id"],
                    metadata=item_data.get("metadata")
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

    def cleanup(self) -> None:
        """
        Clear all chat context items and references to free memory.
        """
        logger.info(f"Cleaning up ChatContext with {len(self._items)} items")
        for item in self._items:
            if isinstance(item, ChatMessage):
                if isinstance(item.content, list):
                    for content_item in item.content:
                        if isinstance(content_item, ImageContent):
                            content_item.image = None
                item.content = None
            elif isinstance(item, FunctionCall):
                item.arguments = None
            elif isinstance(item, FunctionCallOutput):
                item.output = None
        self._items.clear()
        try:
            import gc
            gc.collect()
            logger.info("ChatContext garbage collection completed")
        except Exception as e:
            logger.error(f"Error during ChatContext garbage collection: {e}")
        
        logger.info("ChatContext cleanup completed")
