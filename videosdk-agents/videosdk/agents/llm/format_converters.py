from __future__ import annotations

import base64
import json
from typing import List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .chat_context import ChatContext

from .chat_context import (
    ChatMessage,
    ChatRole,
    FunctionCall,
    FunctionCallOutput,
    ImageContent,
    ChatContent,
)

def _format_content_openai(content: Union[str, List[ChatContent]]) -> Union[str, list]:
    """Format content for OpenAI API."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    formatted_parts = []
    for part in content:
        if isinstance(part, str):
            formatted_parts.append({"type": "text", "text": part})
        elif isinstance(part, ImageContent):
            image_url_data = {"url": part.to_data_url()}
            if part.inference_detail != "auto":
                image_url_data["detail"] = part.inference_detail
            formatted_parts.append({"type": "image_url", "image_url": image_url_data})
    return formatted_parts


def to_openai_messages(ctx: ChatContext, *, reasoning_model: bool = False) -> list[dict]:
    """
    Convert context to OpenAI chat completion messages format.

    Handles ChatMessage, FunctionCall batching, FunctionCallOutput,
    reasoning model role mapping, image content, and interrupted messages.

    Args:
        ctx: The ChatContext to convert.
        reasoning_model: If True, maps 'system' role to 'developer'.

    Returns:
        list[dict]: OpenAI-formatted messages list.
    """
    openai_messages = []
    i = 0
    items = ctx.items
    while i < len(items):
        msg = items[i]
        if msg is None:
            i += 1
            continue
        if isinstance(msg, ChatMessage):
            role = msg.role.value
            if reasoning_model and role == "system":
                role = "developer"
            content = _format_content_openai(msg.content)
            openai_messages.append({
                "role": role,
                "content": content,
                **({"name": msg.name} if hasattr(msg, "name") and getattr(msg, "name", None) else {}),
            })
            i += 1
        elif isinstance(msg, FunctionCall):
            tool_calls = []
            while i < len(items) and isinstance(items[i], FunctionCall):
                fc = items[i]
                tool_calls.append({
                    "id": fc.call_id,
                    "type": "function",
                    "function": {"name": fc.name, "arguments": fc.arguments},
                })
                i += 1
            openai_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            })
        elif isinstance(msg, FunctionCallOutput):
            openai_messages.append({
                "role": "tool",
                "tool_call_id": msg.call_id,
                "content": msg.output,
            })
            i += 1
        else:
            i += 1
    return openai_messages




def _format_content_anthropic(content):
    """Format content for Anthropic API."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    has_images = any(isinstance(p, ImageContent) for p in content)
    if not has_images and len(content) == 1 and isinstance(content[0], str):
        return content[0]

    formatted_parts = []
    image_parts = [p for p in content if isinstance(p, ImageContent)]
    text_parts = [p for p in content if isinstance(p, str)]

    for part in image_parts:
        data_url = part.to_data_url()
        if data_url.startswith("data:"):
            header, b64_data = data_url.split(",", 1)
            media_type = header.split(";")[0].split(":")[1]
            formatted_parts.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64_data},
            })
        else:
            formatted_parts.append(
                {"type": "image", "source": {"type": "url", "url": data_url}}
            )
    for part in text_parts:
        formatted_parts.append({"type": "text", "text": part})
    return formatted_parts


def to_anthropic_messages(ctx: ChatContext, *, caching: bool = False) -> tuple[list[dict], Optional[str]]:
    """
    Convert context to Anthropic messages format with role alternation enforced.

    Args:
        ctx: The ChatContext to convert.
        caching: If True, marks applicable messages with cache_control.

    Returns:
        tuple: (messages_list, system_content_or_none)
    """
    anthropic_messages: list[dict] = []
    system_content: str | None = None
    pending_tool_results: dict[str, FunctionCallOutput] = {}

    for item in ctx.items:
        if isinstance(item, ChatMessage):
            if item.role == ChatRole.SYSTEM:
                if isinstance(item.content, list):
                    system_content = next(
                        (str(p) for p in item.content if isinstance(p, str)), ""
                    )
                else:
                    system_content = str(item.content)
                continue
            else:
                content = _format_content_anthropic(item.content)
                anthropic_messages.append({"role": item.role.value, "content": content})
        elif isinstance(item, FunctionCall):
            tool_use_block = {
                "type": "tool_use",
                "id": item.call_id,
                "name": item.name,
                "input": (
                    json.loads(item.arguments)
                    if isinstance(item.arguments, str)
                    else item.arguments
                ),
            }
            if (
                anthropic_messages
                and anthropic_messages[-1]["role"] == "assistant"
                and isinstance(anthropic_messages[-1]["content"], list)
                and any(p.get("type") == "tool_use" for p in anthropic_messages[-1]["content"])
            ):
                anthropic_messages[-1]["content"].append(tool_use_block)
            else:
                anthropic_messages.append({"role": "assistant", "content": [tool_use_block]})
        elif isinstance(item, FunctionCallOutput):
            pending_tool_results[item.call_id] = item

    final_messages: list[dict] = []
    for msg in anthropic_messages:
        final_messages.append(msg)
        if isinstance(msg.get("content"), list) and any(
            part.get("type") == "tool_use" for part in msg["content"]
        ):
            tool_result_blocks = []
            for part in msg["content"]:
                if part.get("type") != "tool_use":
                    continue
                call_id = part["id"]
                if call_id in pending_tool_results:
                    tool_result = pending_tool_results.pop(call_id)
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": tool_result.output,
                        "is_error": tool_result.is_error,
                    })
            if tool_result_blocks:
                final_messages.append({"role": "user", "content": tool_result_blocks})

    merged: list[dict] = []
    for msg in final_messages:
        if not merged or merged[-1]["role"] != msg["role"]:
            merged.append({"role": msg["role"], "content": msg["content"]})
        else:
            prev_content = merged[-1]["content"]
            new_content = msg["content"]
            if isinstance(prev_content, str):
                prev_content = [{"type": "text", "text": prev_content}]
            if isinstance(new_content, str):
                new_content = [{"type": "text", "text": new_content}]
            merged[-1]["content"] = prev_content + new_content

    return merged, system_content


async def to_google_contents(
    ctx: ChatContext,
    *,
    thought_signatures: dict | None = None,
) -> tuple[list, Optional[str]]:
    """
    Convert context to Google Gemini contents format.

    Args:
        ctx: The ChatContext to convert.
        thought_signatures: Dict mapping function names to thought signature bytes.

    Returns:
        tuple: (contents_list, system_instruction_or_none)

    Note: Requires ``google.genai.types`` to be available.
    """
    try:
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai package is required for to_google_contents(). "
            "Install it with: pip install google-genai"
        )

    thought_signatures = thought_signatures or {}
    contents = []
    system_instruction = None

    for item in ctx.items:
        if isinstance(item, ChatMessage):
            if item.role == ChatRole.SYSTEM:
                if isinstance(item.content, list):
                    system_instruction = next(
                        (str(p) for p in item.content if isinstance(p, str)), ""
                    )
                else:
                    system_instruction = str(item.content)
                continue

            parts = []
            raw_content = item.content if isinstance(item.content, list) else [item.content]
            for part in raw_content:
                if part is None:
                    continue
                if isinstance(part, str):
                    parts.append(types.Part(text=part))
                elif isinstance(part, ImageContent):
                    data_url = part.to_data_url()
                    if data_url.startswith("data:"):
                        header, b64_data = data_url.split(",", 1)
                        media_type = header.split(";")[0].split(":")[1]
                        image_bytes = base64.b64decode(b64_data)
                        parts.append(
                            types.Part(inline_data=types.Blob(mime_type=media_type, data=image_bytes))
                        )

            role = "model" if item.role == ChatRole.ASSISTANT else "user"
            contents.append(types.Content(role=role, parts=parts))

        elif isinstance(item, FunctionCall):
            args = json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments
            function_call = types.FunctionCall(name=item.name, args=args)
            fc_part = types.Part(function_call=function_call)

            thought_sig_b64 = (item.metadata or {}).get("thought_signature")
            if thought_sig_b64:
                try:
                    sig_bytes = base64.b64decode(thought_sig_b64)
                    fc_part = types.Part(function_call=function_call, thought_signature=sig_bytes)
                except Exception:
                    pass
            elif item.name in thought_signatures:
                fc_part = types.Part(
                    function_call=function_call,
                    thought_signature=thought_signatures[item.name],
                )
            contents.append(types.Content(role="model", parts=[fc_part]))

        elif isinstance(item, FunctionCallOutput):
            function_response = types.FunctionResponse(
                name=item.name, response={"output": item.output}
            )
            contents.append(
                types.Content(role="user", parts=[types.Part(function_response=function_response)])
            )

    return contents, system_instruction
