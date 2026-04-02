from __future__ import annotations

import os
import json
from typing import Any, AsyncIterator, List, Union, Literal
import traceback

import httpx
from videosdk.agents import (
    LLM, LLMResponse, ChatContext, ChatRole, ChatMessage, 
    ToolChoice, FunctionTool, 
    ChatContent,
    FunctionCall, FunctionCallOutput,
)
from videosdk.agents.utils import build_openai_schema, is_function_tool

SARVAM_CHAT_COMPLETION_URL = "https://api.sarvam.ai/v1/chat/completions" 
DEFAULT_MODEL = "sarvam-m" 

class SarvamAILLM(LLM):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_completion_tokens: int | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        wiki_grounding:bool = False
    ) -> None:
        """Initialize the SarvamAI LLM plugin.

        Args:
            api_key (Optional[str], optional): SarvamAI API key. Defaults to None.
            model (str): The model to use for the LLM plugin. Defaults to "sarvam-m".
            temperature (float): The temperature to use for the LLM plugin. Defaults to 0.7.
            tool_choice (ToolChoice): The tool choice to use for the LLM plugin. Defaults to "auto".
            max_completion_tokens (Optional[int], optional): The maximum completion tokens to use for the LLM plugin. Defaults to None.
            reasoning_effort (Optional[Literal["low", "medium", "high"]], optional): The reasoning effort to use for the LLM plugin. Defaults to None.
            wiki_grounding (bool): enables Wikipedia search. Defaults to False
        """
        super().__init__()
        self.api_key = api_key or os.getenv("SARVAMAI_API_KEY")
        if not self.api_key:
            raise ValueError("Sarvam AI API key must be provided either through api_key parameter or SARVAMAI_API_KEY environment variable")
        
        self.model = model
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.wiki_grounding = wiki_grounding
        self._cancelled = False
        
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=30.0, write=5.0, pool=5.0),
            follow_redirects=True,
        )

    async def chat(
        self,
        messages: ChatContext,
        tools: list[FunctionTool] | None = None,
        **kwargs: Any
    ) -> AsyncIterator[LLMResponse]:
        self._cancelled = False
        
        def _extract_text_content(content: Union[str, List[ChatContent]]) -> str:
            if isinstance(content, str):
                return content
            text_parts = [part for part in content if isinstance(part, str)]
            return "\n".join(text_parts)

        system_prompt = None
        message_items = list(messages.items)
        if (
            message_items
            and isinstance(message_items[0], ChatMessage)
            and message_items[0].role == ChatRole.SYSTEM
        ):
            system_prompt = {
                "role": "system",
                "content": _extract_text_content(message_items.pop(0).content),
            }

        cleaned_messages = []
        last_role = None
        i = 0
        while i < len(message_items):
            msg = message_items[i]

            if isinstance(msg, FunctionCall):
                tool_calls = []
                while i < len(message_items) and isinstance(message_items[i], FunctionCall):
                    fc = message_items[i]
                    tool_calls.append({
                        "id": fc.call_id,
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": fc.arguments,
                        },
                    })
                    i += 1
                cleaned_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                })
                last_role = "assistant"
                continue

            if isinstance(msg, FunctionCallOutput):
                cleaned_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.call_id,
                    "content": msg.output,
                })
                last_role = "tool"
                i += 1
                continue

            if not isinstance(msg, ChatMessage):
                i += 1
                continue

            current_role_str = msg.role.value
            
            if not cleaned_messages and current_role_str == 'assistant':
                i += 1
                continue

            text_content = _extract_text_content(msg.content)
            if not text_content.strip():
                i += 1
                continue

            if last_role == 'user' and current_role_str == 'user':
                cleaned_messages[-1]['content'] += ' ' + text_content
                i += 1
                continue
            
            if last_role == current_role_str:
                cleaned_messages.pop()

            cleaned_messages.append({"role": current_role_str, "content": text_content})
            last_role = current_role_str
            i += 1

        final_messages = [system_prompt] + cleaned_messages if system_prompt else cleaned_messages
        
        try:
            payload = {
                "model": self.model,
                "messages": final_messages,
                "temperature": self.temperature,
                "stream": True,
                "reasoning_effort": self.reasoning_effort,
                "wiki_grounding": self.wiki_grounding,
                "top_p":1
            }

            if tools:
                formatted_tools = []
                for tool in tools:
                    if not is_function_tool(tool):
                        continue
                    try:
                        tool_schema = build_openai_schema(tool)
                        inner = {k: v for k, v in tool_schema.items() if k != "type"}
                        formatted_tools.append({"type": "function", "function": inner})
                    except Exception as e:
                        self.emit("error", f"Failed to format tool {tool}: {e}")
                        continue
                if formatted_tools:
                    payload["tools"] = formatted_tools
                    payload["tool_choice"] = self.tool_choice
    
            if self.max_completion_tokens:
                payload['max_tokens'] = self.max_completion_tokens
            
            payload.update(kwargs)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            async with self._client.stream("POST", SARVAM_CHAT_COMPLETION_URL, json=payload, headers=headers) as response:
                response.raise_for_status()
                pending_tool_calls = {}
                content_buffer = ""
                MIN_CHUNK_SIZE = 30

                async for line in response.aiter_lines():
                    if self._cancelled:
                        break
                        
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if not data_str:
                        continue
                    if data_str == "[DONE]":
                        break
                    
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    if "content" in delta and delta["content"]:
                        content_buffer += delta["content"]
                        stripped = content_buffer.strip()
                        if stripped and len(stripped) >= MIN_CHUNK_SIZE and any(stripped.endswith(p) for p in (".", "!", "?", ",", ";", ":")):
                            yield LLMResponse(content=content_buffer, role=ChatRole.ASSISTANT)
                            content_buffer = ""
                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            if idx not in pending_tool_calls:
                                pending_tool_calls[idx] = {
                                    "id": tc.get("id", ""),
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": tc.get("function", {}).get("arguments", ""),
                                }
                            else:
                                fn = tc.get("function", {})
                                if fn.get("name"):
                                    pending_tool_calls[idx]["name"] += fn["name"]
                                if fn.get("arguments"):
                                    pending_tool_calls[idx]["arguments"] += fn["arguments"]

                    if finish_reason and pending_tool_calls:
                        for tc_data in sorted(pending_tool_calls.values(), key=lambda x: x["id"]):
                            try:
                                args = json.loads(tc_data["arguments"])
                            except json.JSONDecodeError:
                                self.emit("error", f"Failed to parse tool call arguments: {tc_data['arguments']}")
                                args = {}
                            yield LLMResponse(
                                content="",
                                role=ChatRole.ASSISTANT,
                                metadata={
                                    "function_call": {
                                        "name": tc_data["name"],
                                        "arguments": args,
                                        "id": tc_data["id"],
                                    },
                                },
                            )
                        pending_tool_calls = {}

                if content_buffer.strip():
                    yield LLMResponse(content=content_buffer, role=ChatRole.ASSISTANT)


                if pending_tool_calls:
                    for tc_data in sorted(pending_tool_calls.values(), key=lambda x: x["id"]):
                        try:
                            args = json.loads(tc_data["arguments"])
                        except json.JSONDecodeError:
                            self.emit("error", f"Failed to parse tool call arguments: {tc_data['arguments']}")
                            args = {}
                        yield LLMResponse(
                            content="",
                            role=ChatRole.ASSISTANT,
                            metadata={
                                "function_call": {
                                    "name": tc_data["name"],
                                    "arguments": args,
                                    "id": tc_data["id"],
                                },
                            },
                        )

        except httpx.HTTPStatusError as e:
            if not self._cancelled:
                error_message = f"Sarvam AI API error: {e.response.status_code}"
                try:
                    error_body = await e.response.aread()
                    error_text = error_body.decode()
                    error_message += f" - {error_text}"
                except Exception:
                    pass
                self.emit("error", Exception(error_message))
            raise
        except Exception as e:
            if not self._cancelled:
                traceback.print_exc()
                self.emit("error", e)
            raise

    async def cancel_current_generation(self) -> None:
        self._cancelled = True

    async def aclose(self) -> None:
        await self.cancel_current_generation()
        if self._client:
            await self._client.aclose()
        await super().aclose()
