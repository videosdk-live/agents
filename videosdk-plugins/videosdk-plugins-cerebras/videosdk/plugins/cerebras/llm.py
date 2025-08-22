from __future__ import annotations

import os
from typing import Any, AsyncIterator, List, Union
import json

from cerebras.cloud.sdk import Cerebras
from videosdk.agents import LLM, LLMResponse, ChatContext, ChatRole, ChatMessage, FunctionCall, FunctionCallOutput, ToolChoice, FunctionTool, is_function_tool, build_openai_schema, ChatContent

class CerebrasLLM(LLM):
    """
    Cerebras LLM implementation using the Cerebras Cloud SDK.
    
    Supported models:
    - llama3.3-70b (default)
    - llama3.1-8b
    - llama-4-scout-17b-16e-instruct
    - qwen-3-32b
    - deepseek-r1-distill-llama-70b (private preview)
    """
    
    def __init__(
        self,
        *,
        model: str = "llama3.3-70b",
        api_key: str | None = None,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        stop: str | None = None,
        user: str | None = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        if not self.api_key:
            raise ValueError("Cerebras API key must be provided either through api_key parameter or CEREBRAS_API_KEY environment variable")
        
        self.model = model
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p
        self.seed = seed
        self.stop = stop
        self.user = user     
        self._cancelled = False
        
        self._client = Cerebras(
            api_key=self.api_key,
        )

    async def chat(
        self,
        messages: ChatContext,
        tools: list[FunctionTool] | None = None,
        **kwargs: Any
    ) -> AsyncIterator[LLMResponse]:
        """
        Implement chat functionality using Cerebras's chat completion API
        
        Args:
            messages: ChatContext containing conversation history
            tools: Optional list of function tools available to the model
            **kwargs: Additional arguments passed to the Cerebras API
            
        Yields:
            LLMResponse objects containing the model's responses
        """
        self._cancelled = False

        def _extract_text_content(content: Union[str, List[ChatContent]]) -> str:
            if isinstance(content, str):
                return content
            text_parts = [part for part in content if isinstance(part, str)]
            return "\n".join(text_parts)

        completion_params = {
            "model": self.model,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": _extract_text_content(msg.content),
                    **({"name": msg.name} if hasattr(msg, "name") else {}),
                }
                if isinstance(msg, ChatMessage)
                else {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{msg.name}",
                            "type": "function",
                            "function": {"name": msg.name, "arguments": msg.arguments},
                        }
                    ],
                }
                if isinstance(msg, FunctionCall)
                else {
                    "role": "tool",
                    "tool_call_id": f"call_{msg.name}",
                    "content": msg.output,
                }
                if isinstance(msg, FunctionCallOutput)
                else None
                for msg in messages.items
                if msg is not None
            ],
            "stream": True,
        }
        if self.temperature is not None:
            completion_params["temperature"] = self.temperature
        if self.max_completion_tokens is not None:
            completion_params["max_completion_tokens"] = self.max_completion_tokens
        if self.top_p is not None:
            completion_params["top_p"] = self.top_p
        if self.seed is not None:
            completion_params["seed"] = self.seed
        if self.stop is not None:
            completion_params["stop"] = self.stop
        if self.user is not None:
            completion_params["user"] = self.user

        if tools:
            formatted_tools = []
            for tool in tools:
                if not is_function_tool(tool):
                    continue
                try:
                    tool_schema = build_openai_schema(tool)
                    cerebras_tool = {
                        "type": "function",
                        "function": tool_schema
                    }
                    formatted_tools.append(cerebras_tool)
                except Exception as e:
                    self.emit("error", f"Failed to format tool {tool}: {e}")
                    continue
            
            if formatted_tools:
                completion_params["tools"] = formatted_tools
                completion_params["tool_choice"] = self.tool_choice

        completion_params.update(kwargs)
        try:
            response_stream = self._client.chat.completions.create(**completion_params)
            current_content = ""
            current_tool_calls = {}

            for chunk in response_stream:
                if self._cancelled:
                    break
                    
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        index = tool_call_delta.index
                        if index not in current_tool_calls:
                            current_tool_calls[index] = {
                                "id": tool_call_delta.id or "",
                                "type": tool_call_delta.type or "function",
                                "function": {
                                    "name": tool_call_delta.function.name or "",
                                    "arguments": tool_call_delta.function.arguments or ""
                                }
                            }
                        else:
                            if tool_call_delta.function.name:
                                current_tool_calls[index]["function"]["name"] += tool_call_delta.function.name
                            if tool_call_delta.function.arguments:
                                current_tool_calls[index]["function"]["arguments"] += tool_call_delta.function.arguments
                elif current_tool_calls:
                    for tool_call in current_tool_calls.values():
                        try:
                            args = json.loads(tool_call["function"]["arguments"])
                            tool_call["function"]["arguments"] = args
                        except json.JSONDecodeError:
                            self.emit("error", f"Failed to parse function arguments: {tool_call['function']['arguments']}")
                            tool_call["function"]["arguments"] = {}
                        
                        yield LLMResponse(
                            content="",
                            role=ChatRole.ASSISTANT,
                            metadata={"function_call": {
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"]["arguments"]
                            }}
                        )
                    current_tool_calls = {}
                
                if delta.content is not None:
                    current_content = delta.content
                    yield LLMResponse(
                        content=current_content,
                        role=ChatRole.ASSISTANT
                    )

        except Exception as e:
            if not self._cancelled:
                error_msg = f"Cerebras API error: {str(e)}"
                self.emit("error", Exception(error_msg))
            raise Exception(error_msg) from e

    async def cancel_current_generation(self) -> None:
        self._cancelled = True

    async def aclose(self) -> None:
        """Cleanup resources"""
        await self.cancel_current_generation()