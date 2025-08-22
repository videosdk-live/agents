from __future__ import annotations

import os
from typing import Any, AsyncIterator, List, Union
import json

import httpx
import openai
from videosdk.agents import (
    LLM,
    LLMResponse,
    ChatContext,
    ChatRole,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    ToolChoice,
    FunctionTool,
    is_function_tool,
    build_openai_schema,
)
from videosdk.agents.llm.chat_context import ChatContent, ImageContent

class OpenAILLM(LLM):
    
    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_completion_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either through api_key parameter or OPENAI_API_KEY environment variable")
        
        self.model = model
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.max_completion_tokens = max_completion_tokens
        self._cancelled = False
        
        self._client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url or None,
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

    async def chat(
        self,
        messages: ChatContext,
        tools: list[FunctionTool] | None = None,
        **kwargs: Any
    ) -> AsyncIterator[LLMResponse]:
        """
        Implement chat functionality using OpenAI's chat completion API
        
        Args:
            messages: ChatContext containing conversation history
            tools: Optional list of function tools available to the model
            **kwargs: Additional arguments passed to the OpenAI API
            
        Yields:
            LLMResponse objects containing the model's responses
        """
        self._cancelled = False
        
        def _format_content(content: Union[str, List[ChatContent]]):
            if isinstance(content, str):
                return content

            formatted_parts = []
            for part in content:
                if isinstance(part, str):
                    formatted_parts.append({"type": "text", "text": part})
                elif isinstance(part, ImageContent):
                    image_url_data = {"url": part.to_data_url()}
                    if part.inference_detail != "auto":
                        image_url_data["detail"] = part.inference_detail
                    formatted_parts.append(
                        {
                            "type": "image_url",
                            "image_url": image_url_data,
                        }
                    )
            return formatted_parts

        completion_params = {
            "model": self.model,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": _format_content(msg.content),
                    **({"name": msg.name} if hasattr(msg, "name") else {}),
                }
                if isinstance(msg, ChatMessage)
                else {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": msg.name, "arguments": msg.arguments},
                }
                if isinstance(msg, FunctionCall)
                else {
                    "role": "function",
                    "name": msg.name,
                    "content": msg.output,
                }
                if isinstance(msg, FunctionCallOutput)
                else None
                for msg in messages.items
                if msg is not None
            ],
            "temperature": self.temperature,
            "stream": True,
            "max_tokens": self.max_completion_tokens,
        }
        if tools:
            formatted_tools = []
            for tool in tools:
                if not is_function_tool(tool):
                    continue
                try:
                    tool_schema = build_openai_schema(tool)
                    formatted_tools.append(tool_schema)
                except Exception as e:
                    self.emit("error", f"Failed to format tool {tool}: {e}")
                    continue
            
            if formatted_tools:
                completion_params["functions"] = formatted_tools
                completion_params["function_call"] = self.tool_choice
        completion_params.update(kwargs)
        try:
            response_stream = await self._client.chat.completions.create(**completion_params)
            
            current_content = ""
            current_function_call = None

            async for chunk in response_stream:
                if self._cancelled:
                    break
                
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                if delta.function_call:
                    if current_function_call is None:
                        current_function_call = {
                            "name": delta.function_call.name or "",
                            "arguments": delta.function_call.arguments or ""
                        }
                    else:
                        if delta.function_call.name:
                            current_function_call["name"] += delta.function_call.name
                        if delta.function_call.arguments:
                            current_function_call["arguments"] += delta.function_call.arguments
                elif current_function_call is not None:  
                    try:
                        args = json.loads(current_function_call["arguments"])
                        current_function_call["arguments"] = args
                    except json.JSONDecodeError:
                        self.emit("error", f"Failed to parse function arguments: {current_function_call['arguments']}")
                        current_function_call["arguments"] = {}
                    
                    yield LLMResponse(
                        content="",
                        role=ChatRole.ASSISTANT,
                        metadata={"function_call": current_function_call}
                    )
                    current_function_call = None
                
                elif delta.content is not None:
                    current_content = delta.content
                    yield LLMResponse(
                        content=current_content,
                        role=ChatRole.ASSISTANT
                    )

        except Exception as e:
            if not self._cancelled:
                self.emit("error", e)
            raise

    async def cancel_current_generation(self) -> None:
        self._cancelled = True

    async def aclose(self) -> None:
        """Cleanup resources by closing the HTTP client"""
        await self.cancel_current_generation()
        if self._client:
            await self._client.close()       