from __future__ import annotations
import os
import json
from typing import Any, AsyncIterator, List, Union
import httpx
import anthropic
from videosdk.agents import LLM, LLMResponse, ChatContext, ChatRole, ChatMessage, FunctionCall, FunctionCallOutput, ToolChoice, FunctionTool, is_function_tool, build_openai_schema, ImageContent, ChatContent

class AnthropicLLM(LLM):
    
    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_tokens: int = 1024,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> None:
        super().__init__()
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key must be provided either through api_key parameter or ANTHROPIC_API_KEY environment variable")
        
        self.model = model
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p
        self._cancelled = False
        
        self._client = anthropic.AsyncClient(
            api_key=self.api_key,
            base_url=base_url or None,
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
        Implement chat functionality using Anthropic's Claude API
        
        Args:
            messages: ChatContext containing conversation history
            tools: Optional list of function tools available to the model
            **kwargs: Additional arguments passed to the Anthropic API
            
        Yields:
            LLMResponse objects containing the model's responses
        """
        self._cancelled = False
        
        try:
            anthropic_messages, system_content = self._convert_messages_to_anthropic_format(messages)
            completion_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": True,
            }

            if system_content:
                completion_params["system"] = system_content

            if self.top_k is not None:
                completion_params["top_k"] = self.top_k
            if self.top_p is not None:
                completion_params["top_p"] = self.top_p

            if tools:
                formatted_tools = []
                seen_tool_names = set()
                
                for tool in tools:
                    if not is_function_tool(tool):
                        continue
                    try:
                        openai_schema = build_openai_schema(tool)
                        tool_name = openai_schema["name"]
                        
                        if tool_name in seen_tool_names:
                            continue
                        
                        seen_tool_names.add(tool_name)
                        anthropic_tool = {
                            "name": tool_name,
                            "description": openai_schema["description"],
                            "input_schema": openai_schema["parameters"]
                        }
                        formatted_tools.append(anthropic_tool)
                    except Exception as e:
                        self.emit("error", f"Failed to format tool {tool}: {e}")
                        continue
                
                if formatted_tools:
                    completion_params["tools"] = formatted_tools
                    
                    if self.tool_choice == "required":
                        completion_params["tool_choice"] = {"type": "any"}
                    elif self.tool_choice == "auto":
                        completion_params["tool_choice"] = {"type": "auto"}
                    elif self.tool_choice == "none":
                        del completion_params["tools"]

            completion_params.update(kwargs)

            response_stream = await self._client.messages.create(**completion_params)
            
            current_content = ""
            current_tool_call = None
            current_tool_call_id = None
            current_tool_arguments = ""

            async for event in response_stream:
                if self._cancelled:
                    break
                    
                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool_call_id = event.content_block.id
                        current_tool_call = {
                            "name": event.content_block.name,
                            "arguments": ""
                        }
                        current_tool_arguments = ""
                
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        current_content = delta.text
                        yield LLMResponse(
                            content=current_content,
                            role=ChatRole.ASSISTANT
                        )
                    elif delta.type == "input_json_delta":
                        if current_tool_call:
                            current_tool_arguments += delta.partial_json
                
                elif event.type == "content_block_stop":
                    if current_tool_call and current_tool_call_id:
                        try:
                            parsed_args = json.loads(current_tool_arguments) if current_tool_arguments else {}
                            current_tool_call["arguments"] = parsed_args
                        except json.JSONDecodeError:
                            current_tool_call["arguments"] = {}
                        
                        yield LLMResponse(
                            content="",
                            role=ChatRole.ASSISTANT,
                            metadata={
                                "function_call": {
                                    "id": current_tool_call_id,
                                    "name": current_tool_call["name"],
                                    "arguments": current_tool_call["arguments"],
                                    "call_id": current_tool_call_id
                                }
                            }
                        )
                        current_tool_call = None
                        current_tool_call_id = None
                        current_tool_arguments = ""

        except anthropic.APIError as e:
            if not self._cancelled:
                self.emit("error", e)
            raise
        except Exception as e:
            if not self._cancelled:
                self.emit("error", e)
            raise

    async def cancel_current_generation(self) -> None:
        self._cancelled = True

    def _convert_messages_to_anthropic_format(self, messages: ChatContext) -> tuple[list[dict], str | None]:
        """Convert ChatContext to Anthropic message format"""

        def _format_content(content: Union[str, List[ChatContent]]):
            if isinstance(content, str):
                return content

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
                    formatted_parts.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_data,
                            },
                        }
                    )
                else:
                    formatted_parts.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": data_url},
                        }
                    )

            for part in text_parts:
                formatted_parts.append({"type": "text", "text": part})

            return formatted_parts

        anthropic_messages = []
        system_content = None

        for item in messages.items:
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
                    anthropic_messages.append(
                        {"role": item.role.value, "content": _format_content(item.content)}
                    )
            elif isinstance(item, FunctionCall):
                anthropic_messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": item.call_id,
                            "name": item.name,
                            "input": json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments
                        }
                    ]
                })
            elif isinstance(item, FunctionCallOutput):
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": item.call_id,
                            "content": item.output,
                            "is_error": item.is_error
                        }
                    ]
                })
        
        return anthropic_messages, system_content

    async def aclose(self) -> None:
        """Cleanup resources by closing the HTTP client"""
        await self.cancel_current_generation()
        if self._client:
            await self._client.close()
