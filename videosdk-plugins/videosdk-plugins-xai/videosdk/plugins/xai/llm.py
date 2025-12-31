from __future__ import annotations
import os
from typing import Any, AsyncIterator, List, Union, Dict
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
    ConversationalGraphResponse
)
from videosdk.agents.llm.chat_context import ChatContent, ImageContent


def prepare_strict_schema(schema_dict):
    if isinstance(schema_dict, dict):
        if schema_dict.get("type") == "object":
            schema_dict["additionalProperties"] = False
            if "properties" in schema_dict:
                all_props = list(schema_dict["properties"].keys())
                schema_dict["required"] = all_props
        
        for key, value in schema_dict.items():
            if isinstance(value, dict):
                prepare_strict_schema(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        prepare_strict_schema(item)
    return schema_dict

conversational_graph_schema = prepare_strict_schema(ConversationalGraphResponse.model_json_schema())

class XAILLM(LLM):
    """
    LLM Plugin for xAI (Grok) API.
    Supports Grok-4, and reasoning models with standard client-side function calling.
    """
    
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "grok-4-1-fast-non-reasoning", 
        base_url: str = "https://api.x.ai/v1",
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_completion_tokens: int | None = None,
        tools: List[Union[FunctionTool, Dict[str, Any]]] | None = None,
    ) -> None:
        """Initialize the xAI LLM plugin.

        Args:
            api_key (Optional[str], optional): xAI API key. Defaults to XAI_API_KEY env var.
            model (str): The model to use (e.g., "grok-4", "grok-4-1-fast").
            base_url (str): The base URL for the xAI API. Defaults to "https://api.x.ai/v1".
            temperature (float): The temperature to use. Defaults to 0.7.
            tool_choice (ToolChoice): The tool choice to use. Defaults to "auto".
            max_completion_tokens (Optional[int], optional): Max tokens.
            tools (Optional[List], optional): List of FunctionTools to be available to the LLM.
        """
        super().__init__()
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("xAI API key must be provided either through api_key parameter or XAI_API_KEY environment variable")
        
        self.model = model
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.max_completion_tokens = max_completion_tokens
        self.tools = tools or []
        self._cancelled = False
        
        self._client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=60.0, write=5.0, pool=5.0),
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
        tools: list[Union[FunctionTool, Dict[str, Any]]] | None = None,
        conversational_graph: Any | None = None,
        **kwargs: Any
    ) -> AsyncIterator[LLMResponse]:
        """
        Implement chat functionality using xAI's API via OpenAI SDK compatibility.
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

            
        openai_messages = []
        for msg in messages.items:
            if msg is None:
                continue

            if isinstance(msg, ChatMessage):
                openai_messages.append({
                    "role": msg.role.value,
                    "content": _format_content(msg.content),
                    **({"name": msg.name} if hasattr(msg, "name") else {}),
                })
            elif isinstance(msg, FunctionCall):
                openai_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": getattr(msg, "call_id", getattr(msg, "id", "call_unknown")),
                        "type": "function",
                        "function": {
                            "name": msg.name,
                            "arguments": msg.arguments
                        }
                    }]
                })
            elif isinstance(msg, FunctionCallOutput):
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(msg, "call_id", getattr(msg, "id", "call_unknown")),
                    "content": msg.output,
                })

        completion_params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "stream": True,
            "max_tokens": self.max_completion_tokens,
        }
        
        if conversational_graph:
            completion_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "conversational_graph_response",
                    "strict": True,
                    "schema": conversational_graph_schema
                }
            }

        combined_tools = (self.tools or []) + (tools or [])
        
        if combined_tools:
            formatted_tools = []
            for tool in combined_tools:
                if is_function_tool(tool):
                    try:
                        tool_schema = build_openai_schema(tool)
                        if "function" not in tool_schema:
                            inner_tool = {k: v for k, v in tool_schema.items() if k != "type"}
                            formatted_tools.append({
                                "type": "function",
                                "function": inner_tool
                            })
                        else:
                            formatted_tools.append(tool_schema)
                    except Exception as e:
                        self.emit("error", f"Failed to format tool {tool}: {e}")
                        continue
                elif isinstance(tool, dict):
                    formatted_tools.append(tool)
            
            if formatted_tools:
                completion_params["tools"] = formatted_tools
                completion_params["tool_choice"] = self.tool_choice

        completion_params.update(kwargs)

        try:
            response_stream = await self._client.chat.completions.create(**completion_params)
            
            current_content = ""
            current_tool_calls = {} 
            streaming_state = {
                "in_response": False,
                "response_start_index": -1,
                "yielded_content_length": 0
            }

            async for chunk in response_stream:
                if self._cancelled:
                    break
                
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        idx = tool_call.index
                        if idx not in current_tool_calls:
                            current_tool_calls[idx] = {
                                "id": tool_call.id or "",
                                "name": tool_call.function.name or "",
                                "arguments": tool_call.function.arguments or ""
                            }
                        else:
                            if tool_call.function.name:
                                current_tool_calls[idx]["name"] += tool_call.function.name
                            if tool_call.function.arguments:
                                current_tool_calls[idx]["arguments"] += tool_call.function.arguments

                if delta.content is not None:
                    current_content += delta.content   
                    if conversational_graph:                     
                        for content_chunk in conversational_graph.stream_conversational_graph_response(current_content, streaming_state):
                            yield LLMResponse(content=content_chunk, role=ChatRole.ASSISTANT)
                    else:
                        yield LLMResponse(content=delta.content, role=ChatRole.ASSISTANT)

            if current_tool_calls and not self._cancelled:
                for idx in sorted(current_tool_calls.keys()):
                    tool_data = current_tool_calls[idx]
                    try:
                        args_str = tool_data["arguments"]
                        parsed_args = json.loads(args_str) 
                        
                        yield LLMResponse(
                            content="",
                            role=ChatRole.ASSISTANT,
                            metadata={
                                "function_call": {
                                    "name": tool_data["name"],
                                    "arguments": parsed_args
                                },
                                "tool_call_id": tool_data["id"]
                            }
                        )
                    except json.JSONDecodeError:
                        self.emit("error", f"Failed to parse function arguments for tool {tool_data['name']}")

            if current_content and not self._cancelled and conversational_graph:
                try:
                    parsed_json = json.loads(current_content.strip())
                    yield LLMResponse(
                        content="",
                        role=ChatRole.ASSISTANT,
                        metadata=parsed_json
                    )
                except json.JSONDecodeError:
                     pass

        except Exception as e:
            if not self._cancelled:
                self.emit("error", e)
            raise

    async def cancel_current_generation(self) -> None:
        self._cancelled = True

    async def aclose(self) -> None:
        """Cleanup resources"""
        await self.cancel_current_generation()
        if self._client:
            await self._client.close()
        await super().aclose()