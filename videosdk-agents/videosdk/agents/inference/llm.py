"""
VideoSDK Inference Gateway LLM Plugin.

HTTP-based LLM client that connects to VideoSDK's Inference Gateway.
Supports Google Gemini through a unified interface for cascading pipelines.

Example:
    from videosdk.inference import LLM

    # Using factory method (recommended)
    llm = LLM.google(model="gemini-2.0-flash")

    # Use with CascadingPipeline
    pipeline = CascadingPipeline(stt=stt, llm=llm, tts=tts)
"""

from __future__ import annotations

import json
import os
import logging
import traceback
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import aiohttp

from videosdk.agents import (
    LLM as BaseLLM,
    LLMResponse,
    ChatContext,
    ChatRole,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    ToolChoice,
    FunctionTool,
    is_function_tool,
    build_gemini_schema,
    ChatContent,
    ImageContent,
    ConversationalGraphResponse,
)

logger = logging.getLogger(__name__)

# Default inference gateway URL for HTTP LLM
DEFAULT_LLM_HTTP_URL = "https://inference-gateway.videosdk.live"


class LLM(BaseLLM):
    """
    VideoSDK Inference Gateway LLM Plugin.

    A lightweight LLM client that connects to VideoSDK's Inference Gateway via HTTP.
    Supports Google Gemini models through a unified interface.

    Example:
        # Using factory methods (recommended)
        llm = LLM.google(model="gemini-2.0-flash")

        # Use with CascadingPipeline
        pipeline = CascadingPipeline(stt=stt, llm=llm, tts=tts)
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize the VideoSDK Inference LLM plugin.

        Args:
            provider: LLM provider name (e.g., "google")
            model: Model identifier (e.g., "gemini-2.0-flash")
            temperature: Controls randomness in responses (0.0 to 1.0)
            tool_choice: Tool calling mode ("auto", "required", "none")
            max_output_tokens: Maximum tokens in model responses
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            top_k: Limits tokens considered for each generation step
            presence_penalty: Penalizes token presence (-2.0 to 2.0)
            frequency_penalty: Penalizes token frequency (-2.0 to 2.0)
            base_url: Custom inference gateway URL
        """
        super().__init__()

        self._videosdk_token = os.getenv("VIDEOSDK_AUTH_TOKEN")
        if not self._videosdk_token:
            raise ValueError(
                "VIDEOSDK_AUTH_TOKEN environment variable must be set for authentication"
            )

        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.base_url = base_url or DEFAULT_LLM_HTTP_URL

        # HTTP session state
        self._session: Optional[aiohttp.ClientSession] = None
        self._cancelled: bool = False

    # ==================== Factory Methods ====================

    @staticmethod
    def google(
        *,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        base_url: Optional[str] = None,
    ) -> "LLM":
        """
        Create an LLM instance configured for Google Gemini.

        Args:
            model: Gemini model identifier (default: "gemini-2.0-flash")
                   Options: "gemini-2.0-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro", etc.
            temperature: Controls randomness in responses (0.0 to 1.0)
            tool_choice: Tool calling mode ("auto", "required", "none")
            max_output_tokens: Maximum tokens in model responses
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            top_k: Limits tokens considered for each generation step
            presence_penalty: Penalizes token presence (-2.0 to 2.0)
            frequency_penalty: Penalizes token frequency (-2.0 to 2.0)
            base_url: Custom inference gateway URL

        Returns:
            Configured LLM instance for Google Gemini
        """
        return LLM(
            provider="google",
            model_id=model,
            temperature=temperature,
            tool_choice=tool_choice,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            base_url=base_url,
        )

    # ==================== Core Methods ====================

    async def chat(
        self,
        messages: ChatContext,
        tools: List[FunctionTool] | None = None,
        conversational_graph: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        """
        Implement chat functionality using VideoSDK Inference Gateway.

        Args:
            messages: ChatContext containing conversation history
            tools: Optional list of function tools available to the model
            conversational_graph: Optional conversational graph for structured responses
            **kwargs: Additional arguments passed to the inference gateway

        Yields:
            LLMResponse objects containing the model's responses
        """
        self._cancelled = False

        try:
            # Convert messages to OpenAI-compatible format
            formatted_messages = await self._convert_messages_to_dict(messages)

            # Build request payload
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": formatted_messages,
                "stream": True,
                "temperature": self.temperature,
            }

            # Add optional parameters
            if self.max_output_tokens:
                payload["max_tokens"] = self.max_output_tokens
            if self.top_p is not None:
                payload["top_p"] = self.top_p
            if self.top_k is not None:
                payload["top_k"] = self.top_k
            if self.presence_penalty is not None:
                payload["presence_penalty"] = self.presence_penalty
            if self.frequency_penalty is not None:
                payload["frequency_penalty"] = self.frequency_penalty

            # Add conversational graph response format
            if conversational_graph:
                payload["response_format"] = {
                    "type": "json_object",
                    "schema": ConversationalGraphResponse.model_json_schema(),
                }

            # Add tools if provided
            if tools:
                formatted_tools = self._format_tools(tools)
                if formatted_tools:
                    payload["tools"] = formatted_tools
                    payload["tool_choice"] = self.tool_choice

            # Make streaming HTTP request
            async for response in self._stream_request(payload, conversational_graph):
                if self._cancelled:
                    break
                yield response

        except Exception as e:
            traceback.print_exc()
            if not self._cancelled:
                logger.error(f"[InferenceLLM] Error in chat: {e}")
                self.emit("error", e)
            raise

    async def _stream_request(
        self,
        payload: Dict[str, Any],
        conversational_graph: Any | None = None,
    ) -> AsyncIterator[LLMResponse]:
        """
        Make streaming HTTP request to the inference gateway.

        Args:
            payload: Request payload
            conversational_graph: Optional conversational graph for structured responses

        Yields:
            LLMResponse objects
        """
        if not self._session:
            self._session = aiohttp.ClientSession()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._videosdk_token}",
        }

        # Add provider header
        url = f"{self.base_url}/v1/chat/completions?provider={self.provider}"

        current_content = ""
        streaming_state = {
            "in_response": False,
            "response_start_index": -1,
            "yielded_content_length": 0,
        }

        try:
            logger.debug(
                f"[InferenceLLM] Making request to {self.base_url} (provider={self.provider}, model={self.model})"
            )

            async with self._session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120, connect=30),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()

                    raise Exception(f"HTTP {response.status}: {error_text}")

                # Process SSE stream
                async for line in response.content:
                    if self._cancelled:
                        break

                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue

                    # Handle SSE format
                    if line_str.startswith("data:"):
                        data_str = line_str[5:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            async for llm_response in self._process_chunk(
                                chunk,
                                current_content,
                                streaming_state,
                                conversational_graph,
                            ):
                                if llm_response.content:
                                    current_content += llm_response.content
                                yield llm_response
                        except json.JSONDecodeError as e:
                            logger.warning(f"[InferenceLLM] Failed to parse chunk: {e}")
                            continue

            # Handle conversational graph final response
            if current_content and conversational_graph and not self._cancelled:
                try:
                    parsed_json = json.loads(current_content.strip())
                    yield LLMResponse(
                        content="",
                        role=ChatRole.ASSISTANT,
                        metadata=parsed_json,
                    )
                except json.JSONDecodeError:
                    pass

        except aiohttp.ClientError as e:
            logger.error(f"[InferenceLLM] HTTP request failed: {e}")
            raise

    async def _process_chunk(
        self,
        chunk: Dict[str, Any],
        current_content: str,
        streaming_state: Dict[str, Any],
        conversational_graph: Any,
    ) -> AsyncIterator[LLMResponse]:
        """
        Process a single SSE chunk from the response stream.

        Args:
            chunk: Parsed JSON chunk
            current_content: Accumulated content so far
            streaming_state: State for conversational graph streaming
            conversational_graph: Optional conversational graph

        Yields:
            LLMResponse objects
        """
        choices = chunk.get("choices", [])
        if not choices:
            return

        choice = choices[0]
        delta = choice.get("delta", {})
        # Check for tool calls
        if "tool_calls" in delta:
            for tool_call in delta.get("tool_calls") or []:

                function_data = tool_call.get("function", {})
                function_name = function_data.get("name", "")
                function_args = function_data.get("arguments", "")

                if function_name:
                    # Try to parse arguments if complete
                    try:
                        args_dict = json.loads(function_args) if function_args else {}
                    except json.JSONDecodeError:
                        args_dict = function_args

                    function_call = {
                        "name": function_name,
                        "arguments": args_dict,
                    }
                    yield LLMResponse(
                        content="",
                        role=ChatRole.ASSISTANT,
                        metadata={"function_call": function_call},
                    )

        # Check for content
        content = delta.get("content", "")
        if content:
            if conversational_graph:
                # Stream conversational graph response
                full_content = current_content + content
                for (
                    content_chunk
                ) in conversational_graph.stream_conversational_graph_response(
                    full_content, streaming_state
                ):
                    yield LLMResponse(content=content_chunk, role=ChatRole.ASSISTANT)
            else:
                yield LLMResponse(content=content, role=ChatRole.ASSISTANT)

    async def cancel_current_generation(self) -> None:
        """Cancel the current LLM generation."""
        self._cancelled = True
        logger.debug("[InferenceLLM] Generation cancelled")

    # ==================== Message Conversion ====================

    async def _convert_messages_to_dict(
        self, messages: ChatContext
    ) -> List[Dict[str, Any]]:
        """
        Convert ChatContext to OpenAI-compatible message format.

        Args:
            messages: ChatContext containing conversation history

        Returns:
            List of message dictionaries
        """
        formatted_messages = []

        for item in messages.items:
            if isinstance(item, ChatMessage):
                role = self._map_role(item.role)
                content = await self._format_content(item.content)
                formatted_messages.append({"role": role, "content": content})

            elif isinstance(item, FunctionCall):
                # Convert function call to assistant message with tool_calls
                formatted_messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{item.name}",
                                "type": "function",
                                "function": {
                                    "name": item.name,
                                    "arguments": (
                                        item.arguments
                                        if isinstance(item.arguments, str)
                                        else json.dumps(item.arguments)
                                    ),
                                },
                            }
                        ],
                    }
                )

            elif isinstance(item, FunctionCallOutput):
                # Convert function output to tool message
                formatted_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{item.name}",
                        "content": str(item.output),
                    }
                )

        return formatted_messages

    def _map_role(self, role: ChatRole) -> str:
        """Map ChatRole to OpenAI role string."""
        role_map = {
            ChatRole.SYSTEM: "system",
            ChatRole.USER: "user",
            ChatRole.ASSISTANT: "assistant",
        }
        return role_map.get(role, "user")

    async def _format_content(
        self, content: Union[str, List[ChatContent]]
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Format message content to OpenAI-compatible format.

        Args:
            content: String or list of ChatContent

        Returns:
            Formatted content
        """
        if isinstance(content, str):
            return content

        if len(content) == 1 and isinstance(content[0], str):
            return content[0]

        formatted_parts = []
        for part in content:
            if isinstance(part, str):
                formatted_parts.append({"type": "text", "text": part})
            elif isinstance(part, ImageContent):
                image_url = part.to_data_url()
                image_part = {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
                if part.inference_detail != "auto":
                    image_part["image_url"]["detail"] = part.inference_detail
                formatted_parts.append(image_part)

        return formatted_parts if formatted_parts else ""

    # ==================== Tool Formatting ====================

    def _format_tools(self, tools: List[FunctionTool]) -> List[Dict[str, Any]]:
        """
        Format function tools to OpenAI-compatible format.

        Args:
            tools: List of FunctionTool objects

        Returns:
            List of formatted tool dictionaries
        """
        formatted_tools = []

        for tool in tools:
            if not is_function_tool(tool):
                continue

            try:
                # Build Gemini-compatible schema
                gemini_schema = build_gemini_schema(tool)

                # Convert to OpenAI format
                formatted_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": gemini_schema.get("name", ""),
                            "description": gemini_schema.get("description", ""),
                            "parameters": gemini_schema.get("parameters", {}),
                        },
                    }
                )
            except Exception as e:
                logger.error(f"[InferenceLLM] Failed to format tool: {e}")
                continue

        return formatted_tools

    # ==================== Cleanup ====================

    async def aclose(self) -> None:
        """Clean up all resources."""
        logger.info(f"[InferenceLLM] Closing LLM (provider={self.provider})")

        self._cancelled = True

        # Close HTTP session
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        await super().aclose()

        logger.info("[InferenceLLM] Closed successfully")

    # ==================== Properties ====================

    @property
    def label(self) -> str:
        """Get a descriptive label for this LLM instance."""
        return f"videosdk.inference.LLM.{self.provider}.{self.model}"
