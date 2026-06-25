"""
VideoSDK Inference Gateway LLM Plugin.

HTTP-based LLM client that connects to VideoSDK's Inference Gateway.
Supports Google Gemini, Sarvam AI and VideoSDK LLM through a unified interface for cascading pipelines.

Example:
    from videosdk.inference import LLM

    # Google Gemini (unchanged)
    llm = LLM.google(model_id="gemini-2.0-flash")

    # Sarvam AI
    llm = LLM.sarvam(model_id="sarvam-30b")

    # VideoSDK LLM
    llm = LLM.videosdk(model_id="google.gemma-4-31b")

    # Use with CascadingPipeline
    pipeline = CascadingPipeline(stt=stt, llm=llm, tts=tts)
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union

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
    build_openai_schema,
    build_gemini_schema,
    ChatContent,
    ImageContent,
)
from videosdk.agents.utils import resolve_videosdk_auth_token

logger = logging.getLogger(__name__)

# Default inference gateway URL for HTTP LLM
DEFAULT_LLM_HTTP_URL = "https://inference-gateway.videosdk.live"


class LLM(BaseLLM):
    """
    VideoSDK Inference Gateway LLM Plugin.

    A lightweight LLM client that connects to VideoSDK's Inference Gateway via HTTP.
    Supports Google Gemini and Sarvam AI models through a unified interface.

    Example:
        llm = LLM.google(model_id="gemini-2.0-flash")
        llm = LLM.sarvam(model_id="sarvam-30b")

        pipeline = CascadingPipeline(stt=stt, llm=llm, tts=tts)
    """

    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        base_url: Optional[str] = None,
        config: Dict[str, Any] | None = None,
        # Sarvam-specific
        wiki_grounding: bool = False,
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        auth_token: Optional[str] = None,
    ) -> None:
        """
        Initialize the VideoSDK Inference LLM plugin.

        Args:
            provider: LLM provider name (e.g., "google", "sarvam")
            model_id: Model identifier — consistent with STT/TTS convention
            temperature: Controls randomness in responses (0.0 to 1.0)
            tool_choice: Tool calling mode ("auto", "required", "none")
            max_output_tokens: Maximum tokens in model responses
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            top_k: Limits tokens considered per step — Google only
            presence_penalty: Penalizes token presence (-2.0 to 2.0)
            frequency_penalty: Penalizes token frequency (-2.0 to 2.0)
            base_url: Custom inference gateway URL
            wiki_grounding: Enable Wikipedia search grounding — Sarvam only
            reasoning_effort: Reasoning depth "low"|"medium"|"high" — Sarvam only
            stop: Up to 4 stop sequences — Sarvam only
            auth_token: VideoSDK auth token. Falls back to the resolved token
                (RoomOptions/WorkerOptions, VIDEOSDK_AUTH_TOKEN, or
                VIDEOSDK_API_KEY + VIDEOSDK_SECRET_KEY) when not provided.
        """
        super().__init__()

        self._videosdk_token = resolve_videosdk_auth_token(auth_token)
        if not self._videosdk_token:
            raise ValueError(
                "VIDEOSDK_AUTH_TOKEN environment variable must be set for authentication"
            )

        self.provider = provider
        self.model_id = model_id
        self.model = model_id  # OpenAI-compat alias used in request payload
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.base_url = base_url or DEFAULT_LLM_HTTP_URL
        self.config = config or {}

        # Sarvam-specific params
        self.wiki_grounding = wiki_grounding
        self.reasoning_effort = reasoning_effort
        self.stop = stop

        # HTTP session state
        self._session: Optional[aiohttp.ClientSession] = None
        self._cancelled: bool = False

    # ==================== Factory Methods ====================

    @staticmethod
    def google(
        *,
        model_id: str = "gemini-2.5-flash",
        config: Optional[Dict] = None,
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
            model_id: Gemini model identifier (default: "gemini-2.5-flash")
                Stable: "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"
                Preview: "gemini-3.1-pro-preview", "gemini-3-flash-preview",
                         "gemini-3.1-flash-lite-preview"
            config: Optional extra config dict (merged on top of defaults)
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
        resolved_config: Dict[str, Any] = {"model_id": model_id}
        if config:
            resolved_config.update(config)

        return LLM(
            provider="google",
            model_id=model_id,
            temperature=temperature,
            tool_choice=tool_choice,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            base_url=base_url,
            config=resolved_config,
        )

    @staticmethod
    def sarvam(
        *,
        model_id: str = "sarvam-30b",
        config: Optional[Dict] = None,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        base_url: Optional[str] = None,
    ) -> "LLM":
        """
        Create an LLM instance configured for Sarvam AI.

        Args:
            model_id: Sarvam model identifier (default: "sarvam-30b")
            config: Optional extra config dict
            temperature: Controls randomness in responses (0.0 to 1.0)
            tool_choice: Tool calling mode ("auto", "required", "none")
            max_output_tokens: Maximum tokens in model responses
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            base_url: Custom inference gateway URL

        Returns:
            Configured LLM instance for Sarvam
        """
        resolved_config: Dict[str, Any] = {"model_id": model_id}
        if config:
            resolved_config.update(config)

        return LLM(
            provider="sarvamai",
            model_id=model_id,
            temperature=temperature,
            tool_choice=tool_choice,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            base_url=base_url,
            config=resolved_config,
        )

    @staticmethod
    def videosdk(
        *,
        model_id: str = "google.gemma-4-31b",
        config: Optional[Dict] = None,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        base_url: Optional[str] = None,
    ) -> "LLM":
        """
        Create an LLM instance configured for VideoSDK LLM.

        It is served through its OpenAI-compatible chat-completions
        surface, so it supports text (streaming and non-streaming) plus vision
        (text + image) requests. Image parts are sent as inline data URLs.

        Args:
            model_id: Model identifier (default: "google.gemma-4-31b")
            config: Optional extra config dict (merged on top of defaults)
            temperature: Controls randomness in responses (0.0 to 1.0)
            tool_choice: Tool calling mode ("auto", "required", "none")
            max_output_tokens: Maximum tokens in model responses
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            presence_penalty: Penalizes token presence (-2.0 to 2.0)
            frequency_penalty: Penalizes token frequency (-2.0 to 2.0)
            base_url: Custom inference gateway URL

        Returns:
            Configured LLM instance for VideoSDK LLM
        """
        resolved_config: Dict[str, Any] = {"model_id": model_id}
        if config:
            resolved_config.update(config)

        return LLM(
            provider="videosdk",
            model_id=model_id,
            temperature=temperature,
            tool_choice=tool_choice,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            base_url=base_url,
            config=resolved_config,
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

            # Build base request payload
            payload: Dict[str, Any] = {
                "model": self.model_id,
                "messages": formatted_messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": self.temperature,
            }

            # Common optional parameters
            if self.max_output_tokens:
                payload["max_tokens"] = self.max_output_tokens
            if self.top_p is not None:
                payload["top_p"] = self.top_p
            if self.presence_penalty is not None:
                payload["presence_penalty"] = self.presence_penalty
            if self.frequency_penalty is not None:
                payload["frequency_penalty"] = self.frequency_penalty

            # Google-only parameters
            if self.provider == "google":
                if self.top_k is not None:
                    payload["top_k"] = self.top_k

            # Sarvam-only parameters
            if self.provider == "sarvam":
                payload["wiki_grounding"] = self.wiki_grounding
                if self.reasoning_effort is not None:
                    payload["reasoning_effort"] = self.reasoning_effort
                if self.stop is not None:
                    payload["stop"] = self.stop

            # Add conversational graph response format
            if conversational_graph:
                payload["response_format"] = {
                    "type": "json_object",
                    "schema": conversational_graph.get_response_schema(),
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

        url = f"{self.base_url}/v1/chat/completions?provider={self.provider}"

        current_content = ""
        # Tool-call fragments accumulate here across SSE chunks, keyed by index.
        pending_tool_calls: Dict[int, Dict[str, Any]] = {}
        streaming_state = {
            "in_response": False,
            "response_start_index": -1,
            "yielded_content_length": 0,
        }

        try:
            logger.debug(
                f"[InferenceLLM] Making request to {self.base_url} "
                f"(provider={self.provider}, model_id={self.model_id})"
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
                                pending_tool_calls,
                                conversational_graph,
                            ):
                                if llm_response.content:
                                    current_content += llm_response.content
                                yield llm_response
                        except json.JSONDecodeError as e:
                            logger.warning(f"[InferenceLLM] Failed to parse chunk: {e}")
                            continue

                # Stream ended without an explicit tool-call finish_reason —
                # flush any tool calls still buffered.
                if pending_tool_calls and not self._cancelled:
                    for llm_response in self._finalize_tool_calls(pending_tool_calls):
                        yield llm_response
                    pending_tool_calls.clear()

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
        pending_tool_calls: Dict[int, Dict[str, Any]],
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
        # Usage metadata arrives in a dedicated final chunk (choices may be empty).
        usage = chunk.get("usage")
        if usage:
            usage_metadata: Dict[str, Any] = {
                "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
                "completion_tokens": usage.get("completion_tokens", 0) or 0,
                "total_tokens": usage.get("total_tokens", 0) or 0,
                "prompt_cached_tokens": 0,
                "reasoning_tokens": 0,
                "request_id": chunk.get("id"),
                "model": chunk.get("model", self.model_id),
            }
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict):
                usage_metadata["prompt_cached_tokens"] = (
                    prompt_details.get("cached_tokens", 0) or 0
                )
            completion_details = usage.get("completion_tokens_details") or {}
            if isinstance(completion_details, dict):
                usage_metadata["reasoning_tokens"] = (
                    completion_details.get("reasoning_tokens", 0) or 0
                )
            yield LLMResponse(
                content="",
                role=ChatRole.ASSISTANT,
                metadata={"usage": usage_metadata},
            )

        choices = chunk.get("choices", [])
        if not choices:
            return

        choice = choices[0]
        delta = choice.get("delta", {})

        # Accumulate streamed tool-call fragments, keyed by delta index. A
        # single tool call arrives across many chunks (name in the first,
        # arguments in fragments after) — JSON-parsing any one fragment alone
        # yields invalid/partial arguments.
        for tool_call in delta.get("tool_calls") or []:
            idx = tool_call.get("index", 0)
            function_data = tool_call.get("function") or {}
            if idx not in pending_tool_calls:
                pending_tool_calls[idx] = {
                    "id": tool_call.get("id") or "",
                    "name": function_data.get("name") or "",
                    "arguments": function_data.get("arguments") or "",
                }
            else:
                if function_data.get("name"):
                    pending_tool_calls[idx]["name"] += function_data["name"]
                if function_data.get("arguments"):
                    pending_tool_calls[idx]["arguments"] += function_data["arguments"]

        # Emit the accumulated tool calls once the model signals completion.
        if choice.get("finish_reason") == "tool_calls" and pending_tool_calls:
            for response in self._finalize_tool_calls(pending_tool_calls):
                yield response
            pending_tool_calls.clear()

        # Check for content
        content = delta.get("content", "")
        if content:
            if conversational_graph:
                full_content = current_content + content
                for (
                    content_chunk
                ) in conversational_graph.stream_conversational_graph_response(
                    full_content, streaming_state
                ):
                    yield LLMResponse(content=content_chunk, role=ChatRole.ASSISTANT)
            else:
                yield LLMResponse(content=content, role=ChatRole.ASSISTANT)

    def _finalize_tool_calls(
        self, pending_tool_calls: Dict[int, Dict[str, Any]]
    ) -> List[LLMResponse]:
        """Parse accumulated tool-call fragments into function_call responses.

        Arguments are JSON-parsed exactly once, when the call is complete. On a
        parse failure — or a value that isn't a JSON object — the arguments
        default to ``{}`` rather than a raw/partial string, so the tool is
        never invoked with a malformed argument mapping.
        """
        responses: List[LLMResponse] = []
        for tc in sorted(pending_tool_calls.values(), key=lambda t: t["id"]):
            if not tc["name"]:
                continue
            raw_args = tc["arguments"] or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                logger.error(
                    "[InferenceLLM] Failed to parse tool-call arguments for "
                    "'%s': %r",
                    tc["name"],
                    raw_args,
                )
                args = {}
            if not isinstance(args, dict):
                args = {}
            responses.append(
                LLMResponse(
                    content="",
                    role=ChatRole.ASSISTANT,
                    metadata={"function_call": {"name": tc["name"], "arguments": args}},
                )
            )
        return responses

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
                # build_gemini_schema returns a types.FunctionDeclaration
                # whose ``parameters`` field is itself a types.Schema
                # (both Pydantic-v2 objects). Use model_dump to flatten
                # the entire thing to plain dicts in one pass; the
                # gateway request body is JSON-serialized by aiohttp and
                # chokes on typed objects otherwise.
                gemini_schema = build_gemini_schema(tool)
                schema_dict = gemini_schema.model_dump(exclude_none=True)
                formatted_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": schema_dict.get("name", ""),
                            "description": schema_dict.get("description", ""),
                            "parameters": schema_dict.get("parameters", {}),
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

        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        await super().aclose()

        logger.info("[InferenceLLM] Closed successfully")

    # ==================== Properties ====================

    @property
    def label(self) -> str:
        """Get a descriptive label for this LLM instance."""
        return f"videosdk.inference.LLM.{self.provider}.{self.model_id}"
