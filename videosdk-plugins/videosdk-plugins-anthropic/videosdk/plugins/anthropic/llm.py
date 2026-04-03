from __future__ import annotations
import os
import json
from typing import Any, AsyncIterator, List, Literal, Union
import httpx
import anthropic
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
    ImageContent,
    ChatContent,
    ConversationalGraphResponse,
)


class AnthropicLLM(LLM):

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        base_url: str | None = None,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_tokens: int = 1024,
        top_k: int | None = None,
        top_p: float | None = None,
        caching: Literal["ephemeral"] | None = None,
        parallel_tool_calls: bool | None = None,
        thinking: dict | None = None,
        client: anthropic.AsyncClient | None = None,
        max_retries: int = 0,
    ) -> None:
        """Initialize the Anthropic LLM.

        Args:
            api_key: Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
            model: Claude model name. Defaults to "claude-sonnet-4-20250514".
            base_url: Override the default Anthropic API base URL.
            temperature: Sampling temperature. Defaults to 0.7.
            tool_choice: Controls which (if any) tool is called. Defaults to "auto".
            max_tokens: Maximum tokens in the response. Defaults to 1024.
            top_k: Top-K tokens considered during sampling.
            top_p: Nucleus sampling probability mass.
            caching: Set to ``"ephemeral"`` to enable Anthropic prompt caching.
                When enabled, ``cache_control`` markers are applied to the system
                prompt, tool schemas, and recent conversation turns, and cache token
                counts are surfaced in usage metadata.
            parallel_tool_calls: Allow (``True``) or disallow (``False``) the model
                from issuing multiple tool calls in a single turn. Maps to
                ``disable_parallel_tool_use`` in the Anthropic API.
            thinking: Extended-thinking configuration dict, e.g.
                ``{"type": "enabled", "budget_tokens": 1024}``. When set, the
                ``interleaved-thinking-2025-05-14`` beta path is used.
            client: Optional pre-built ``anthropic.AsyncClient`` instance. When
                provided, ``api_key``, ``base_url``, ``timeout``, and ``max_retries``
                are ignored. The caller retains ownership and is responsible for
                closing the client.
            max_retries: Number of automatic retries on transient errors. Defaults to 0.
        """
        super().__init__()

        self.model = model
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.caching = caching
        self.parallel_tool_calls = parallel_tool_calls
        self.thinking = thinking
        self._cancelled = False

        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Anthropic API key must be provided either through api_key parameter "
                    "or ANTHROPIC_API_KEY environment variable"
                )
            self._client = anthropic.AsyncClient(
                api_key=self.api_key,
                base_url=base_url or None,
                max_retries=max_retries,
                http_client=httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                    follow_redirects=True,
                    limits=httpx.Limits(
                        max_connections=1000,
                        max_keepalive_connections=100,
                        keepalive_expiry=120,
                    ),
                ),
            )

    async def chat(
        self,
        messages: ChatContext,
        tools: list[FunctionTool] | None = None,
        conversational_graph: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        """
        Implement chat functionality using Anthropic's Claude API.

        Args:
            messages: ChatContext containing conversation history.
            tools: Optional list of function tools available to the model.
            **kwargs: Additional arguments forwarded to the Anthropic API.

        Yields:
            LLMResponse objects containing the model's responses.
        """
        self._cancelled = False

        response_stream = None
        try:
            anthropic_messages, system_content = messages.to_anthropic_messages(
                caching=(self.caching is not None)
            )

            completion_params: dict = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": True,
            }

            if system_content:
                # system can be a plain string or a list of content blocks
                if self.caching == "ephemeral":
                    completion_params["system"] = [
                        {
                            "type": "text",
                            "text": system_content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                else:
                    completion_params["system"] = system_content

            if self.top_k is not None:
                completion_params["top_k"] = self.top_k
            if self.top_p is not None:
                completion_params["top_p"] = self.top_p

            formatted_tools: list[dict] = []
            if tools:
                seen_tool_names: set[str] = set()
                for tool in tools:
                    if not is_function_tool(tool):
                        continue
                    try:
                        openai_schema = build_openai_schema(tool)
                        tool_name = openai_schema["name"]
                        if tool_name in seen_tool_names:
                            continue
                        seen_tool_names.add(tool_name)
                        formatted_tools.append(
                            {
                                "name": tool_name,
                                "description": openai_schema["description"],
                                "input_schema": openai_schema["parameters"],
                            }
                        )
                    except Exception as e:
                        self.emit("error", f"Failed to format tool {tool}: {e}")
                        continue

                if formatted_tools:
                    # Apply cache_control to the last tool schema when caching is on
                    if self.caching == "ephemeral":
                        formatted_tools[-1]["cache_control"] = {"type": "ephemeral"}
                    completion_params["tools"] = formatted_tools

                    # Build tool_choice dict
                    tool_choice_dict: dict = {}
                    if isinstance(self.tool_choice, dict) and self.tool_choice.get("type") == "function":
                        # Specific tool by name: OpenAI-style → Anthropic-style
                        tool_choice_dict = {
                            "type": "tool",
                            "name": self.tool_choice["function"]["name"],
                        }
                    elif self.tool_choice == "required":
                        tool_choice_dict = {"type": "any"}
                    elif self.tool_choice == "auto":
                        tool_choice_dict = {"type": "auto"}
                    elif self.tool_choice == "none":
                        del completion_params["tools"]
                        tool_choice_dict = {}
                    else:
                        tool_choice_dict = {"type": "auto"}

                    if tool_choice_dict:
                        if self.parallel_tool_calls is not None:
                            tool_choice_dict["disable_parallel_tool_use"] = not self.parallel_tool_calls
                        completion_params["tool_choice"] = tool_choice_dict

            # Prompt caching: mark last user/assistant messages
            if self.caching == "ephemeral" and completion_params.get("messages"):
                msgs = completion_params["messages"]
                last_user_idx = last_asst_idx = -1
                for idx, m in enumerate(msgs):
                    if m["role"] == "user":
                        last_user_idx = idx
                    elif m["role"] == "assistant":
                        last_asst_idx = idx
                for idx in (last_user_idx, last_asst_idx):
                    if idx < 0:
                        continue
                    content = msgs[idx]["content"]
                    if isinstance(content, str):
                        msgs[idx]["content"] = [
                            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                        ]
                    elif isinstance(content, list) and content:
                        content[-1]["cache_control"] = {"type": "ephemeral"}

            completion_params.update(kwargs)

            # Build extra_headers
            extra_headers: dict = {}
            if conversational_graph:
                extra_headers["anthropic-beta"] = "structured-outputs-2025-11-13"
            if self.thinking:
                existing_betas = extra_headers.get("anthropic-beta", "")
                thinking_beta = "interleaved-thinking-2025-05-14"
                extra_headers["anthropic-beta"] = (
                    f"{existing_betas},{thinking_beta}" if existing_betas else thinking_beta
                )
                completion_params["thinking"] = self.thinking

            if extra_headers:
                completion_params["extra_headers"] = extra_headers

            # Use beta API path when extended thinking is active
            if self.thinking:
                response_stream = await self._client.beta.messages.create(**completion_params)
            else:
                response_stream = await self._client.messages.create(**completion_params)

            current_content = ""
            current_tool_call: dict | None = None
            current_tool_call_id: str | None = None
            current_tool_arguments = ""

            usage_metadata: dict = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_cached_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
            }

            streaming_state = {
                "in_response": False,
                "response_start_index": -1,
                "yielded_content_length": 0,
            }

            async for event in response_stream:
                if self._cancelled:
                    break

                if event.type == "message_start":
                    usage_metadata["prompt_tokens"] = event.message.usage.input_tokens
                    usage_metadata["prompt_cached_tokens"] = (
                        getattr(event.message.usage, "cache_read_input_tokens", 0) or 0
                    )
                    usage_metadata["cache_creation_tokens"] = (
                        getattr(event.message.usage, "cache_creation_input_tokens", 0) or 0
                    )
                    usage_metadata["cache_read_tokens"] = (
                        getattr(event.message.usage, "cache_read_input_tokens", 0) or 0
                    )
                    yield LLMResponse(
                        content="", role=ChatRole.ASSISTANT, metadata={"usage": usage_metadata}
                    )

                elif event.type == "message_delta":
                    if hasattr(event, "usage"):
                        usage_metadata["completion_tokens"] = event.usage.output_tokens
                        usage_metadata["total_tokens"] = (
                            usage_metadata["prompt_tokens"] + usage_metadata["completion_tokens"]
                        )
                        yield LLMResponse(
                            content="", role=ChatRole.ASSISTANT, metadata={"usage": usage_metadata}
                        )

                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool_call_id = event.content_block.id
                        current_tool_call = {
                            "name": event.content_block.name,
                            "arguments": "",
                        }
                        current_tool_arguments = ""

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        current_content += delta.text
                        if conversational_graph:
                            for content_chunk in conversational_graph.stream_conversational_graph_response(
                                current_content, streaming_state
                            ):
                                yield LLMResponse(
                                    content=content_chunk,
                                    role=ChatRole.ASSISTANT,
                                    metadata={"usage": usage_metadata},
                                )
                        else:
                            yield LLMResponse(
                                content=delta.text,
                                role=ChatRole.ASSISTANT,
                                metadata={"usage": usage_metadata},
                            )
                    elif delta.type == "input_json_delta":
                        if current_tool_call is not None:
                            current_tool_arguments += delta.partial_json

                elif event.type == "content_block_stop":
                    if current_tool_call is not None and current_tool_call_id is not None:
                        try:
                            parsed_args = (
                                json.loads(current_tool_arguments)
                                if current_tool_arguments
                                else {}
                            )
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
                                    "call_id": current_tool_call_id,
                                },
                                "usage": usage_metadata,
                            },
                        )
                        current_tool_call = None
                        current_tool_call_id = None
                        current_tool_arguments = ""

            if not self._cancelled:
                yield LLMResponse(
                    content="", role=ChatRole.ASSISTANT, metadata={"usage": usage_metadata}
                )

            if current_content and not self._cancelled and conversational_graph:
                try:
                    parsed_json = json.loads(current_content.strip())
                    yield LLMResponse(
                        content="",
                        role=ChatRole.ASSISTANT,
                        metadata={"graph_response": parsed_json, "usage": usage_metadata},
                    )
                except json.JSONDecodeError:
                    yield LLMResponse(
                        content=current_content,
                        role=ChatRole.ASSISTANT,
                        metadata={"usage": usage_metadata},
                    )

        except anthropic.APIError as e:
            if not self._cancelled:
                self.emit("error", e)
            raise
        except Exception as e:
            if not self._cancelled:
                self.emit("error", e)
            raise
        finally:
            if response_stream is not None:
                try:
                    await response_stream.close()
                except Exception:
                    pass

    async def cancel_current_generation(self) -> None:
        self._cancelled = True

    async def aclose(self) -> None:
        """Cleanup resources. Only closes the underlying HTTP client if this instance owns it."""
        await self.cancel_current_generation()
        if self._owns_client and self._client:
            await self._client.close()
        await super().aclose()
