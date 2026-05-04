from __future__ import annotations
import asyncio
import logging
import os
from typing import Any, AsyncIterator, List, Literal, Optional, Union
import json

import aiohttp
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

logger = logging.getLogger(__name__)

OPENAI_RESPONSES_WSS_URL = "wss://api.openai.com/v1/responses"


def _format_responses_content(
    content: Union[str, List[ChatContent], None], role: str
) -> list[dict]:
    """Format ChatMessage content into Responses API content parts."""
    text_type = "output_text" if role == "assistant" else "input_text"
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": text_type, "text": content}]
    parts: list[dict] = []
    for p in content:
        if p is None:
            continue
        if isinstance(p, str):
            parts.append({"type": text_type, "text": p})
        elif isinstance(p, ImageContent):
            entry = {"type": "input_image", "image_url": p.to_data_url()}
            if p.inference_detail != "auto":
                entry["detail"] = p.inference_detail
            parts.append(entry)
    return parts


def _chat_items_to_responses_input(items: list) -> list[dict]:
    """Convert ChatContext items into Responses API ``input`` items."""
    out: list[dict] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, ChatMessage):
            role = item.role.value
            content_parts = _format_responses_content(item.content, role)
            if not content_parts:
                continue
            out.append({"type": "message", "role": role, "content": content_parts})
        elif isinstance(item, FunctionCall):
            args = item.arguments
            if not isinstance(args, str):
                args = json.dumps(args)
            out.append(
                {
                    "type": "function_call",
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": args,
                }
            )
        elif isinstance(item, FunctionCallOutput):
            out.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": item.output,
                }
            )
    return out


class OpenAILLM(LLM):

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        organization: str | None = None,
        project: str | None = None,
        parallel_tool_calls: bool | None = None,
        timeout: httpx.Timeout | None = None,
        extra_headers: dict | None = None,
        extra_query: dict | None = None,
        extra_body: dict | None = None,
        client: openai.AsyncOpenAI | None = None,
        max_retries: int = 0,
        reasoning_effort: Literal["none", "low", "medium", "high"] | None = None,
        verbosity: Literal["low", "medium", "high"] | None = None,
        streaming: bool = False,
        store: bool = False,
        wss_url: str | None = None,
    ) -> None:
        """Initialize the OpenAI LLM plugin.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: Chat model name. Defaults to "gpt-4o-mini".
            base_url: Override the default OpenAI API base URL.
            temperature: Sampling temperature. Defaults to 0.7.
            tool_choice: Controls which (if any) tool is called. Defaults to "auto".
            max_completion_tokens: Maximum tokens in the completion.
            top_p: Nucleus sampling probability mass.
            frequency_penalty: Penalise repeated tokens by frequency.
            presence_penalty: Penalise tokens that have already appeared.
            seed: Seed for deterministic sampling.
            organization: OpenAI organisation ID.
            project: OpenAI project ID.
            parallel_tool_calls: Allow the model to call multiple tools in one turn.
            timeout: Custom httpx.Timeout for the underlying HTTP client.
            extra_headers: Additional HTTP headers forwarded to every API call.
            extra_query: Additional query-string parameters forwarded to every API call.
            extra_body: Additional JSON body fields forwarded to every API call.
            client: Optional pre-built ``openai.AsyncOpenAI`` instance to use instead of
                creating a new one. Useful for sharing a client across instances or for
                testing. When provided, ``api_key``, ``base_url``, ``organization``,
                ``project``, ``timeout``, and ``max_retries`` are ignored.
            max_retries: Number of automatic retries on transient errors. Defaults to 0.
            reasoning_effort: Controls reasoning depth for reasoning models.
                Supported values: "none", "low", "medium", "high". Defaults to None
                (uses the model's default). Only applied for reasoning / GPT-5 models.
            verbosity: Controls output verbosity for reasoning / GPT-5 models.
                Supported values: "low", "medium", "high". Defaults to None.
            streaming: When True, use OpenAI's WebSocket Responses API
                (``wss://api.openai.com/v1/responses``) instead of the standard HTTP
                chat completions endpoint. The connection is reused across turns and
                continues with ``previous_response_id`` for lower per-turn latency.
                Defaults to False (HTTP mode).
            store: Only used when ``streaming=True``. Controls whether responses are
                persisted server-side. Defaults to False (ZDR-friendly). With
                ``store=False`` and an unrecoverable cache miss the connection
                resends the full context.
            wss_url: Override the WebSocket Responses URL. Defaults to OpenAI's
                public endpoint.
        """
        super().__init__()

        self.model = model
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.parallel_tool_calls = parallel_tool_calls
        self.extra_headers = extra_headers
        self.extra_query = extra_query
        self.extra_body = extra_body
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self._cancelled = False

        self.streaming = streaming
        self.store = store
        self._wss_url = wss_url or OPENAI_RESPONSES_WSS_URL

        # Always remember the API key for WSS use (even if a client was passed in).
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # WSS state — created lazily on first use.
        self._ws_session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_lock = asyncio.Lock()
        self._previous_response_id: Optional[str] = None
        self._last_seen_items_count: int = 0
        self._prewarm_task: Optional[asyncio.Task] = None

        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key must be provided either through api_key parameter "
                    "or OPENAI_API_KEY environment variable"
                )
            _timeout = timeout or httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0)
            self._client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=base_url or None,
                organization=organization or os.getenv("OPENAI_ORG_ID"),
                project=project or os.getenv("OPENAI_PROJECT_ID"),
                max_retries=max_retries,
                http_client=httpx.AsyncClient(
                    timeout=_timeout,
                    follow_redirects=True,
                    limits=httpx.Limits(
                        max_connections=50,
                        max_keepalive_connections=50,
                        keepalive_expiry=120,
                    ),
                ),
            )

        if self.streaming and not self.api_key:
            raise ValueError(
                "streaming=True requires an OpenAI API key (api_key parameter or "
                "OPENAI_API_KEY env var). The pre-built client cannot be introspected for it."
            )

        # Eagerly open the WSS connection if a loop is already running so the
        # first chat() call doesn't pay the TLS + WS handshake cost.
        if self.streaming:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                self._prewarm_task = loop.create_task(self._prewarm_safely())

    async def _prewarm_safely(self) -> None:
        try:
            await self._ensure_ws()
        except Exception as e:
            logger.warning("OpenAI WSS prewarm failed (will retry on first chat): %s", e)

    async def prewarm(
        self,
        *,
        instructions: str | None = None,
        tools: list[FunctionTool] | None = None,
    ) -> None:
        """Eagerly establish the WSS connection (and optionally prime request state).

        Call this once after constructing ``OpenAILLM(streaming=True)`` to avoid
        paying the TLS + WebSocket handshake on the first ``chat()`` call. When
        ``instructions`` and/or ``tools`` are provided, also sends a warmup
        ``response.create`` with ``generate: false`` so the server pre-builds
        request state for the first real turn (per OpenAI's WSS docs). The
        returned response id is used as ``previous_response_id`` on the first
        real turn for further latency reduction.

        No-op when ``streaming=False``.
        """
        if not self.streaming:
            return

        if self._prewarm_task is not None and not self._prewarm_task.done():
            try:
                await self._prewarm_task
            except Exception:
                pass

        ws = await self._ensure_ws()

        if instructions is None and not tools:
            return

        warmup_input: list[dict] = []
        if instructions:
            warmup_input.append(
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": instructions}],
                }
            )

        payload = self._build_responses_payload(
            input_items=warmup_input,
            previous_response_id=None,
            tools=tools,
            conversational_graph=None,
            extra={"generate": False},
        )

        try:
            await ws.send_str(json.dumps(payload))
        except Exception as e:
            logger.warning("OpenAI WSS warmup send failed: %s", e)
            return

        try:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        break
                    continue
                try:
                    event = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "response.created":
                    self._previous_response_id = (
                        (event.get("response") or {}).get("id")
                    )
                elif etype == "response.completed":
                    resp = event.get("response") or {}
                    self._previous_response_id = (
                        resp.get("id") or self._previous_response_id
                    )
                    break
                elif etype == "error":
                    err = event.get("error") or {}
                    logger.warning("OpenAI WSS warmup error: %s", err.get("message") or err)
                    self._previous_response_id = None
                    break
        except Exception as e:
            logger.warning("OpenAI WSS warmup read failed: %s", e)
            self._previous_response_id = None

    def _is_reasoning_model(self) -> bool:
        """Return True if the configured model is a reasoning / GPT-5 family model
        that requires special parameter handling."""
        model_lower = self.model.lower()
        if model_lower.startswith(("o1", "o3", "o4")):
            return True
        if model_lower.startswith("gpt-5"):
            return True
        return False

    @staticmethod
    def azure(
        *,
        model: str = "gpt-4o-mini",
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        parallel_tool_calls: bool | None = None,
        timeout: httpx.Timeout | None = None,
        extra_headers: dict | None = None,
        extra_query: dict | None = None,
        extra_body: dict | None = None,
        client: openai.AsyncAzureOpenAI | None = None,
        max_retries: int = 0,
        reasoning_effort: Literal["none", "low", "medium", "high"] | None = "none",
        verbosity: Literal["low", "medium", "high"] | None = "low",
    ) -> "OpenAILLM":
        """
        Create a new instance of Azure OpenAI LLM.

        Automatically infers the following from environment variables when not provided:
        - ``api_key`` from ``AZURE_OPENAI_API_KEY``
        - ``organization`` from ``OPENAI_ORG_ID``
        - ``project`` from ``OPENAI_PROJECT_ID``
        - ``azure_ad_token`` from ``AZURE_OPENAI_AD_TOKEN``
        - ``api_version`` from ``OPENAI_API_VERSION``
        - ``azure_endpoint`` from ``AZURE_OPENAI_ENDPOINT``
        - ``azure_deployment`` from ``AZURE_OPENAI_DEPLOYMENT`` (falls back to ``model``)

        Pass ``client`` to supply a pre-built ``openai.AsyncAzureOpenAI`` instance.
        When ``client`` is provided, connection/credential params are ignored.
        """
        if client is not None:
            instance = OpenAILLM(
                model=model,
                temperature=temperature,
                tool_choice=tool_choice,
                max_completion_tokens=max_completion_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed,
                parallel_tool_calls=parallel_tool_calls,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                client=client,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
            )
            return instance

        azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = api_version or os.getenv("OPENAI_API_VERSION")
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        azure_ad_token = azure_ad_token or os.getenv("AZURE_OPENAI_AD_TOKEN")
        organization = organization or os.getenv("OPENAI_ORG_ID")
        project = project or os.getenv("OPENAI_PROJECT_ID")

        if not azure_deployment:
            azure_deployment = model

        if not azure_endpoint:
            raise ValueError(
                "Azure endpoint must be provided either through azure_endpoint parameter "
                "or AZURE_OPENAI_ENDPOINT environment variable"
            )

        if not api_key and not azure_ad_token:
            raise ValueError("Either API key or Azure AD token must be provided")

        _timeout = timeout or httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0)
        azure_client = openai.AsyncAzureOpenAI(
            max_retries=max_retries,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=_timeout,
        )

        instance = OpenAILLM(
            model=model,
            temperature=temperature,
            tool_choice=tool_choice,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            parallel_tool_calls=parallel_tool_calls,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            client=azure_client,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
        )
        return instance

    async def chat(
        self,
        messages: ChatContext,
        tools: list[FunctionTool] | None = None,
        conversational_graph: Any | None = None,
        **kwargs: Any
    ) -> AsyncIterator[LLMResponse]:
        """
        Stream chat completions. Routes between the existing HTTP path and the
        WSS Responses path based on the ``streaming`` flag.
        """
        self._cancelled = False
        if self.streaming:
            async for response in self._chat_websocket(
                messages, tools=tools, conversational_graph=conversational_graph, **kwargs
            ):
                yield response
        else:
            async for response in self._chat_http(
                messages, tools=tools, conversational_graph=conversational_graph, **kwargs
            ):
                yield response

    async def _chat_http(
        self,
        messages: ChatContext,
        tools: list[FunctionTool] | None = None,
        conversational_graph: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        """
        Implement chat functionality using OpenAI's chat completion API.

        Args:
            messages: ChatContext containing conversation history.
            tools: Optional list of function tools available to the model.
            **kwargs: Additional arguments forwarded to the OpenAI API.

        Yields:
            LLMResponse objects containing the model's responses.
        """
        is_reasoning = self._is_reasoning_model()

        openai_messages = messages.to_openai_messages(
            reasoning_model=is_reasoning
        )

        completion_params: dict = {
            "model": self.model,
            "messages": openai_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if is_reasoning:
            if self.max_completion_tokens is not None:
                completion_params["max_completion_tokens"] = self.max_completion_tokens
            if self.reasoning_effort is not None:
                completion_params["reasoning_effort"] = self.reasoning_effort
            if self.verbosity is not None:
                completion_params["text"] = {"format": {"type": "text"}, "verbosity": self.verbosity}
        else:
            completion_params["temperature"] = self.temperature
            if self.max_completion_tokens is not None:
                completion_params["max_completion_tokens"] = self.max_completion_tokens

            if self.top_p is not None:
                completion_params["top_p"] = self.top_p
            if self.frequency_penalty is not None:
                completion_params["frequency_penalty"] = self.frequency_penalty
            if self.presence_penalty is not None:
                completion_params["presence_penalty"] = self.presence_penalty

        if self.seed is not None:
            completion_params["seed"] = self.seed

        if conversational_graph:
            completion_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "conversational_graph_response",
                    "strict": True,
                    "schema": conversational_graph._get_graph_schema()
                }
            }

        # Modern tools API (replaces deprecated functions/function_call)
        if tools:
            formatted_tools = []
            for tool in tools:
                if not is_function_tool(tool):
                    continue
                try:
                    tool_schema = build_openai_schema(tool)
                    formatted_tools.append({"type": "function", "function": tool_schema})
                except Exception as e:
                    self.emit("error", f"Failed to format tool {tool}: {e}")
                    continue

            if formatted_tools:
                completion_params["tools"] = formatted_tools
                # tool_choice: "auto"|"required"|"none" or {"type":"function","function":{"name":"..."}}
                if isinstance(self.tool_choice, dict):
                    completion_params["tool_choice"] = self.tool_choice
                else:
                    completion_params["tool_choice"] = self.tool_choice
                if self.parallel_tool_calls is not None:
                    completion_params["parallel_tool_calls"] = self.parallel_tool_calls

        # Pass-through overrides from caller
        completion_params.update(kwargs)

        # Passthrough extra headers / query / body
        create_kwargs: dict = {}
        if self.extra_headers:
            create_kwargs["extra_headers"] = self.extra_headers
        if self.extra_query:
            create_kwargs["extra_query"] = self.extra_query
        if self.extra_body:
            create_kwargs["extra_body"] = self.extra_body

        response_stream = None
        try:
            response_stream = await self._client.chat.completions.create(
                **completion_params, **create_kwargs
            )
            current_content = ""
            # Accumulate streamed tool call fragments keyed by delta index
            pending_tool_calls: dict[int, dict] = {}
            streaming_state = {
                "in_response": False,
                "response_start_index": -1,
                "yielded_content_length": 0
            }

            usage_metadata: dict = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_cached_tokens": 0,
                "reasoning_tokens": 0,
                "request_id": None,
                "model": self.model,
            }

            async for chunk in response_stream:
                if self._cancelled:
                    break

                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    usage_metadata["prompt_tokens"] = chunk.usage.prompt_tokens or 0
                    usage_metadata["completion_tokens"] = chunk.usage.completion_tokens or 0
                    usage_metadata["total_tokens"] = chunk.usage.total_tokens or 0
                    usage_metadata["request_id"] = getattr(chunk, "id", None)
                    usage_metadata["model"] = getattr(chunk, "model", self.model)

                    if hasattr(chunk.usage, 'prompt_tokens_details') and chunk.usage.prompt_tokens_details:
                        usage_metadata["prompt_cached_tokens"] = getattr(
                            chunk.usage.prompt_tokens_details, 'cached_tokens', 0
                        ) or 0
                    if hasattr(chunk.usage, 'completion_tokens_details') and chunk.usage.completion_tokens_details:
                        usage_metadata["reasoning_tokens"] = getattr(
                            chunk.usage.completion_tokens_details, 'reasoning_tokens', 0
                        ) or 0

                    yield LLMResponse(content="", role=ChatRole.ASSISTANT, metadata={"usage": usage_metadata})

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                # Accumulate tool call fragments per index
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in pending_tool_calls:
                            pending_tool_calls[idx] = {
                                "id": tc.id or "",
                                "name": (tc.function.name or "") if tc.function else "",
                                "arguments": (tc.function.arguments or "") if tc.function else "",
                            }
                        else:
                            if tc.function:
                                if tc.function.name:
                                    pending_tool_calls[idx]["name"] += tc.function.name
                                if tc.function.arguments:
                                    pending_tool_calls[idx]["arguments"] += tc.function.arguments

                # Emit all accumulated tool calls once the model signals it is done
                if finish_reason == "tool_calls" and pending_tool_calls:
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
                                "function_call": {"name": tc_data["name"], "arguments": args, "id": tc_data["id"]},
                                "usage": usage_metadata,
                            }
                        )
                    pending_tool_calls = {}

                elif delta.content is not None:
                    current_content += delta.content
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
                            content=delta.content,
                            role=ChatRole.ASSISTANT,
                            metadata={"usage": usage_metadata},
                        )

            # Flush any tool calls not yet emitted (stream ended without explicit finish_reason)
            if pending_tool_calls and not self._cancelled:
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
                            "function_call": {"name": tc_data["name"], "arguments": args, "id": tc_data["id"]},
                            "usage": usage_metadata,
                        }
                    )

            if current_content and not self._cancelled and conversational_graph:
                try:
                    parsed_json = json.loads(current_content.strip())
                    yield LLMResponse(
                        content="",
                        role=ChatRole.ASSISTANT,
                        metadata={"usage": usage_metadata, "graph_response": parsed_json}
                    )
                except json.JSONDecodeError:
                    yield LLMResponse(
                        content=current_content,
                        role=ChatRole.ASSISTANT,
                        metadata={"usage": usage_metadata}
                    )

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

    # ------------------------------------------------------------------
    # WSS Responses API path
    # ------------------------------------------------------------------

    async def _ensure_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Ensure a live WebSocket connection to the Responses endpoint."""
        async with self._ws_lock:
            if self._ws is not None and not self._ws.closed:
                return self._ws

            if self._ws_session is None or self._ws_session.closed:
                self._ws_session = aiohttp.ClientSession()

            headers = {"Authorization": f"Bearer {self.api_key}"}
            if self.extra_headers:
                headers.update(self.extra_headers)

            self._ws = await self._ws_session.ws_connect(
                self._wss_url,
                headers=headers,
                autoping=True,
                heartbeat=30,
                autoclose=False,
                timeout=30,
            )
            # Fresh connection — chain state is invalid.
            self._previous_response_id = None
            self._last_seen_items_count = 0
            return self._ws

    async def _close_ws(self) -> None:
        async with self._ws_lock:
            if self._ws is not None:
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None
            self._previous_response_id = None
            self._last_seen_items_count = 0

    def _build_responses_payload(
        self,
        *,
        input_items: list[dict],
        previous_response_id: str | None,
        tools: list[FunctionTool] | None,
        conversational_graph: Any | None,
        extra: dict,
    ) -> dict:
        """Build a ``response.create`` event payload for the Responses API."""
        is_reasoning = self._is_reasoning_model()
        payload: dict = {
            "type": "response.create",
            "model": self.model,
            "store": self.store,
            "input": input_items,
        }
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id

        if is_reasoning:
            if self.max_completion_tokens is not None:
                payload["max_output_tokens"] = self.max_completion_tokens
            if self.reasoning_effort is not None and self.reasoning_effort != "none":
                payload["reasoning"] = {"effort": self.reasoning_effort}
            if self.verbosity is not None:
                payload.setdefault("text", {})["verbosity"] = self.verbosity
        else:
            payload["temperature"] = self.temperature
            if self.max_completion_tokens is not None:
                payload["max_output_tokens"] = self.max_completion_tokens
            if self.top_p is not None:
                payload["top_p"] = self.top_p

        if self.seed is not None:
            payload["seed"] = self.seed

        if conversational_graph:
            text_cfg = payload.setdefault("text", {})
            text_cfg["format"] = {
                "type": "json_schema",
                "name": "conversational_graph_response",
                "strict": True,
                "schema": conversational_graph._get_graph_schema(),
            }

        if tools:
            formatted_tools: list[dict] = []
            for tool in tools:
                if not is_function_tool(tool):
                    continue
                try:
                    schema = build_openai_schema(tool)
                    fn_tool = {
                        "type": "function",
                        "name": schema["name"],
                        "description": schema.get("description", ""),
                        "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
                    }
                    if schema.get("strict") is not None:
                        fn_tool["strict"] = schema["strict"]
                    formatted_tools.append(fn_tool)
                except Exception as e:
                    self.emit("error", f"Failed to format tool {tool}: {e}")
                    continue
            if formatted_tools:
                payload["tools"] = formatted_tools
                payload["tool_choice"] = self.tool_choice
                if self.parallel_tool_calls is not None:
                    payload["parallel_tool_calls"] = self.parallel_tool_calls

        if self.extra_body:
            payload.update(self.extra_body)
        if extra:
            payload.update(extra)
        return payload

    def _slice_incremental_items(self, all_items: list) -> list:
        """Filter items added since the last turn down to those the server
        does not already know about (i.e., not part of the previous response)."""
        new_items = all_items[self._last_seen_items_count:]
        return [
            item for item in new_items
            if not (
                isinstance(item, FunctionCall)
                or (
                    isinstance(item, ChatMessage)
                    and item.role == ChatRole.ASSISTANT
                )
            )
        ]

    async def _chat_websocket(
        self,
        messages: ChatContext,
        tools: list[FunctionTool] | None = None,
        conversational_graph: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        """Stream chat responses over the WSS Responses API."""

        all_items = list(messages.items)

        # Decide initial payload: incremental (chained) or full.
        can_chain = (
            self._previous_response_id is not None
            and self._ws is not None
            and not self._ws.closed
            and self._last_seen_items_count > 0
            and len(all_items) >= self._last_seen_items_count
        )

        if can_chain:
            send_objs = self._slice_incremental_items(all_items)
            input_items = _chat_items_to_responses_input(send_objs)
            previous_response_id = self._previous_response_id
        else:
            input_items = _chat_items_to_responses_input(all_items)
            previous_response_id = None

        payload = self._build_responses_payload(
            input_items=input_items,
            previous_response_id=previous_response_id,
            tools=tools,
            conversational_graph=conversational_graph,
            extra=kwargs,
        )

        fallback_done = False

        while True:
            if self._cancelled:
                return

            try:
                ws = await self._ensure_ws()
                logger.info(
                    "[openai-wss] sending request | chained=%s items=%d",
                    bool(payload.get("previous_response_id")),
                    len(input_items),
                )
                await ws.send_str(json.dumps(payload))
            except Exception as e:
                if fallback_done:
                    if not self._cancelled:
                        self.emit("error", e)
                    raise
                fallback_done = True
                await self._close_ws()
                payload["input"] = _chat_items_to_responses_input(all_items)
                payload.pop("previous_response_id", None)
                continue

            usage_metadata: dict = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_cached_tokens": 0,
                "reasoning_tokens": 0,
                "request_id": None,
                "model": self.model,
            }
            current_content = ""
            streaming_state = {
                "in_response": False,
                "response_start_index": -1,
                "yielded_content_length": 0,
            }
            # item_id -> {"call_id", "name", "arguments"}
            function_calls: dict[str, dict] = {}
            response_id_this_turn: Optional[str] = None
            need_retry = False
            completed = False

            try:
                async for ws_msg in ws:
                    if self._cancelled:
                        break

                    if ws_msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            event = json.loads(ws_msg.data)
                        except json.JSONDecodeError:
                            logger.debug("Skipping non-JSON WSS frame: %r", ws_msg.data)
                            continue

                        etype = event.get("type")

                        if etype == "error":
                            err = event.get("error") or {}
                            code = err.get("code")
                            recoverable = code in (
                                "previous_response_not_found",
                                "websocket_connection_limit_reached",
                            )
                            if recoverable and not fallback_done:
                                fallback_done = True
                                need_retry = True
                                if code == "websocket_connection_limit_reached":
                                    await self._close_ws()
                                else:
                                    self._previous_response_id = None
                                    self._last_seen_items_count = 0
                                payload["input"] = _chat_items_to_responses_input(all_items)
                                payload.pop("previous_response_id", None)
                                break
                            raise RuntimeError(
                                f"OpenAI WSS error: {err.get('message') or event}"
                            )

                        elif etype == "response.created":
                            response_id_this_turn = (event.get("response") or {}).get("id")
                            logger.info("[openai-wss] response.created received id=%s", response_id_this_turn)

                        elif etype == "response.output_text.delta":
                            delta = event.get("delta", "")
                            if not delta:
                                continue
                            current_content += delta
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
                                    content=delta,
                                    role=ChatRole.ASSISTANT,
                                    metadata={"usage": usage_metadata},
                                )

                        elif etype == "response.output_item.added":
                            item = event.get("item") or {}
                            if item.get("type") == "function_call":
                                iid = item.get("id", "")
                                function_calls[iid] = {
                                    "call_id": item.get("call_id", "") or "",
                                    "name": item.get("name", "") or "",
                                    "arguments": item.get("arguments", "") or "",
                                }

                        elif etype == "response.function_call_arguments.delta":
                            iid = event.get("item_id")
                            delta = event.get("delta", "")
                            if iid and iid in function_calls and delta:
                                function_calls[iid]["arguments"] += delta

                        elif etype == "response.output_item.done":
                            item = event.get("item") or {}
                            if item.get("type") == "function_call":
                                iid = item.get("id", "")
                                existing = function_calls.get(iid, {})
                                fc_entry = {
                                    "call_id": item.get("call_id") or existing.get("call_id", ""),
                                    "name": item.get("name") or existing.get("name", ""),
                                    "arguments": item.get("arguments") or existing.get("arguments", ""),
                                    "dispatched": existing.get("dispatched", False),
                                }
                                function_calls[iid] = fc_entry
                                if not fc_entry["dispatched"]:
                                    args_str = fc_entry.get("arguments") or ""
                                    try:
                                        args = json.loads(args_str) if args_str else {}
                                    except json.JSONDecodeError:
                                        self.emit(
                                            "error",
                                            f"Failed to parse tool call arguments: {args_str}",
                                        )
                                        args = {}
                                    fc_entry["dispatched"] = True
                                    yield LLMResponse(
                                        content="",
                                        role=ChatRole.ASSISTANT,
                                        metadata={
                                            "function_call": {
                                                "name": fc_entry.get("name", ""),
                                                "arguments": args,
                                                "id": fc_entry.get("call_id", ""),
                                            },
                                            "usage": usage_metadata,
                                        },
                                    )

                        elif etype == "response.completed":
                            completed = True
                            resp = event.get("response") or {}
                            response_id_this_turn = resp.get("id") or response_id_this_turn
                            usage = resp.get("usage") or {}
                            usage_metadata["prompt_tokens"] = (
                                usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                            )
                            usage_metadata["completion_tokens"] = (
                                usage.get("output_tokens") or usage.get("completion_tokens") or 0
                            )
                            usage_metadata["total_tokens"] = usage.get("total_tokens") or 0
                            usage_metadata["request_id"] = response_id_this_turn
                            usage_metadata["model"] = resp.get("model") or self.model
                            in_details = usage.get("input_tokens_details") or {}
                            usage_metadata["prompt_cached_tokens"] = (
                                in_details.get("cached_tokens") or 0
                            )
                            out_details = usage.get("output_tokens_details") or {}
                            usage_metadata["reasoning_tokens"] = (
                                out_details.get("reasoning_tokens") or 0
                            )

                            yield LLMResponse(
                                content="",
                                role=ChatRole.ASSISTANT,
                                metadata={"usage": usage_metadata},
                            )

                            for fc in function_calls.values():
                                if fc.get("dispatched"):
                                    continue
                                args_str = fc.get("arguments") or ""
                                try:
                                    args = json.loads(args_str) if args_str else {}
                                except json.JSONDecodeError:
                                    self.emit(
                                        "error",
                                        f"Failed to parse tool call arguments: {args_str}",
                                    )
                                    args = {}
                                fc["dispatched"] = True
                                yield LLMResponse(
                                    content="",
                                    role=ChatRole.ASSISTANT,
                                    metadata={
                                        "function_call": {
                                            "name": fc.get("name", ""),
                                            "arguments": args,
                                            "id": fc.get("call_id", ""),
                                        },
                                        "usage": usage_metadata,
                                    },
                                )

                            if current_content and conversational_graph:
                                try:
                                    parsed_json = json.loads(current_content.strip())
                                    yield LLMResponse(
                                        content="",
                                        role=ChatRole.ASSISTANT,
                                        metadata={
                                            "usage": usage_metadata,
                                            "graph_response": parsed_json,
                                        },
                                    )
                                except json.JSONDecodeError:
                                    yield LLMResponse(
                                        content=current_content,
                                        role=ChatRole.ASSISTANT,
                                        metadata={"usage": usage_metadata},
                                    )

                            # Update chain tracking only on success.
                            if response_id_this_turn:
                                self._previous_response_id = response_id_this_turn
                                self._last_seen_items_count = len(all_items)
                            break

                    elif ws_msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        await self._close_ws()
                        if not completed and not fallback_done:
                            fallback_done = True
                            need_retry = True
                            payload["input"] = _chat_items_to_responses_input(all_items)
                            payload.pop("previous_response_id", None)
                        break
            except Exception as e:
                if not self._cancelled:
                    self.emit("error", e)
                raise

            if need_retry and not self._cancelled:
                continue
            return

    async def cancel_current_generation(self) -> None:
        self._cancelled = True
        # Close the WS so leftover server events from the cancelled response
        # don't contaminate the next request. _close_ws also clears
        # _previous_response_id and _last_seen_items_count for us.
        await self._close_ws()
        self._last_seen_items_count = 0

    async def aclose(self) -> None:
        """Cleanup resources. Closes the underlying HTTP client (if owned) and any
        WSS connection / session that was opened for streaming mode."""
        await self.cancel_current_generation()
        if self._prewarm_task is not None and not self._prewarm_task.done():
            self._prewarm_task.cancel()
            try:
                await self._prewarm_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._close_ws()
        if self._ws_session is not None and not self._ws_session.closed:
            try:
                await self._ws_session.close()
            except Exception:
                pass
            self._ws_session = None
        if self._owns_client and self._client:
            await self._client.close()
        await super().aclose()