from __future__ import annotations
import os
from typing import Any, AsyncIterator, List, Literal, Union
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

        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
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
        Implement chat functionality using OpenAI's chat completion API.

        Args:
            messages: ChatContext containing conversation history.
            tools: Optional list of function tools available to the model.
            **kwargs: Additional arguments forwarded to the OpenAI API.

        Yields:
            LLMResponse objects containing the model's responses.
        """
        self._cancelled = False

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

    async def cancel_current_generation(self) -> None:
        self._cancelled = True

    async def aclose(self) -> None:
        """Cleanup resources. Only closes the underlying HTTP client if this instance owns it."""
        await self.cancel_current_generation()
        if self._owns_client and self._client:
            await self._client.close()
        await super().aclose()