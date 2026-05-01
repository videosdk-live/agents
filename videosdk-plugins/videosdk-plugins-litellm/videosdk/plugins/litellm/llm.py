"""VideoSDK LLM plugin that routes through the LiteLLM SDK.

Subclasses `videosdk.plugins.openai.OpenAILLM` and swaps only the underlying
HTTP client for an `AsyncOpenAI`-shaped facade over `litellm.acompletion`.
LiteLLM normalizes every backing's response to OpenAI's chat-completions
shape, so all of the parent's streaming, tool-call accumulation, and usage
parsing logic inherits unchanged.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from videosdk.agents import ToolChoice
from videosdk.plugins.openai import OpenAILLM


class _LiteLLMCompletions:
    """Mimics `client.chat.completions` by dispatching to litellm.acompletion."""

    def __init__(self, parent: "_LiteLLMClientShim") -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> Any:
        import litellm

        merged: dict[str, Any] = dict(kwargs)
        if self._parent.api_key and "api_key" not in merged:
            merged["api_key"] = self._parent.api_key
        if self._parent.api_base and "api_base" not in merged:
            merged["api_base"] = self._parent.api_base
        if self._parent.api_version and "api_version" not in merged:
            merged["api_version"] = self._parent.api_version
        if self._parent.extra_headers and "extra_headers" not in merged:
            merged["extra_headers"] = self._parent.extra_headers
        merged.setdefault("drop_params", self._parent.drop_params)
        for key, value in self._parent.extra_kwargs.items():
            merged.setdefault(key, value)
        return await litellm.acompletion(**merged)


class _LiteLLMChat:
    def __init__(self, parent: "_LiteLLMClientShim") -> None:
        self.completions = _LiteLLMCompletions(parent)


class _LiteLLMClientShim:
    """`AsyncOpenAI`-shaped facade used in place of `openai.AsyncOpenAI`.

    Implements only the surface `OpenAILLM` actually calls
    (`chat.completions.create` and `close()`). Anything else raises so the
    failure mode stays loud.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        api_base: str | None,
        api_version: str | None,
        extra_headers: dict[str, Any] | None,
        drop_params: bool,
        extra_kwargs: dict[str, Any] | None,
    ) -> None:
        self.api_key = api_key or None
        self.api_base = api_base or None
        self.api_version = api_version or None
        self.extra_headers = extra_headers or None
        self.drop_params = drop_params
        self.extra_kwargs = dict(extra_kwargs or {})
        self.chat = _LiteLLMChat(self)

    async def close(self) -> None:
        # No persistent client to close; LiteLLM manages its own connections.
        return None


class LiteLLM(OpenAILLM):
    """LLM plugin routing through the LiteLLM SDK.

    Reuses every streaming, tool-call, structured-output, and usage-tracking
    path from `OpenAILLM` (since LiteLLM normalizes responses to OpenAI
    shape) and only swaps the underlying transport.

    Two deployment modes:

    1. **Embedded SDK (default)** — model spec like
       ``"anthropic/claude-sonnet-4-6"`` resolves backing-provider
       credentials from each provider's standard env var
       (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``AWS_*``, ...).
    2. **Proxy** — set ``api_base`` (and optionally ``api_key``) to a
       LiteLLM proxy server URL to route every call through it.

    Example::

        from videosdk import Pipeline
        from videosdk.plugins.litellm import LiteLLM

        pipeline = Pipeline(
            stt=DeepgramSTT(),
            llm=LiteLLM(model="anthropic/claude-sonnet-4-6"),
            tts=ElevenLabsTTS(),
            ...
        )
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        parallel_tool_calls: bool | None = None,
        extra_headers: dict[str, Any] | None = None,
        extra_query: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        drop_params: bool = True,
        extra_kwargs: dict[str, Any] | None = None,
        reasoning_effort: Literal["none", "low", "medium", "high"] | None = None,
        verbosity: Literal["low", "medium", "high"] | None = None,
    ) -> None:
        """Initialize the LiteLLM LLM plugin.

        Args:
            model: LiteLLM model spec, e.g. ``"anthropic/claude-sonnet-4-6"``,
                ``"openai/gpt-4o"``,
                ``"bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"``.
            api_key: Optional explicit API key. When unset, LiteLLM resolves
                from each backing's standard env var. Falls back to
                ``LITELLM_API_KEY`` for proxy mode.
            api_base: Optional base URL. Set to a LiteLLM proxy address
                (e.g. ``http://localhost:4000``) for proxy mode. Falls back
                to ``LITELLM_API_BASE``.
            api_version: Optional API version (Azure-style endpoints).
            temperature: Sampling temperature. Defaults to 0.7.
            tool_choice: Tool selection strategy. Defaults to ``"auto"``.
            max_completion_tokens: Maximum tokens in the completion.
            top_p: Nucleus sampling probability mass.
            frequency_penalty: Penalize repeated tokens by frequency.
            presence_penalty: Penalize tokens that have already appeared.
            seed: Seed for deterministic sampling.
            parallel_tool_calls: Allow the model to call multiple tools in
                one turn.
            extra_headers: HTTP headers forwarded as ``extra_headers`` to
                ``litellm.acompletion``.
            extra_query: Currently unused; kept for parity with OpenAILLM.
            extra_body: Currently unused; kept for parity with OpenAILLM.
            drop_params: When True (default), LiteLLM strips kwargs the
                chosen backing doesn't accept rather than raising. Mirrors
                the lenient behavior of OpenAI-compatible adapters.
            extra_kwargs: Additional kwargs forwarded verbatim to
                ``litellm.acompletion`` (e.g. ``metadata``, ``tags``,
                ``caching``).
            reasoning_effort: Reasoning depth for reasoning models.
            verbosity: Output verbosity for reasoning / GPT-5 models.
        """
        resolved_api_key = api_key or os.getenv("LITELLM_API_KEY")
        resolved_api_base = api_base or os.getenv("LITELLM_API_BASE")
        resolved_api_version = api_version or os.getenv("LITELLM_API_VERSION")

        client = _LiteLLMClientShim(
            api_key=resolved_api_key,
            api_base=resolved_api_base,
            api_version=resolved_api_version,
            extra_headers=extra_headers,
            drop_params=drop_params,
            extra_kwargs=extra_kwargs,
        )

        super().__init__(
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
            client=client,  # type: ignore[arg-type]
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
        )
