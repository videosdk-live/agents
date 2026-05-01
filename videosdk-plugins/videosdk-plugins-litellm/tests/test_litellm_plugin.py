"""Unit tests for the LiteLLM VideoSDK plugin."""

from __future__ import annotations

import os
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from videosdk.plugins.litellm import LiteLLM
from videosdk.plugins.litellm.llm import (
    _LiteLLMChat,
    _LiteLLMClientShim,
    _LiteLLMCompletions,
)


def _make_shim(**overrides: Any) -> _LiteLLMClientShim:
    defaults = {
        "api_key": None,
        "api_base": None,
        "api_version": None,
        "extra_headers": None,
        "drop_params": True,
        "extra_kwargs": None,
    }
    defaults.update(overrides)
    return _LiteLLMClientShim(**defaults)


def test_shim_exposes_chat_completions_create() -> None:
    shim = _make_shim()
    assert isinstance(shim.chat, _LiteLLMChat)
    assert isinstance(shim.chat.completions, _LiteLLMCompletions)
    assert callable(shim.chat.completions.create)


def test_shim_close_is_async_noop() -> None:
    import asyncio

    shim = _make_shim()
    asyncio.run(shim.close())  # must not raise


@pytest.mark.asyncio
async def test_shim_dispatches_to_litellm_acompletion(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def _fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return MagicMock()

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.acompletion = _fake_acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    shim = _make_shim(
        api_key="proxy-key",
        api_base="http://localhost:4000",
        extra_headers={"X-Trace": "abc"},
    )
    await shim.chat.completions.create(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )

    assert captured["model"] == "anthropic/claude-sonnet-4-6"
    assert captured["api_key"] == "proxy-key"
    assert captured["api_base"] == "http://localhost:4000"
    assert captured["extra_headers"] == {"X-Trace": "abc"}
    # drop_params=True is the default; matches lenient OpenAI-compat behavior
    assert captured["drop_params"] is True
    assert captured["stream"] is True


@pytest.mark.asyncio
async def test_shim_does_not_overwrite_explicit_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def _fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return MagicMock()

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.acompletion = _fake_acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    shim = _make_shim(api_key="from-shim", drop_params=False)
    await shim.chat.completions.create(
        model="openai/gpt-4o",
        messages=[],
        api_key="from-caller",
    )

    assert captured["api_key"] == "from-caller"
    assert captured["drop_params"] is False


@pytest.mark.asyncio
async def test_shim_extra_kwargs_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def _fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return MagicMock()

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.acompletion = _fake_acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    shim = _make_shim(extra_kwargs={"metadata": {"tag": "videosdk"}})
    await shim.chat.completions.create(model="openai/gpt-4o", messages=[])

    assert captured["metadata"] == {"tag": "videosdk"}


def test_litellm_llm_uses_shim_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)

    lm = LiteLLM(model="anthropic/claude-sonnet-4-6")
    # `_client` is set by OpenAILLM.__init__; we passed our shim to it
    assert isinstance(lm._client, _LiteLLMClientShim)
    assert lm.model == "anthropic/claude-sonnet-4-6"
    assert lm._owns_client is False  # parent flag means it won't try to close an HTTP client


def test_litellm_llm_picks_up_proxy_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "env-proxy-key")
    monkeypatch.setenv("LITELLM_API_BASE", "http://litellm-proxy:4000")

    lm = LiteLLM(model="openai/gpt-4o")
    assert isinstance(lm._client, _LiteLLMClientShim)
    assert lm._client.api_key == "env-proxy-key"
    assert lm._client.api_base == "http://litellm-proxy:4000"


def test_litellm_llm_explicit_args_win_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "env-key")
    monkeypatch.setenv("LITELLM_API_BASE", "http://env-base")

    lm = LiteLLM(
        model="openai/gpt-4o",
        api_key="explicit-key",
        api_base="http://explicit-base",
    )
    assert lm._client.api_key == "explicit-key"
    assert lm._client.api_base == "http://explicit-base"


def test_litellm_llm_no_openai_api_key_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAILLM normally requires OPENAI_API_KEY when no client is passed; the
    LiteLLM subclass passes a shim, so OPENAI_API_KEY is not needed."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # If this raised, OpenAILLM's API-key check would have triggered.
    LiteLLM(model="anthropic/claude-sonnet-4-6")
