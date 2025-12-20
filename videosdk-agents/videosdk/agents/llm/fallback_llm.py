import asyncio
from typing import List, AsyncIterator, Any
from .llm import LLM, LLMResponse
from .chat_context import ChatContext
from ..fallback_base import FallbackBase

class FallbackLLM(LLM, FallbackBase):
    def __init__(self, providers: List[LLM], temporary_disable_sec: float = 60.0, permanent_disable_after_attempts: int = 3):
        LLM.__init__(self)
        FallbackBase.__init__(self, providers, "LLM", temporary_disable_sec=temporary_disable_sec, permanent_disable_after_attempts=permanent_disable_after_attempts)
        self._setup_event_listeners()

    def _setup_event_listeners(self):
        self.active_provider.on("error", self._on_provider_error)

    def _on_provider_error(self, error_msg):
        failed_p = self.active_provider
        asyncio.create_task(self._handle_async_error(str(error_msg), failed_p))

    async def _handle_async_error(self, error_msg: str, failed_provider: Any):
        switched = await self._switch_provider(f"Async Error: {error_msg}", failed_provider=failed_provider)
        if not switched:
            self.emit("error", error_msg)

    async def _switch_provider(self, reason: str, failed_provider: Any = None):
        provider_to_cleanup = failed_provider if failed_provider else self.active_provider
        try:
            provider_to_cleanup.off("error", self._on_provider_error)
        except: pass

        active_before = self.active_provider
        switched = await super()._switch_provider(reason, failed_provider)
        active_after = self.active_provider
        
        if switched:
            if active_before != active_after:
                self.active_provider.on("error", self._on_provider_error)
            return True
        return False

    async def chat(self, messages: ChatContext, **kwargs) -> AsyncIterator[LLMResponse]:
        """
        Attempts to chat with current provider. 
        Loops until one succeeds or all fail.
        Checks for recovery of primary providers before starting.
        """
        self.check_recovery()

        while True:
            current_provider = self.active_provider
            try:
                async for chunk in current_provider.chat(messages, **kwargs):
                    yield chunk
                return 
            except Exception as e:
                switched = await self._switch_provider(str(e), failed_provider=current_provider)
                if not switched:
                    raise e

    async def cancel_current_generation(self) -> None:
        await self.active_provider.cancel_current_generation()

    async def aclose(self) -> None:
        for p in self.providers:
            await p.aclose()
        await super().aclose()
