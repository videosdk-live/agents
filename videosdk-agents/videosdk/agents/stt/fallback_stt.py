import asyncio
from typing import List, Any, Optional
from .stt import STT
from ..fallback_base import FallbackBase
from ..event_bus import global_event_emitter

class FallbackSTT(STT, FallbackBase):
    """STT wrapper that automatically fails over to backup providers on errors, latency degradation, and attempts recovery of higher-priority ones."""
    def __init__(
        self,
        providers: List[STT],
        temporary_disable_sec: float = 60.0,
        permanent_disable_after_attempts: int = 3,
        latency_threshold_ms: Optional[float] = None,
        consecutive_latency_hits: int = 3,
    ):
        STT.__init__(self)
        FallbackBase.__init__(
            self,
            providers,
            "STT",
            temporary_disable_sec=temporary_disable_sec,
            permanent_disable_after_attempts=permanent_disable_after_attempts,
            latency_threshold_ms=latency_threshold_ms,
            consecutive_latency_hits=consecutive_latency_hits,
        )

        self._transcript_callback = None
        self._setup_event_listeners()
        self._setup_latency_listener()

    def _setup_event_listeners(self):
        """Attach error listener to the currently active provider."""
        self.active_provider.on("error", self._on_provider_error)

    def _setup_latency_listener(self):
        if self.latency_threshold_ms is None:
            return
        global_event_emitter.on("COMPONENT_METRIC", self._on_component_metric)

    def _on_component_metric(self, event: dict):
        if event.get("component") != "stt":
            return
        metrics = event.get("metrics") or {}
        latency = metrics.get("stt_latency")
        if latency is None:
            return
        asyncio.create_task(self._record_latency(float(latency)))

    def _on_provider_error(self, error_msg):
        """Handle async errors (e.g. WebSocket disconnects)"""
        failed_p = self.active_provider
        asyncio.create_task(self._handle_async_error(str(error_msg), failed_p))

    async def _handle_async_error(self, error_msg: str, failed_provider: Any):
        """Async wrapper to handle switching logic"""
        switched = await self._switch_provider(f"Async Error: {error_msg}", failed_provider=failed_provider)
        self.emit("error", error_msg)

    async def _switch_provider(self, reason: str, failed_provider: Any = None):
        """Override switch to handle STT specific setup"""
        provider_to_cleanup = failed_provider if failed_provider else self.active_provider

        try:
            provider_to_cleanup.off("error", self._on_provider_error)
        except: pass

        active_before = self.active_provider
        switched = await super()._switch_provider(reason, failed_provider)
        active_after = self.active_provider
        
        if switched:
            if active_before != active_after:
                if self._transcript_callback:
                    self.active_provider.on_stt_transcript(self._transcript_callback)
                self.active_provider.on("error", self._on_provider_error)
            return True
        return False

    def on_stt_transcript(self, callback) -> None:
        """Capture the callback so we can re-apply it after switching."""
        self._transcript_callback = callback
        self.active_provider.on_stt_transcript(callback)

    async def process_audio(self, audio_frames: bytes, **kwargs) -> None:
        """
        Main entry point. If this fails, it's usually a connection error.
        We catch, switch, and retry immediately.
        """
        if self.check_recovery():
            if self._transcript_callback:
                self.active_provider.on_stt_transcript(self._transcript_callback)
            self.active_provider.on("error", self._on_provider_error)

        current_provider = self.active_provider
        try:
            await current_provider.process_audio(audio_frames, **kwargs)
        except Exception as e:
            switched = await self._switch_provider(str(e), failed_provider=current_provider)
            self.emit("error", str(e))
            if switched:
                await self.active_provider.process_audio(audio_frames, **kwargs)
            else:
                raise e

    async def aclose(self) -> None:
        """Close all providers."""
        if self.latency_threshold_ms is not None:
            try:
                global_event_emitter.off("COMPONENT_METRIC", self._on_component_metric)
            except Exception:
                pass
        for p in self.providers:
            await p.aclose()
        await super().aclose()
