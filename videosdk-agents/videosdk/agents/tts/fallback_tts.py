import logging
import asyncio
from typing import List, Any
from .tts import TTS
from ..fallback_base import FallbackBase

logger = logging.getLogger(__name__)

class FallbackTTS(TTS, FallbackBase):
    def __init__(self, providers: List[TTS], temporary_disable_sec: float = 60.0, permanent_disable_after_attempts: int = 3):
        TTS.__init__(
            self,
            sample_rate=providers[0].sample_rate, 
            num_channels=providers[0].num_channels
        )
        FallbackBase.__init__(self, providers, "TTS", temporary_disable_sec=temporary_disable_sec, permanent_disable_after_attempts=permanent_disable_after_attempts)
        self._initializing = False
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
                if hasattr(self, "loop") and self.loop and hasattr(self, "audio_track") and self.audio_track:
                    self._propagate_settings(self.active_provider)
            return True
        return False
    
    def _propagate_settings(self, provider):
        """Helper to set loop/audio_track on a provider."""
        try:
            name = "loop"
            value = self.loop
            if hasattr(provider, f"_set_{name}"):
                getattr(provider, f"_set_{name}")(value)
            else:
                setattr(provider, name, value)
                
            name = "audio_track"
            value = self.audio_track
            if hasattr(provider, f"_set_{name}"):
                getattr(provider, f"_set_{name}")(value)
            else:
                setattr(provider, name, value)
        except Exception as e:
            logger.warning(f"[TTS] Failed to propagate settings to {provider.label}: {e}")

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Intercept attribute assignments to propagate loop and audio_track to all providers.
        This allows FallbackTTS to work without CascadingPipeline needing to know about it.
        """
        object.__setattr__(self, name, value)
        
        if name in ("loop", "audio_track") and hasattr(self, "providers") and not getattr(self, "_initializing", False):
            logger.info(f"[TTS] FallbackTTS: {name} was set to {value}, propagating to all providers")
            for provider in self.providers:
                try:
                    if hasattr(provider, f"_set_{name}"):
                        getattr(provider, f"_set_{name}")(value)
                    else:
                        setattr(provider, name, value)
                    logger.info(f"[TTS] Set {name} on provider {provider.label}")
                except Exception as e:
                    logger.warning(f"[TTS] Failed to set {name} on provider {provider.label}: {e}")

    def _set_loop_and_audio_track(self, loop, audio_track):
        """
        Optional method for explicit setup (for compatibility).
        Setting these attributes will trigger __setattr__ which propagates to all providers.
        """
        logger.info(f"[TTS] _set_loop_and_audio_track called on FallbackTTS. loop={loop}, audio_track={audio_track}")
        self.loop = loop  
        self.audio_track = audio_track  

    async def synthesize(self, text, **kwargs) -> None:
        """
        Try active provider. If exception, switch and retry with same text.
        Checks for recovery of primary providers before starting.
        """
        if self.check_recovery():
             if hasattr(self, "loop") and self.loop and hasattr(self, "audio_track") and self.audio_track:
                 self._propagate_settings(self.active_provider)

        while True:
            current_provider = self.active_provider
            try:
                logger.info(f"[TTS] Attempting synthesis with {current_provider.label}")
                await current_provider.synthesize(text, **kwargs)
                logger.info(f"[TTS] Synthesis successful with {current_provider.label}")
                return
            except Exception as e:
                logger.error(f"[TTS] Synthesis failed with {current_provider.label}: {e}")
                switched = await self._switch_provider(str(e), failed_provider=current_provider)
                if not switched:
                    logger.error(f"[TTS] All providers exhausted. Raising error.")
                    raise e
                logger.info(f"[TTS] Retrying with new provider: {self.active_provider.label}")

    async def interrupt(self):
        if self.active_provider:
            await self.active_provider.interrupt()

    async def aclose(self):
        for p in self.providers:
            await p.aclose()
        await super().aclose()
