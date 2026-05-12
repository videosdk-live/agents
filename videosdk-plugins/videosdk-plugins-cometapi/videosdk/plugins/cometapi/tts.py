from __future__ import annotations

import os
from typing import Any, AsyncIterator, Optional
import httpx
import openai
from videosdk.agents import TTS, segment_text
from videosdk.plugins.openai.tts import OpenAITTS # Re-use the existing OpenAI TTS

class CometAPITTS(OpenAITTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "tts-1",
        voice: str = "alloy",
        speed: float = 1.0,
        response_format: str = "pcm",
        chunked_synthesis: bool = False,
    ) -> None:
        """Initialize the CometAPI TTS plugin.

        CometAPI exposes an OpenAI-compatible TTS endpoint, so this class is a
        thin wrapper that swaps the base URL and inherits all of
        ``OpenAITTS``'s behavior — including ``FlushMarker`` handling,
        ``prewarm()``, and the ``chunked_synthesis`` accumulation toggle.

        Args:
            api_key (Optional[str], optional): CometAPI API key. Defaults to None.
            model (str): The model to use for the TTS plugin. Defaults to "tts-1".
            voice (str): The voice to use for the TTS plugin. Defaults to "alloy".
            speed (float): The speed to use for the TTS plugin. Defaults to 1.0.
            response_format (str): The response format to use for the TTS plugin. Defaults to "pcm".
            chunked_synthesis (bool): Forwarded to :class:`OpenAITTS`. When ``True``,
                each ``FlushMarker`` boundary triggers a separate POST. Defaults to
                ``False`` (single POST per utterance).
        """
        super().__init__(
            api_key=api_key or os.getenv("COMETAPI_API_KEY"),
            model=model,
            voice=voice,
            speed=speed,
            base_url="https://api.cometapi.com/v1/",
            response_format=response_format,
            chunked_synthesis=chunked_synthesis,
        )