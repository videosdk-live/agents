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
    ) -> None:
        """Initialize the CometAPI TTS plugin.

        Args:
            api_key (Optional[str], optional): CometAPI API key. Defaults to None.
            model (str): The model to use for the TTS plugin. Defaults to "tts-1".
            voice (str): The voice to use for the TTS plugin. Defaults to "alloy".
            speed (float): The speed to use for the TTS plugin. Defaults to 1.0.
            response_format (str): The response format to use for the TTS plugin. Defaults to "pcm".
        """
        super().__init__(
            api_key=api_key or os.getenv("COMETAPI_API_KEY"),
            model=model,
            voice=voice,
            speed=speed,
            base_url="https://api.cometapi.com/v1/",
            response_format=response_format,
        )