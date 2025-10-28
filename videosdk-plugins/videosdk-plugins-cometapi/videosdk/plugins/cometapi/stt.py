from __future__ import annotations

import os
from videosdk.plugins.openai.stt import OpenAISTT 
class CometAPISTT(OpenAISTT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "whisper-1",
        prompt: str | None = None,
        language: str = "en",
    ) -> None:
        """Initialize the CometAPI STT plugin.

        Args:
            api_key (Optional[str], optional): CometAPI API key. Defaults to None.
            model (str): The model to use for the STT plugin. Defaults to "whisper-1".
            prompt (Optional[str], optional): The prompt for the STT plugin. Defaults to None.
            language (str): The language to use for the STT plugin. Defaults to "en".
        """
        super().__init__(
            api_key=api_key or os.getenv("COMETAPI_API_KEY"),
            model=model,
            base_url="https://api.cometapi.com/v1/",
            prompt=prompt,
            language=language,
            enable_streaming=False,
        )