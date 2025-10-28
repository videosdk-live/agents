from __future__ import annotations
from videosdk.plugins.openai.llm import OpenAILLM 
import os
class CometAPILLM(OpenAILLM):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_completion_tokens: int | None = None,
    ) -> None:
        """Initialize the CometAPI LLM plugin.

        Args:
            api_key (Optional[str], optional): CometAPI API key. Defaults to None.
            model (str): The model to use for the LLM plugin. Defaults to "gpt-4o-mini".
            temperature (float): The temperature to use for the LLM plugin. Defaults to 0.7.
            max_completion_tokens (Optional[int], optional): The maximum completion tokens to use for the LLM plugin. Defaults to None.
        """
        super().__init__(
            api_key=api_key or os.getenv("COMETAPI_API_KEY"),
            model=model,
            base_url="https://api.cometapi.com/v1/",
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )