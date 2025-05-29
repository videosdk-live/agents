from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List

import httpx
import openai
from videosdk.agents.llm.llm import LLM, LLMResponse
from videosdk.agents.llm.chat_context import ChatContext, ChatRole, ChatMessage
from videosdk.agents.utils import ToolChoice, FunctionTool, is_function_tool, build_openai_schema
from videosdk.agents.llm.function_handler import FunctionHandler

class OpenAILLM(LLM):
    
    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_completion_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either through api_key parameter or OPENAI_API_KEY environment variable")
        
        self.model = model
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.max_completion_tokens = max_completion_tokens
        
        self._client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url or None,
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

    async def chat(
        self,
        messages: ChatContext,
        tools: list[FunctionTool] | None = None,
        **kwargs: Any
    ) -> AsyncIterator[LLMResponse]:
        """
        Implement chat functionality using OpenAI's chat completion API
        
        Args:
            messages: ChatContext containing conversation history
            tools: Optional list of function tools available to the model
            **kwargs: Additional arguments passed to the OpenAI API
            
        Yields:
            LLMResponse objects containing the model's responses
        """
        completion_params = {
            "model": self.model,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content
                }
                for msg in messages.items
                if isinstance(msg, ChatMessage)
            ],
            "temperature": self.temperature,
            "stream": True,
            "max_tokens": self.max_completion_tokens,
            **kwargs
        }
        try:
            response_stream = await self._client.chat.completions.create(**completion_params)
            current_content = ""

            async for chunk in response_stream:
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                
                if delta.content is not None:
                    current_content += delta.content
                    yield LLMResponse(
                        content=current_content,
                        role=ChatRole.ASSISTANT
                    )

        except Exception as e:
            self.emit("error", e)
            raise

    async def aclose(self) -> None:
        """Cleanup resources by closing the HTTP client"""
        if self._client:
            await self._client.close() 