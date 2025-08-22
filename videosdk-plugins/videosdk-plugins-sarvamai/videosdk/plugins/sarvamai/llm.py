from __future__ import annotations

import os
import json
from typing import Any, AsyncIterator, List, Union
import traceback

import httpx
from videosdk.agents import (
    LLM, LLMResponse, ChatContext, ChatRole, ChatMessage, 
    ToolChoice, FunctionTool, 
    ChatContent,
)

SARVAM_CHAT_COMPLETION_URL = "https://api.sarvam.ai/v1/chat/completions" 
DEFAULT_MODEL = "sarvam-m" 

class SarvamAILLM(LLM):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_completion_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key or os.getenv("SARVAMAI_API_KEY")
        if not self.api_key:
            raise ValueError("Sarvam AI API key must be provided either through api_key parameter or SARVAMAI_API_KEY environment variable")
        
        self.model = model
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.max_completion_tokens = max_completion_tokens
        self._cancelled = False
        
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=30.0, write=5.0, pool=5.0),
            follow_redirects=True,
        )

    async def chat(
        self,
        messages: ChatContext,
        tools: list[FunctionTool] | None = None,
        **kwargs: Any
    ) -> AsyncIterator[LLMResponse]:
        self._cancelled = False
        
        def _extract_text_content(content: Union[str, List[ChatContent]]) -> str:
            if isinstance(content, str):
                return content
            text_parts = [part for part in content if isinstance(part, str)]
            return "\n".join(text_parts)

        system_prompt = None
        message_items = list(messages.items)
        if (
            message_items
            and isinstance(message_items[0], ChatMessage)
            and message_items[0].role == ChatRole.SYSTEM
        ):
            system_prompt = {
                "role": "system",
                "content": _extract_text_content(message_items.pop(0).content),
            }

        cleaned_messages = []
        last_role = None
        for msg in message_items:
            if not isinstance(msg, ChatMessage):
                continue

            current_role_str = msg.role.value
            
            if not cleaned_messages and current_role_str == 'assistant':
                continue

            text_content = _extract_text_content(msg.content)
            if not text_content.strip():
                continue

            if last_role == 'user' and current_role_str == 'user':
                cleaned_messages[-1]['content'] += ' ' + text_content
                continue
            
            if last_role == current_role_str:
                cleaned_messages.pop()

            cleaned_messages.append({"role": current_role_str, "content": text_content})
            last_role = current_role_str

        final_messages = [system_prompt] + cleaned_messages if system_prompt else cleaned_messages
        
        try:
            payload = {
                "model": self.model,
                "messages": final_messages,
                "temperature": self.temperature,
                "stream": True,
            }

            if self.max_completion_tokens:
                payload['max_tokens'] = self.max_completion_tokens
            
            payload.update(kwargs)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            async with self._client.stream("POST", SARVAM_CHAT_COMPLETION_URL, json=payload, headers=headers) as response:
                response.raise_for_status()
                
                current_content = ""
                async for line in response.aiter_lines():
                    if self._cancelled:
                        break
                        
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if not data_str:
                        continue
                    if data_str == "[DONE]":
                        break
                    
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta and delta["content"] is not None:
                        content_chunk = delta["content"]
                        current_content += content_chunk
                        yield LLMResponse(content=current_content, role=ChatRole.ASSISTANT)

        except httpx.HTTPStatusError as e:
            if not self._cancelled:
                error_message = f"Sarvam AI API error: {e.response.status_code}"
                try:
                    error_body = await e.response.aread()
                    error_text = error_body.decode()
                    error_message += f" - {error_text}"
                except Exception:
                    pass
                self.emit("error", Exception(error_message))
            raise
        except Exception as e:
            if not self._cancelled:
                traceback.print_exc()
                self.emit("error", e)
            raise

    async def cancel_current_generation(self) -> None:
        self._cancelled = True

    async def aclose(self) -> None:
        await self.cancel_current_generation()
        if self._client:
            await self._client.aclose()
