from __future__ import annotations

import base64
import os
import json
from typing import Any, AsyncIterator, List, Union

import httpx
from google.genai import Client, types
from google.genai.errors import APIError, ClientError, ServerError
import logging

logger = logging.getLogger(__name__)

from videosdk.agents import (
    LLM,
    LLMResponse,
    ChatContext,
    ChatRole,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    ToolChoice,
    FunctionTool,
    is_function_tool,
    build_gemini_schema,
    ChatContent,
    ImageContent,
)


class GoogleLLM(LLM):
    
    def __init__(
        self,
        *,
        model: str = "gemini-2.0-flash-001",
        api_key: str | None = None,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
    ) -> None:
        super().__init__()
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key must be provided either through api_key parameter or GOOGLE_API_KEY environment variable")
        
        self.model = model
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self._cancelled = False
        
        self._client = Client(api_key=self.api_key)

    async def chat(
        self,
        messages: ChatContext,
        tools: list[FunctionTool] | None = None,
        **kwargs: Any
    ) -> AsyncIterator[LLMResponse]:
        """
        Implement chat functionality using Google's Gemini API
        
        Args:
            messages: ChatContext containing conversation history
            tools: Optional list of function tools available to the model
            **kwargs: Additional arguments passed to the Google API
            
        Yields:
            LLMResponse objects containing the model's responses
        """
        self._cancelled = False
        
        try:
            (
                contents,
                system_instruction,
            ) = await self._convert_messages_to_contents_async(messages)
            config_params = {
                "temperature": self.temperature,
                **kwargs
            }
            if system_instruction:
                config_params["system_instruction"] = [types.Part(text=system_instruction)]
            
            if self.max_output_tokens is not None:
                config_params["max_output_tokens"] = self.max_output_tokens
            if self.top_p is not None:
                config_params["top_p"] = self.top_p
            if self.top_k is not None:
                config_params["top_k"] = self.top_k
            if self.presence_penalty is not None:
                config_params["presence_penalty"] = self.presence_penalty
            if self.frequency_penalty is not None:
                config_params["frequency_penalty"] = self.frequency_penalty

            if tools:
                function_declarations = []
                for tool in tools:
                    if is_function_tool(tool):
                        try:
                            gemini_tool = build_gemini_schema(tool)
                            function_declarations.append(gemini_tool)
                        except Exception as e:
                            logger.error(f"Failed to format tool {tool}: {e}")
                            continue
                
                if function_declarations:
                    config_params["tools"] = [types.Tool(function_declarations=function_declarations)]
                    
                    if self.tool_choice == "required":
                        config_params["tool_config"] = types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode=types.FunctionCallingConfigMode.ANY
                            )
                        )
                    elif self.tool_choice == "auto":
                        config_params["tool_config"] = types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode=types.FunctionCallingConfigMode.AUTO
                            )
                        )
                    elif self.tool_choice == "none":
                        config_params["tool_config"] = types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode=types.FunctionCallingConfigMode.NONE
                            )
                        )

            config = types.GenerateContentConfig(**config_params)

            response_stream = await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )

            current_content = ""
            current_function_calls = []

            async for response in response_stream:
                if self._cancelled:
                    break
                    
                if response.prompt_feedback:
                    error_msg = f"Prompt feedback error: {response.prompt_feedback}"
                    self.emit("error", Exception(error_msg))
                    raise Exception(error_msg)

                if not response.candidates or not response.candidates[0].content:
                    continue

                candidate = response.candidates[0]
                if not candidate.content.parts:
                    continue

                for part in candidate.content.parts:
                    if part.function_call:
                        function_call = {
                            "name": part.function_call.name,
                            "arguments": dict(part.function_call.args)
                        }
                        current_function_calls.append(function_call)
                        
                        yield LLMResponse(
                            content="",
                            role=ChatRole.ASSISTANT,
                            metadata={"function_call": function_call}
                        )
                    elif part.text:
                        current_content = part.text
                        yield LLMResponse(
                            content=current_content,
                            role=ChatRole.ASSISTANT
                        )

        except (ClientError, ServerError, APIError) as e:
            if not self._cancelled:
                error_msg = f"Google API error: {e}"
                self.emit("error", Exception(error_msg))
            raise Exception(error_msg) from e
        except Exception as e:
            if not self._cancelled:
                self.emit("error", e)
            raise

    async def cancel_current_generation(self) -> None:
        self._cancelled = True

    async def _convert_messages_to_contents_async(
        self, messages: ChatContext
    ) -> tuple[list[types.Content], str | None]:
        """Convert ChatContext to Google Content format"""

        async def _format_content_parts_async(
            content: Union[str, List[ChatContent]]
        ) -> List[types.Part]:
            if isinstance(content, str):
                return [types.Part(text=content)]

            if len(content) == 1 and isinstance(content[0], str):
                return [types.Part(text=content[0])]

            formatted_parts = []
            for part in content:
                if isinstance(part, str):
                    formatted_parts.append(types.Part(text=part))
                elif isinstance(part, ImageContent):
                    data_url = part.to_data_url()
                    if data_url.startswith("data:"):
                        header, b64_data = data_url.split(",", 1)
                        media_type = header.split(";")[0].split(":")[1]
                        image_bytes = base64.b64decode(b64_data)

                        formatted_parts.append(
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type=media_type, data=image_bytes
                                )
                            )
                        )
                    else: # Fetch image from URL
                        async with httpx.AsyncClient() as client:
                            try:
                                response = await client.get(data_url)
                                response.raise_for_status()
                                image_bytes = response.content
                                media_type = response.headers.get(
                                    "Content-Type", "image/jpeg"
                                )
                                formatted_parts.append(
                                    types.Part(
                                        inline_data=types.Blob(
                                            mime_type=media_type, data=image_bytes
                                        )
                                    )
                                )
                            except httpx.HTTPStatusError as e:
                                logger.error(f"Failed to fetch image from URL {data_url}: {e}")
                                continue 

            return formatted_parts

        contents = []
        system_instruction = None

        for item in messages.items:
            if isinstance(item, ChatMessage):
                if item.role == ChatRole.SYSTEM:
                    if isinstance(item.content, list):
                        system_instruction = next(
                            (str(p) for p in item.content if isinstance(p, str)), ""
                        )
                    else:
                        system_instruction = str(item.content)
                    continue
                elif item.role == ChatRole.USER:
                    parts = await _format_content_parts_async(item.content)
                    contents.append(types.Content(role="user", parts=parts))
                elif item.role == ChatRole.ASSISTANT:
                    parts = await _format_content_parts_async(item.content)
                    contents.append(types.Content(role="model", parts=parts))
            elif isinstance(item, FunctionCall):
                function_call = types.FunctionCall(
                    name=item.name,
                    args=(
                        json.loads(item.arguments)
                        if isinstance(item.arguments, str)
                        else item.arguments
                    ),
                )
                contents.append(
                    types.Content(role="model", parts=[types.Part(function_call=function_call)])
                )
            elif isinstance(item, FunctionCallOutput):
                function_response = types.FunctionResponse(
                    name=item.name, response={"output": item.output}
                )
                contents.append(
                    types.Content(
                        role="user", parts=[types.Part(function_response=function_response)]
                    )
                )

        return contents, system_instruction

    async def aclose(self) -> None:
        await self.cancel_current_generation()
