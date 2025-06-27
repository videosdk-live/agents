from __future__ import annotations

import os
import json
from typing import Any, AsyncIterator, Optional

from google.genai import Client, types
from google.genai.errors import APIError, ClientError, ServerError

from videosdk.agents import LLM, LLMResponse, ChatContext, ChatRole, ChatMessage, FunctionCall, FunctionCallOutput, ToolChoice, FunctionTool, is_function_tool, build_gemini_schema


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
        try:
            contents, system_instruction = self._convert_messages_to_contents(messages)

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
                            print(f"Failed to format tool {tool}: {e}")
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
                        current_content += part.text
                        yield LLMResponse(
                            content=current_content,
                            role=ChatRole.ASSISTANT
                        )

        except (ClientError, ServerError, APIError) as e:
            error_msg = f"Google API error: {e}"
            self.emit("error", Exception(error_msg))
            raise Exception(error_msg) from e
        except Exception as e:
            self.emit("error", e)
            raise

    def _convert_messages_to_contents(self, messages: ChatContext) -> tuple[list[types.Content], str | None]:
        """Convert ChatContext to Google Content format"""
        contents = []
        system_instruction = None
        
        for item in messages.items:
            if isinstance(item, ChatMessage):
                if item.role == ChatRole.SYSTEM:
                    system_instruction = str(item.content)
                    continue
                elif item.role == ChatRole.USER:
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part(text=str(item.content))]
                    ))
                elif item.role == ChatRole.ASSISTANT:
                    contents.append(types.Content(
                        role="model",
                        parts=[types.Part(text=str(item.content))]
                    ))
            elif isinstance(item, FunctionCall):
                function_call = types.FunctionCall(
                    name=item.name,
                    args=json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments
                )
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(function_call=function_call)]
                ))
            elif isinstance(item, FunctionCallOutput):
                function_response = types.FunctionResponse(
                    name=item.name,
                    response={"output": item.output}
                )
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(function_response=function_response)]
                ))
        
        return contents, system_instruction
