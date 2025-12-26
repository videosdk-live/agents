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
    ConversationalGraphResponse
)
from dataclasses import dataclass

@dataclass
class VertexAIConfig:
    project_id: str| None = None
    location: str| None = None

class GoogleLLM(LLM):
    
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        vertexai: bool = False,
        vertexai_config: VertexAIConfig| None = None,
    ) -> None:
        """Initialize the Google LLM plugin
        
        Args:
            api_key (str): Google API key. If not provided, will attempt to read from GOOGLE_API_KEY env var
            model (str): The model to use for the LLM plugin.
            temperature (float): The temperature to use for the LLM plugin
            tool_choice (ToolChoice): The tool choice to use for the LLM plugin
            max_output_tokens (int): The maximum output tokens to use for the LLM plugin
            top_p (float): The top P to use for the LLM plugin
            top_k (int): The top K to use for the LLM plugin
            presence_penalty (float): The presence penalty to use for the LLM plugin
            frequency_penalty (float): The frequency penalty to use for the LLM plugin
            vertexai (bool): Whether to use Vertex AI
            vertexai_config (VertexAIConfig): The Vertex AI config if custom otherwise from env
        """
        super().__init__()
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.vertexai = vertexai
        self.vertexai_config = vertexai_config
        if not self.vertexai and not self.api_key:
            raise ValueError("For VertexAI: Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the service account key file"\
                            "The Google Cloud project and location can be set via VertexAIConfig or the environment variables `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`. location defaults to `us-central1`"\
                            "For Google Gemini API: Set the `api_key` argument or the `GOOGLE_API_KEY` environment variable.")
        self.model = model
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self._cancelled = False
        if self.vertexai:
            project_id = (self.vertexai_config.project_id if self.vertexai_config else None) or os.getenv("GOOGLE_CLOUD_PROJECT")
            if project_id is None:
                service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                if service_account_path:
                    from google.oauth2 import service_account
                    creds = service_account.Credentials.from_service_account_file(service_account_path)
                    project_id = creds.project_id

            location = (self.vertexai_config.location if self.vertexai_config else None) or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"
            self._client = Client(
                vertexai=True,
                project=project_id,
                location=location,
            )
        else:
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY required")
            self._client = Client(api_key=self.api_key)

    async def chat(
        self,
        messages: ChatContext,
        tools: list[FunctionTool] | None = None,
        conversational_graph: Any | None = None,
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
            if conversational_graph:
                config_params["response_mime_type"] = "application/json"
                config_params["response_json_schema"] = ConversationalGraphResponse.model_json_schema()
            
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

            streaming_state = {
                "in_response": False,
                "response_start_index": -1,
                "yielded_content_length": 0
            }
            
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
                
                usage = None
                if response.usage_metadata:
                    usage = {
                        "prompt_tokens": response.usage_metadata.prompt_token_count,
                        "completion_tokens": response.usage_metadata.candidates_token_count,
                        "total_tokens": response.usage_metadata.total_token_count,
                        # Gemini uses cached_content_token_count if caching is enabled
                        "prompt_cached_tokens": getattr(response.usage_metadata, 'cached_content_token_count', 0)
                    }

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
                            metadata={"function_call": function_call, "usage": usage}
                        )
                    elif part.text:
                        current_content += part.text
                        if conversational_graph:
                            for content_chunk in conversational_graph.stream_conversational_graph_response(current_content, streaming_state):
                                yield LLMResponse(content=content_chunk, role=ChatRole.ASSISTANT, metadata={"usage": usage})
                        else:
                            yield LLMResponse(content=part.text, role=ChatRole.ASSISTANT, metadata={"usage": usage})
            
            if current_content and not self._cancelled:
                if conversational_graph:
                    try:
                        parsed_json = json.loads(current_content.strip())
                        yield LLMResponse(
                            content="",
                            role=ChatRole.ASSISTANT,
                            metadata={"graph_response":parsed_json, "usage": usage}
                        )
                    except json.JSONDecodeError:
                        yield LLMResponse(
                            content=current_content,
                            role=ChatRole.ASSISTANT,
                            metadata={"usage": usage}
                        )
                else:
                    pass

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
        await super().aclose()