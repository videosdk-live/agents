from __future__ import annotations

import os
from typing import Any, AsyncIterator, List, Union
import json

import httpx
import openai
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
    build_openai_schema,
)
from videosdk.agents.llm.chat_context import ChatContent, ImageContent

class OpenAILLM(LLM):
    
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_completion_tokens: int | None = None,
    ) -> None:
        """Initialize the OpenAI LLM plugin.

        Args:
            api_key (Optional[str], optional): OpenAI API key. Defaults to None.
            model (str): The model to use for the LLM plugin. Defaults to "gpt-4o".
            base_url (Optional[str], optional): The base URL for the OpenAI API. Defaults to None.
            temperature (float): The temperature to use for the LLM plugin. Defaults to 0.7.
            tool_choice (ToolChoice): The tool choice to use for the LLM plugin. Defaults to "auto".
            max_completion_tokens (Optional[int], optional): The maximum completion tokens to use for the LLM plugin. Defaults to None.
        """
        super().__init__()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either through api_key parameter or OPENAI_API_KEY environment variable")
        
        self.model = model
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.max_completion_tokens = max_completion_tokens
        self._cancelled = False
        
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

    @staticmethod
    def azure(
        *,
        model: str = "gpt-4o-mini",
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_completion_tokens: int | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> "OpenAILLM":
        """
        Create a new instance of Azure OpenAI LLM.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `AZURE_OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        - `project` from `OPENAI_PROJECT_ID`
        - `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
        - `api_version` from `OPENAI_API_VERSION`
        - `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`
        - `azure_deployment` from `AZURE_OPENAI_DEPLOYMENT` (if not provided, uses `model` as deployment name)
        """
        
        azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = api_version or os.getenv("OPENAI_API_VERSION")
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        azure_ad_token = azure_ad_token or os.getenv("AZURE_OPENAI_AD_TOKEN")
        organization = organization or os.getenv("OPENAI_ORG_ID")
        project = project or os.getenv("OPENAI_PROJECT_ID")
        
        if not azure_deployment:
            azure_deployment = model
        
        if not azure_endpoint:
            raise ValueError("Azure endpoint must be provided either through azure_endpoint parameter or AZURE_OPENAI_ENDPOINT environment variable")
        
        if not api_key and not azure_ad_token:
            raise ValueError("Either API key or Azure AD token must be provided")
        
        azure_client = openai.AsyncAzureOpenAI(
            max_retries=0,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=timeout
            if timeout
            else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
        )
        
        instance = OpenAILLM(
            model=model,
            temperature=temperature,
            tool_choice=tool_choice,
            max_completion_tokens=max_completion_tokens,
        )
        instance._client = azure_client
        return instance

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
        self._cancelled = False
        
        def _format_content(content: Union[str, List[ChatContent]]):
            if isinstance(content, str):
                return content

            formatted_parts = []
            for part in content:
                if isinstance(part, str):
                    formatted_parts.append({"type": "text", "text": part})
                elif isinstance(part, ImageContent):
                    image_url_data = {"url": part.to_data_url()}
                    if part.inference_detail != "auto":
                        image_url_data["detail"] = part.inference_detail
                    formatted_parts.append(
                        {
                            "type": "image_url",
                            "image_url": image_url_data,
                        }
                    )
            return formatted_parts

        completion_params = {
            "model": self.model,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": _format_content(msg.content),
                    **({"name": msg.name} if hasattr(msg, "name") else {}),
                }
                if isinstance(msg, ChatMessage)
                else {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": msg.name, "arguments": msg.arguments},
                }
                if isinstance(msg, FunctionCall)
                else {
                    "role": "function",
                    "name": msg.name,
                    "content": msg.output,
                }
                if isinstance(msg, FunctionCallOutput)
                else None
                for msg in messages.items
                if msg is not None
            ],
            "temperature": self.temperature,
            "stream": True,
            "max_tokens": self.max_completion_tokens,
        }
        if tools:
            formatted_tools = []
            for tool in tools:
                if not is_function_tool(tool):
                    continue
                try:
                    tool_schema = build_openai_schema(tool)
                    formatted_tools.append(tool_schema)
                except Exception as e:
                    self.emit("error", f"Failed to format tool {tool}: {e}")
                    continue
            
            if formatted_tools:
                completion_params["functions"] = formatted_tools
                completion_params["function_call"] = self.tool_choice
        completion_params.update(kwargs)
        try:
            response_stream = await self._client.chat.completions.create(**completion_params)
            
            current_content = ""
            current_function_call = None

            async for chunk in response_stream:
                if self._cancelled:
                    break
                
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                if delta.function_call:
                    if current_function_call is None:
                        current_function_call = {
                            "name": delta.function_call.name or "",
                            "arguments": delta.function_call.arguments or ""
                        }
                    else:
                        if delta.function_call.name:
                            current_function_call["name"] += delta.function_call.name
                        if delta.function_call.arguments:
                            current_function_call["arguments"] += delta.function_call.arguments
                elif current_function_call is not None:  
                    try:
                        args = json.loads(current_function_call["arguments"])
                        current_function_call["arguments"] = args
                    except json.JSONDecodeError:
                        self.emit("error", f"Failed to parse function arguments: {current_function_call['arguments']}")
                        current_function_call["arguments"] = {}
                    
                    yield LLMResponse(
                        content="",
                        role=ChatRole.ASSISTANT,
                        metadata={"function_call": current_function_call}
                    )
                    current_function_call = None
                
                elif delta.content is not None:
                    current_content = delta.content
                    yield LLMResponse(
                        content=current_content,
                        role=ChatRole.ASSISTANT
                    )

        except Exception as e:
            if not self._cancelled:
                self.emit("error", e)
            raise

    async def cancel_current_generation(self) -> None:
        self._cancelled = True

    async def aclose(self) -> None:
        """Cleanup resources by closing the HTTP client"""
        await self.cancel_current_generation()
        if self._client:
            await self._client.close()