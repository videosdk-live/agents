from __future__ import annotations
import os
import json
from typing import Any, AsyncIterator, List, Union
import httpx
import anthropic
from videosdk.agents import LLM, LLMResponse, ChatContext, ChatRole, ChatMessage, FunctionCall, FunctionCallOutput, ToolChoice, FunctionTool, is_function_tool, build_openai_schema, ImageContent, ChatContent,ConversationalGraphResponse

class AnthropicLLM(LLM):

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        base_url: str | None = None,
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_tokens: int = 1024,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> None:
        """Initialize the Anthropic LLM

        Args:
            api_key (str | None, optional): Anthropic API key. Uses ANTHROPIC_API_KEY environment variable if not provided. Defaults to None.
            model (str): The anthropic model to use for the LLM, e.g. "claude-sonnet-4-20250514". Defaults to "claude-sonnet-4-20250514".
            base_url (str | None, optional): The base URL to use for the LLM. Defaults to None.
            temperature (float): The temperature to use for the LLM, e.g. 0.7. Defaults to 0.7.
            tool_choice (ToolChoice): The tool choice to use for the LLM, e.g. "auto". Defaults to "auto".
            max_tokens (int): The maximum number of tokens to use for the LLM, e.g. 1024. Defaults to 1024.
            top_k (int | None, optional): The top K to use for the LLM. Defaults to None.
            top_p (float | None, optional): The top P to use for the LLM. Defaults to None.
        """
        super().__init__()

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided either through api_key parameter or ANTHROPIC_API_KEY environment variable")

        self.model = model
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p
        self._cancelled = False

        self._client = anthropic.AsyncClient(
            api_key=self.api_key,
            base_url=base_url or None,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=15.0, read=5.0, write=5.0, pool=5.0),
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
        conversational_graph: Any | None = None,
        **kwargs: Any
    ) -> AsyncIterator[LLMResponse]:
        """
        Implement chat functionality using Anthropic's Claude API

        Args:
            messages: ChatContext containing conversation history
            tools: Optional list of function tools available to the model
            **kwargs: Additional arguments passed to the Anthropic API

        Yields:
            LLMResponse objects containing the model's responses
        """
        self._cancelled = False

        try:
            anthropic_messages, system_content = self._convert_messages_to_anthropic_format(
                messages)
            completion_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": True,
            }

            # Enhance system prompt to request JSON output
            if system_content and conversational_graph:
                schema_example = json.dumps(ConversationalGraphResponse.model_json_schema(), indent=2)
                system_content += f"\n\nCRITICAL: You MUST respond with ONLY valid JSON. Do NOT wrap it in ```json code blocks or any markdown. Return raw JSON matching this schema:\n{schema_example}"

            if system_content:
                completion_params["system"] = system_content

            if self.top_k is not None:
                completion_params["top_k"] = self.top_k
            if self.top_p is not None:
                completion_params["top_p"] = self.top_p

            if tools:
                formatted_tools = []
                seen_tool_names = set()

                for tool in tools:
                    if not is_function_tool(tool):
                        continue
                    try:
                        openai_schema = build_openai_schema(tool)
                        tool_name = openai_schema["name"]

                        if tool_name in seen_tool_names:
                            continue

                        seen_tool_names.add(tool_name)
                        anthropic_tool = {
                            "name": tool_name,
                            "description": openai_schema["description"],
                            "input_schema": openai_schema["parameters"]
                        }
                        formatted_tools.append(anthropic_tool)
                    except Exception as e:
                        self.emit(
                            "error", f"Failed to format tool {tool}: {e}")
                        continue

                if formatted_tools:
                    completion_params["tools"] = formatted_tools

                    if self.tool_choice == "required":
                        completion_params["tool_choice"] = {"type": "any"}
                    elif self.tool_choice == "auto":
                        completion_params["tool_choice"] = {"type": "auto"}
                    elif self.tool_choice == "none":
                        del completion_params["tools"]

            completion_params.update(kwargs)

            if conversational_graph:
                completion_params["extra_headers"] = {
                    "anthropic-beta": "structured-outputs-2025-11-13"
                }

            response_stream = await self._client.messages.create(**completion_params)

            # Accumulate JSON response
            current_content = ""
            current_tool_call = None
            current_tool_call_id = None
            current_tool_arguments = ""
            
            usage_metadata = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_cached_tokens": 0
            }

            # State for partial JSON parsing
            streaming_state = {
                "in_response": False,
                "response_start_index": -1,
                "yielded_content_length": 0
            }

            async for event in response_stream:
                if self._cancelled:
                    break

                if event.type == "message_start":
                    usage_metadata["prompt_tokens"] = event.message.usage.input_tokens
                    usage_metadata["prompt_cached_tokens"] = getattr(event.message.usage, 'cache_read_input_tokens', 0) or 0
                    yield LLMResponse(content="", role=ChatRole.ASSISTANT, metadata={"usage": usage_metadata})

                elif event.type == "message_delta":
                    if hasattr(event, "usage"):
                        usage_metadata["completion_tokens"] = event.usage.output_tokens
                        usage_metadata["total_tokens"] = usage_metadata["prompt_tokens"] + usage_metadata["completion_tokens"]
                        yield LLMResponse(content="", role=ChatRole.ASSISTANT, metadata={"usage": usage_metadata})


                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool_call_id = event.content_block.id
                        current_tool_call = {
                            "name": event.content_block.name,
                            "arguments": ""
                        }
                        current_tool_arguments = ""

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        current_content += delta.text
                        if conversational_graph:
                            for content_chunk in conversational_graph.stream_conversational_graph_response(current_content, streaming_state):
                                yield LLMResponse(content=content_chunk, role=ChatRole.ASSISTANT, metadata={"usage": usage_metadata})
                        else:
                            yield LLMResponse(content=delta.text, role=ChatRole.ASSISTANT, metadata={"usage": usage_metadata})

                    elif delta.type == "input_json_delta":
                        if current_tool_call:
                            current_tool_arguments += delta.partial_json

                elif event.type == "content_block_stop":
                    if current_tool_call and current_tool_call_id:
                        try:
                            parsed_args = json.loads(
                                current_tool_arguments) if current_tool_arguments else {}
                            current_tool_call["arguments"] = parsed_args
                        except json.JSONDecodeError:
                            current_tool_call["arguments"] = {}

                        yield LLMResponse(
                            content="",
                            role=ChatRole.ASSISTANT,
                            metadata={
                                "function_call": {
                                    "id": current_tool_call_id,
                                    "name": current_tool_call["name"],
                                    "arguments": current_tool_call["arguments"],
                                    "call_id": current_tool_call_id
                                },
                                "usage": usage_metadata
                            }
                        )
                        current_tool_call = None
                        current_tool_call_id = None
                        current_tool_arguments = ""

            if not self._cancelled:
                yield LLMResponse(content="", role=ChatRole.ASSISTANT, metadata={"usage": usage_metadata})


            if current_content and not self._cancelled:
                if conversational_graph:
                    try:
                        parsed_json = json.loads(current_content.strip())
                        yield LLMResponse(
                            content="",
                            role=ChatRole.ASSISTANT,
                            metadata={"graph_response":parsed_json, "usage": usage_metadata}
                        )
                    except json.JSONDecodeError:
                             yield LLMResponse(
                                content=current_content,
                                role=ChatRole.ASSISTANT,
                                metadata={"usage": usage_metadata}
                            )
                else:
                    pass

        except anthropic.APIError as e:
            if not self._cancelled:
                self.emit("error", e)
            raise
        except Exception as e:
            if not self._cancelled:
                self.emit("error", e)
            raise

    async def cancel_current_generation(self) -> None:
        self._cancelled = True

    def _convert_messages_to_anthropic_format(self, messages: ChatContext) -> tuple[list[dict], str | None]:
        """Internal Method: Convert ChatContext to Anthropic message format"""

        def _format_content(content: Union[str, List[ChatContent]]):
            if isinstance(content, str):
                return content

            has_images = any(isinstance(p, ImageContent) for p in content)

            if not has_images and len(content) == 1 and isinstance(content[0], str):
                return content[0]

            formatted_parts = []
            image_parts = [p for p in content if isinstance(p, ImageContent)]
            text_parts = [p for p in content if isinstance(p, str)]

            for part in image_parts:
                data_url = part.to_data_url()

                if data_url.startswith("data:"):
                    header, b64_data = data_url.split(",", 1)
                    media_type = header.split(";")[0].split(":")[1]
                    formatted_parts.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_data,
                            },
                        }
                    )
                else:
                    formatted_parts.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": data_url},
                        }
                    )

            for part in text_parts:
                formatted_parts.append({"type": "text", "text": part})

            return formatted_parts

        anthropic_messages = []
        system_content = None
        pending_tool_results = {} 

        for item in messages.items:
            if isinstance(item, ChatMessage):
                if item.role == ChatRole.SYSTEM:
                    if isinstance(item.content, list):
                        system_content = next(
                            (str(p)
                             for p in item.content if isinstance(p, str)), ""
                        )
                    else:
                        system_content = str(item.content)
                    continue
                else:
                    anthropic_messages.append(
                        {"role": item.role.value,
                            "content": _format_content(item.content)}
                    )
            elif isinstance(item, FunctionCall):
                anthropic_messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": item.call_id,
                            "name": item.name,
                            "input": json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments
                        }
                    ]
                })
            elif isinstance(item, FunctionCallOutput):
                pending_tool_results[item.call_id] = item

        final_messages = []
        i = 0
        while i < len(anthropic_messages):
            msg = anthropic_messages[i]
            final_messages.append(msg)

            if (isinstance(msg.get("content"), list) and
                any(part.get("type") == "tool_use" for part in msg["content"])):
                tool_use_part = next(
                    part for part in msg["content"] if part.get("type") == "tool_use"
                )
                tool_call_id = tool_use_part["id"]

                if tool_call_id in pending_tool_results:
                    tool_result = pending_tool_results[tool_call_id]
                    final_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": tool_result.output,
                                "is_error": tool_result.is_error
                            }
                        ]
                    })
                    del pending_tool_results[tool_call_id]

            i += 1

        return final_messages, system_content

    async def aclose(self) -> None:
        """Internal Method: Cleanup resources by closing the HTTP client"""
        await self.cancel_current_generation()
        if self._client:
            await self._client.close()
        await super().aclose()
