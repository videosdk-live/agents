from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any, AsyncIterator

from videosdk.agents import (
    LLM,
    LLMResponse,
    ChatContext,
    ChatRole,
    ToolChoice,
    FunctionTool,
    is_function_tool,
    build_openai_schema,
)

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_TEXT_MODEL = "amazon.nova-lite-v1:0"

# Streaming exception event keys returned inline by the Bedrock event stream.
_STREAM_ERROR_KEYS = (
    "internalServerException",
    "modelStreamErrorException",
    "validationException",
    "throttlingException",
    "serviceUnavailableException",
)

_THINK_OPEN = "<thinking>"
_THINK_CLOSE = "</thinking>"


class _ThinkingStripper:
    """Incrementally removes ``<thinking>...</thinking>`` spans from a token stream.

    Bedrock models in the Amazon Nova family emit chain-of-thought wrapped in
    ``<thinking>`` tags inline with the spoken answer. In a voice pipeline that
    reasoning would be sent to TTS and read aloud, so it must be stripped before
    the text is yielded. Tags can be split across stream chunks, so this keeps a
    small pending buffer and only releases text it knows sits outside a thinking
    span.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._in_thinking = False
        self._thinking_buffer = ""

    @staticmethod
    def _partial_tag_suffix_len(text: str, tag: str) -> int:
        """Length of the longest suffix of ``text`` that is a prefix of ``tag``.

        Used to hold back a trailing fragment that might be the start of a tag
        (e.g. ``"...<thin"``) until the next chunk confirms or denies it.
        """
        max_len = min(len(text), len(tag) - 1)
        for size in range(max_len, 0, -1):
            if text[-size:] == tag[:size]:
                return size
        return 0

    def feed(self, text: str) -> str:
        """Add a chunk and return the text that is safe to emit now."""
        self._buffer += text
        out: list[str] = []

        while self._buffer:
            if not self._in_thinking:
                idx = self._buffer.find(_THINK_OPEN)
                if idx != -1:
                    out.append(self._buffer[:idx])
                    self._buffer = self._buffer[idx + len(_THINK_OPEN):]
                    self._in_thinking = True
                    self._thinking_buffer = ""
                    continue
                hold = self._partial_tag_suffix_len(self._buffer, _THINK_OPEN)
                emit_to = len(self._buffer) - hold
                out.append(self._buffer[:emit_to])
                self._buffer = self._buffer[emit_to:]
                break
            else:
                idx = self._buffer.find(_THINK_CLOSE)
                if idx != -1:
                    self._buffer = self._buffer[idx + len(_THINK_CLOSE):]
                    self._in_thinking = False
                    self._thinking_buffer = ""
                    continue
                hold = self._partial_tag_suffix_len(self._buffer, _THINK_CLOSE)
                keep = len(self._buffer) - hold
                self._thinking_buffer += self._buffer[:keep]
                self._buffer = self._buffer[keep:]
                break

        return "".join(out)

    def flush(self) -> str:
        """Return any remaining safe text once the stream has ended."""
        if self._in_thinking:
            text = self._thinking_buffer + self._buffer
            self._thinking_buffer = ""
            self._buffer = ""
            self._in_thinking = False
            return text
        remaining = self._buffer
        self._buffer = ""
        return remaining


class AWSBedrockLLM(LLM):
    """AWS Bedrock LLM (Converse API) plugin for VideoSDK Agents.

    Provides streaming text generation and function/tool calling end-to-end,
    matching the behaviour of the OpenAI and Google LLM plugins.

    The Bedrock ``converse_stream`` API is uniform across Bedrock-hosted model
    families (Amazon Nova, Anthropic Claude, Meta Llama, Mistral, ...), so the
    same plugin works for any model id that supports the Converse API.

    Reference:
        https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_TEXT_MODEL,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        region: str = "us-east-1",
        temperature: float = 0.7,
        tool_choice: ToolChoice = "auto",
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
        additional_request_fields: dict[str, Any] | None = None,
        cache_system: bool = False,
        cache_tools: bool = False,
        strip_thinking: bool = True,
        client: Any | None = None,
    ) -> None:
        """Initialize the AWS Bedrock LLM plugin.

        ``aws_access_key_id`` / ``aws_secret_access_key`` may be passed directly
        or read from the ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``
        environment variables. When neither is provided the standard boto3
        credential chain is used (e.g. an attached IAM role).

        Args:
            model: Bedrock model id or inference profile ARN. Defaults to
                ``"amazon.nova-lite-v1:0"``. Falls back to the
                ``BEDROCK_INFERENCE_PROFILE_ARN`` env var when not given.
                See https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-use.html
            aws_access_key_id: AWS access key id. Falls back to ``AWS_ACCESS_KEY_ID``.
            aws_secret_access_key: AWS secret access key. Falls back to ``AWS_SECRET_ACCESS_KEY``.
            aws_session_token: Optional AWS session token. Falls back to ``AWS_SESSION_TOKEN``.
            region: AWS region for Bedrock Runtime. Falls back to ``AWS_DEFAULT_REGION``.
                Defaults to ``"us-east-1"``.
            temperature: Sampling temperature. Defaults to 0.7.
            tool_choice: Controls which (if any) tool is called. One of
                ``"auto"``, ``"required"``, ``"none"`` or a specific-tool dict.
                Defaults to ``"auto"``.
            max_tokens: Maximum tokens to generate in the response.
            top_p: Nucleus sampling probability mass.
            top_k: Top-K tokens considered during sampling. Sent via
                ``additionalModelRequestFields`` (model support varies).
            stop_sequences: Sequences that stop generation.
            additional_request_fields: Extra fields merged into
                ``additionalModelRequestFields`` for model-specific parameters.
            cache_system: Append a prompt-cache checkpoint after the system
                prompt to reduce input token usage. Defaults to False.
            cache_tools: Append a prompt-cache checkpoint after the tool
                definitions. Defaults to False.
            strip_thinking: Remove ``<thinking>...</thinking>`` spans from the
                streamed text before it is yielded. Amazon Nova models emit
                chain-of-thought in these tags, which would otherwise be read
                aloud by TTS. Defaults to True.
            client: Optional pre-built boto3 ``bedrock-runtime`` client. When
                provided, the credential/region arguments are ignored and the
                caller retains ownership of the client.
        """
        super().__init__()

        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is not installed. Please install it with 'pip install boto3'"
            )

        self.model = model or os.getenv("BEDROCK_INFERENCE_PROFILE_ARN")
        if not self.model:
            raise ValueError(
                "model or inference profile arn must be set using the `model` argument "
                "or the BEDROCK_INFERENCE_PROFILE_ARN environment variable."
            )

        self.temperature = temperature
        self.tool_choice = tool_choice
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.stop_sequences = stop_sequences
        self.additional_request_fields = additional_request_fields
        self.cache_system = cache_system
        self.cache_tools = cache_tools
        self.strip_thinking = strip_thinking
        self._cancelled = False

        self.region = region or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            access_key = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
            session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")

            client_kwargs: dict[str, Any] = {
                "service_name": "bedrock-runtime",
                "region_name": self.region,
                "config": Config(
                    user_agent_extra="x-client-framework:videosdk-plugins-aws",
                    retries={"max_attempts": 1, "mode": "standard"},
                ),
            }
            # Only pass explicit credentials when available; otherwise defer to
            # the default boto3 credential chain (IAM role, profile, etc.).
            if access_key and secret_key:
                client_kwargs["aws_access_key_id"] = access_key
                client_kwargs["aws_secret_access_key"] = secret_key
                if session_token:
                    client_kwargs["aws_session_token"] = session_token

            self._client = boto3.client(**client_kwargs)

    # ── Format conversion ────────────────────────────────────────────────

    def _to_bedrock_messages(
        self, messages: ChatContext
    ) -> tuple[list[dict], list[dict] | None]:
        """Convert a ChatContext into Bedrock Converse ``messages`` + ``system``.

        Reuses the framework's Anthropic converter (which already enforces
        user/assistant alternation and pairs tool calls with their results)
        and rewrites each block into Bedrock Converse shape.
        """
        anthropic_messages, system_content = messages.to_anthropic_messages()

        bedrock_messages: list[dict] = []
        for msg in anthropic_messages:
            role = msg["role"]
            content = msg["content"]
            blocks: list[dict] = []

            if isinstance(content, str):
                if content:
                    blocks.append({"text": content})
            else:
                for part in content:
                    ptype = part.get("type")
                    if ptype == "text":
                        if part.get("text"):
                            blocks.append({"text": part["text"]})
                    elif ptype == "image":
                        block = self._convert_image_block(part.get("source", {}))
                        if block:
                            blocks.append(block)
                    elif ptype == "tool_use":
                        blocks.append(
                            {
                                "toolUse": {
                                    "toolUseId": part["id"],
                                    "name": part["name"],
                                    "input": part.get("input", {}) or {},
                                }
                            }
                        )
                    elif ptype == "tool_result":
                        raw = part.get("content", "")
                        text = raw if isinstance(raw, str) else json.dumps(raw)
                        tool_result: dict[str, Any] = {
                            "toolUseId": part["tool_use_id"],
                            "content": [{"text": text}],
                        }
                        if part.get("is_error"):
                            tool_result["status"] = "error"
                        blocks.append({"toolResult": tool_result})

            if blocks:
                bedrock_messages.append({"role": role, "content": blocks})

        while bedrock_messages and bedrock_messages[0]["role"] != "user":
            bedrock_messages.pop(0)

        system: list[dict] | None = None
        if system_content:
            system = [{"text": system_content}]
            if self.cache_system:
                system.append({"cachePoint": {"type": "default"}})

        return bedrock_messages, system

    @staticmethod
    def _convert_image_block(source: dict) -> dict | None:
        """Convert an Anthropic image source into a Bedrock image block.

        Bedrock Converse only accepts raw image bytes, so URL-based images are
        skipped (Bedrock cannot fetch them).
        """
        if source.get("type") != "base64":
            logger.warning("AWS Bedrock: skipping non-base64 image (URLs are unsupported)")
            return None
        media_type = source.get("media_type", "image/png")
        fmt = media_type.split("/")[-1].lower()
        if fmt == "jpg":
            fmt = "jpeg"
        try:
            data = base64.b64decode(source["data"])
        except Exception:
            logger.warning("AWS Bedrock: failed to decode image data")
            return None
        return {"image": {"format": fmt, "source": {"bytes": data}}}

    def _build_tool_config(self, tools: list[FunctionTool] | None) -> dict | None:
        """Build the Bedrock ``toolConfig`` from the agent's function tools."""
        if not tools:
            return None

        if self.tool_choice == "none":
            return None

        tool_specs: list[dict] = []
        seen: set[str] = set()
        for tool in tools:
            if not is_function_tool(tool):
                continue
            try:
                schema = build_openai_schema(tool)
            except Exception as e:
                self.emit("error", f"Failed to format tool {tool}: {e}")
                continue
            name = schema["name"]
            if name in seen:
                continue
            seen.add(name)
            tool_specs.append(
                {
                    "toolSpec": {
                        "name": name,
                        "description": schema.get("description", "") or "",
                        "inputSchema": {
                            "json": schema.get("parameters")
                            or {"type": "object", "properties": {}}
                        },
                    }
                }
            )

        if not tool_specs:
            return None

        if self.cache_tools:
            tool_specs.append({"cachePoint": {"type": "default"}})

        tool_config: dict[str, Any] = {"tools": tool_specs}

        if isinstance(self.tool_choice, dict) and self.tool_choice.get("type") == "function":
            tool_config["toolChoice"] = {
                "tool": {"name": self.tool_choice["function"]["name"]}
            }
        elif self.tool_choice == "required":
            tool_config["toolChoice"] = {"any": {}}
        elif self.tool_choice == "auto":
            tool_config["toolChoice"] = {"auto": {}}

        return tool_config

    # ── Chat ─────────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: ChatContext,
        tools: list[FunctionTool] | None = None,
        conversational_graph: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        """Stream a chat completion from AWS Bedrock via the Converse API.

        Args:
            messages: ChatContext containing the conversation history.
            tools: Optional list of function tools available to the model.
            conversational_graph: Not supported by Bedrock; ignored if provided.
            **kwargs: Additional fields merged into the Converse request.

        Yields:
            LLMResponse objects with streamed text and tool calls.
        """
        self._cancelled = False

        if conversational_graph is not None:
            logger.warning(
                "AWS Bedrock LLM does not support conversational_graph; ignoring it."
            )

        bedrock_messages, system = self._to_bedrock_messages(messages)

        params: dict[str, Any] = {
            "modelId": self.model,
            "messages": bedrock_messages,
        }
        if system:
            params["system"] = system

        tool_config = self._build_tool_config(tools)
        if tool_config:
            params["toolConfig"] = tool_config

        inference_config: dict[str, Any] = {"temperature": self.temperature}
        if self.max_tokens is not None:
            inference_config["maxTokens"] = self.max_tokens
        if self.top_p is not None:
            inference_config["topP"] = self.top_p
        if self.stop_sequences:
            inference_config["stopSequences"] = self.stop_sequences
        params["inferenceConfig"] = inference_config

        additional_fields: dict[str, Any] = {}
        if self.top_k is not None:
            additional_fields["top_k"] = self.top_k
        if self.additional_request_fields:
            additional_fields.update(self.additional_request_fields)
        if additional_fields:
            params["additionalModelRequestFields"] = additional_fields

        params.update(kwargs)

        usage_metadata: dict = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cached_tokens": 0,
            "request_id": None,
            "model": self.model,
        }
        tool_calls: dict[int, dict] = {}

        stripper = _ThinkingStripper() if self.strip_thinking else None

        try:
            response = await asyncio.to_thread(self._client.converse_stream, **params)
            usage_metadata["request_id"] = (
                response.get("ResponseMetadata", {}).get("RequestId")
            )

            stream = response.get("stream")
            if stream is None:
                return

            iterator = iter(stream)
            sentinel = object()

            while True:
                if self._cancelled:
                    break

                event = await asyncio.to_thread(next, iterator, sentinel)
                if event is sentinel:
                    break

                for err_key in _STREAM_ERROR_KEYS:
                    if err_key in event:
                        raise RuntimeError(
                            f"AWS Bedrock stream error ({err_key}): "
                            f"{event[err_key].get('message', event[err_key])}"
                        )

                if "contentBlockStart" in event:
                    block = event["contentBlockStart"]
                    index = block.get("contentBlockIndex", 0)
                    start = block.get("start", {})
                    if "toolUse" in start:
                        tool_use = start["toolUse"]
                        tool_calls[index] = {
                            "id": tool_use.get("toolUseId", ""),
                            "name": tool_use.get("name", ""),
                            "arguments": "",
                        }

                elif "contentBlockDelta" in event:
                    block = event["contentBlockDelta"]
                    index = block.get("contentBlockIndex", 0)
                    delta = block.get("delta", {})
                    if "text" in delta:
                        text = delta["text"]
                        if stripper is not None:
                            text = stripper.feed(text)
                        if text:
                            yield LLMResponse(
                                content=text,
                                role=ChatRole.ASSISTANT,
                                metadata={"usage": usage_metadata},
                            )
                    elif "toolUse" in delta and index in tool_calls:
                        tool_calls[index]["arguments"] += delta["toolUse"].get("input", "")

                elif "contentBlockStop" in event:
                    index = event["contentBlockStop"].get("contentBlockIndex", 0)
                    tc = tool_calls.pop(index, None)
                    if tc is not None:
                        yield self._emit_tool_call(tc, usage_metadata)

                elif "metadata" in event:
                    usage = event["metadata"].get("usage", {})
                    usage_metadata["prompt_tokens"] = usage.get("inputTokens", 0) or 0
                    usage_metadata["completion_tokens"] = usage.get("outputTokens", 0) or 0
                    usage_metadata["total_tokens"] = usage.get("totalTokens", 0) or 0
                    usage_metadata["prompt_cached_tokens"] = (
                        usage.get("cacheReadInputTokens", 0) or 0
                    )
                    yield LLMResponse(
                        content="",
                        role=ChatRole.ASSISTANT,
                        metadata={"usage": usage_metadata},
                    )

            if not self._cancelled:
                if stripper is not None:
                    tail = stripper.flush()
                    if tail:
                        yield LLMResponse(
                            content=tail,
                            role=ChatRole.ASSISTANT,
                            metadata={"usage": usage_metadata},
                        )
                for tc in tool_calls.values():
                    yield self._emit_tool_call(tc, usage_metadata)
            tool_calls.clear()

        except (BotoCoreError, ClientError) as e:
            if not self._cancelled:
                self.emit("error", e)
            raise
        except Exception as e:
            if not self._cancelled:
                self.emit("error", e)
            raise

    def _emit_tool_call(self, tc: dict, usage_metadata: dict) -> LLMResponse:
        """Build a function-call LLMResponse from an accumulated tool-use block."""
        args_str = tc.get("arguments") or ""
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            self.emit("error", f"Failed to parse tool call arguments: {args_str}")
            args = {}
        return LLMResponse(
            content="",
            role=ChatRole.ASSISTANT,
            metadata={
                "function_call": {
                    "id": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "arguments": args,
                    "call_id": tc.get("id", ""),
                },
                "usage": usage_metadata,
            },
        )

    async def cancel_current_generation(self) -> None:
        self._cancelled = True

    async def aclose(self) -> None:
        """Cleanup resources. Closes the boto3 client only if this instance owns it."""
        await self.cancel_current_generation()
        if self._owns_client and self._client is not None:
            try:
                await asyncio.to_thread(self._client.close)
            except Exception:
                pass
        await super().aclose()
