from __future__ import annotations

import ast
import asyncio
import base64
import json
import logging
import os
import uuid
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


def _parse_tool_call_expression(
    text: str, tool_params: dict[str, list[str]]
) -> dict | None:
    """Parse a text-form function call into ``{"name", "arguments"}`` or None.

    Models without native Converse tool use (e.g. Google Gemma) emit function
    calls as plain text such as ``end_call(message="...")`` rather than as
    Bedrock ``toolUse`` blocks. This tolerates a surrounding markdown code fence
    and a ``print(...)`` / list wrapper, and uses the ``ast`` module so quoted
    arguments — including non-ASCII (Hindi) strings with commas — parse safely.

    Returns None when the text does not (yet) contain a complete, balanced call
    to one of the known tools, which lets the streaming sniffer keep buffering.
    """
    if not text or not tool_params:
        return None

    src = text.strip()
    if src.startswith("```"):
        nl = src.find("\n")
        if nl != -1:
            src = src[nl + 1:]
        src = src.rstrip()
        if src.endswith("```"):
            src = src[:-3]
    src = src.strip()
    if not src:
        return None

    try:
        tree: ast.AST = ast.parse(src, mode="eval")
    except SyntaxError:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return None

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in tool_params
        ):
            arguments: dict[str, Any] = {}
            try:
                for kw in node.keywords:
                    if kw.arg is None:
                        continue
                    arguments[kw.arg] = ast.literal_eval(kw.value)
                positional = [ast.literal_eval(a) for a in node.args]
            except Exception:
                # Incomplete during streaming, or non-literal argument.
                return None

            param_names = tool_params[node.func.id]
            for i, value in enumerate(positional):
                if i < len(param_names):
                    arguments.setdefault(param_names[i], value)

            return {"name": node.func.id, "arguments": arguments}

    return None


class _ToolCallSniffer:
    """Convert a model's text-form function calls into structured tool calls.

    Models like Google Gemma do not emit Bedrock ``toolUse`` content blocks.
    When prompted with available functions they print the call as plain text,
    e.g. ``end_call(message="...")`` (optionally inside a ```tool_code``` fence
    or wrapped in ``print(...)``). Without interception that text streams to TTS
    and is spoken aloud instead of executing the tool.

    This buffers the start of an assistant turn while the text could still be
    such a call. If it resolves to a call to a known tool, the call is parsed
    and the text is suppressed; otherwise the buffered text is released
    unchanged, so ordinary responses stream with negligible added latency. Only
    the *start* of a turn is held back — once disarmed the turn streams as-is.
    """

    def __init__(self, tool_params: dict[str, list[str]]) -> None:
        self._tool_params = tool_params
        self._buffer = ""
        self._armed = True   # still deciding whether this turn is a tool call
        self._done = False   # a call was emitted; swallow any trailing text

    def feed(self, text: str) -> tuple[str, dict | None]:
        """Return ``(text_to_emit, parsed_call_or_None)`` for a chunk."""
        if self._done:
            return "", None
        if not self._armed:
            return text, None

        self._buffer += text

        call = _parse_tool_call_expression(self._buffer, self._tool_params)
        if call is not None:
            self._done = True
            self._buffer = ""
            return "", call

        if self._could_be_call(self._buffer):
            return "", None

        # Definitely not a tool call → release everything and stream normally.
        self._armed = False
        out, self._buffer = self._buffer, ""
        return out, None

    def flush(self) -> tuple[str, dict | None]:
        """Resolve whatever is still buffered once the stream ends."""
        if self._done or not self._buffer:
            self._buffer = ""
            return "", None
        call = _parse_tool_call_expression(self._buffer, self._tool_params)
        out = self._buffer
        self._buffer = ""
        self._done = True
        if call is not None:
            return "", call
        return out, None

    def _could_be_call(self, buf: str) -> bool:
        """Whether ``buf`` might still be building toward a known tool call."""
        probe = buf.lstrip()
        if not probe:
            return True
        # A partial or in-progress markdown code fence.
        if probe.startswith("```"):
            nl = probe.find("\n")
            if nl == -1:
                return True
            probe = probe[nl + 1:].lstrip()
            if not probe:
                return True
        elif probe.startswith("`"):
            return True
        # Wrappers some models add around the call (possibly still being typed).
        for prefix in ("print(", "["):
            if probe.startswith(prefix):
                probe = probe[len(prefix):].lstrip()
                break
            if prefix.startswith(probe):
                return True
        if not probe:
            return True
        for name in self._tool_params:
            opener = name + "("
            if probe.startswith(opener) or opener.startswith(probe):
                return True
        return False


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
        text_tool_calls: bool | None = None,
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
            text_tool_calls: Parse function calls that the model prints as plain
                text (e.g. ``end_call(message="...")``) instead of emitting
                Bedrock ``toolUse`` blocks, and convert them into real tool
                calls. Required for models without native Converse tool use such
                as Google Gemma. Defaults to None, which auto-enables it for
                model ids known to lack native tool use (e.g. Gemma).
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
        self.text_tool_calls = (
            text_tool_calls
            if text_tool_calls is not None
            else self._model_uses_text_tool_calls(self.model)
        )
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

    @staticmethod
    def _model_uses_text_tool_calls(model: str) -> bool:
        """Whether a model emits tool calls as plain text rather than as
        Converse ``toolUse`` blocks (i.e. it lacks native tool use on Bedrock).
        """
        return "gemma" in (model or "").lower()

    @staticmethod
    def _build_tool_params(
        tools: list[FunctionTool] | None,
    ) -> dict[str, list[str]]:
        """Map each tool name to its parameter names in declaration order.

        Used to interpret text-form function calls (positional args are mapped
        back to parameter names) when the model lacks native tool use.
        """
        tool_params: dict[str, list[str]] = {}
        for tool in tools or []:
            if not is_function_tool(tool):
                continue
            try:
                schema = build_openai_schema(tool)
            except Exception:
                continue
            props = (schema.get("parameters") or {}).get("properties") or {}
            tool_params[schema["name"]] = list(props.keys())
        return tool_params

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

        sniffer: _ToolCallSniffer | None = None
        if self.text_tool_calls and tool_config:
            tool_params = self._build_tool_params(tools)
            if tool_params:
                sniffer = _ToolCallSniffer(tool_params)

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
                            for resp in self._text_responses(
                                text, sniffer, usage_metadata
                            ):
                                yield resp
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
                        for resp in self._text_responses(
                            tail, sniffer, usage_metadata
                        ):
                            yield resp
                if sniffer is not None:
                    emit_text, call = sniffer.flush()
                    if emit_text:
                        yield LLMResponse(
                            content=emit_text,
                            role=ChatRole.ASSISTANT,
                            metadata={"usage": usage_metadata},
                        )
                    if call is not None:
                        yield self._text_tool_call_response(call, usage_metadata)
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

    def _text_responses(
        self,
        text: str,
        sniffer: _ToolCallSniffer | None,
        usage_metadata: dict,
    ) -> list[LLMResponse]:
        """Route streamed text through the tool-call sniffer.

        Returns the LLMResponses to yield: spoken text and/or a parsed
        text-form tool call. When no sniffer is active the text passes through
        unchanged.
        """
        if sniffer is None:
            return [
                LLMResponse(
                    content=text,
                    role=ChatRole.ASSISTANT,
                    metadata={"usage": usage_metadata},
                )
            ]

        emit_text, call = sniffer.feed(text)
        responses: list[LLMResponse] = []
        if emit_text:
            responses.append(
                LLMResponse(
                    content=emit_text,
                    role=ChatRole.ASSISTANT,
                    metadata={"usage": usage_metadata},
                )
            )
        if call is not None:
            responses.append(self._text_tool_call_response(call, usage_metadata))
        return responses

    def _text_tool_call_response(
        self, call: dict, usage_metadata: dict
    ) -> LLMResponse:
        """Build a function-call LLMResponse from a parsed text-form tool call.

        A synthetic id is generated because models without native tool use do
        not supply a ``toolUseId``.
        """
        call_id = uuid.uuid4().hex
        return LLMResponse(
            content="",
            role=ChatRole.ASSISTANT,
            metadata={
                "function_call": {
                    "id": call_id,
                    "name": call["name"],
                    "arguments": call["arguments"],
                    "call_id": call_id,
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
