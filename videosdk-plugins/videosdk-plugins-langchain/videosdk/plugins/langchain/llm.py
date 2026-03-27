from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from videosdk.agents import LLM, LLMResponse, ChatContext, ChatRole
from videosdk.agents.llm.chat_context import ChatMessage, FunctionCall, FunctionCallOutput
from videosdk.agents.utils import FunctionTool, is_function_tool, get_tool_info, build_openai_schema

logger = logging.getLogger(__name__)

_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _make_stub_tool(name: str, description: str, parameters: dict) -> StructuredTool:
    """Build a LangChain StructuredTool stub from an OpenAI-compatible parameter schema.

    The stub's ``_run`` does nothing — it exists only so the model receives the
    correct tool schema via ``bind_tools``.  Actual execution is handled by
    VideoSDK's ContentGeneration dispatcher.
    """
    props = parameters.get("properties", {})
    required_set = set(parameters.get("required", []))

    model_fields: dict[str, Any] = {}
    for fname, fschema in props.items():
        py_type = _JSON_TYPE_MAP.get(fschema.get("type", "string"), str)
        fdesc = fschema.get("description", "")
        if fname in required_set:
            model_fields[fname] = (py_type, Field(description=fdesc))
        else:
            model_fields[fname] = (Optional[py_type], Field(None, description=fdesc))

    ArgsModel: type | None = (
        create_model(f"{name}_args", **model_fields) if model_fields else None
    )

    def _stub(**kwargs: Any) -> None:
        return None

    _stub.__name__ = name
    _stub.__doc__ = description or name

    kwargs: dict[str, Any] = {
        "func": _stub,
        "name": name,
        "description": description or name,
    }
    if ArgsModel is not None:
        kwargs["args_schema"] = ArgsModel

    return StructuredTool.from_function(**kwargs)


class LangChainLLM(LLM):
    """VideoSDK LLM adapter for any LangChain ``BaseChatModel``.

    **Two usage modes:**

    Mode A — VideoSDK tools (``@function_tool`` methods on the Agent class):
        Pass no tools at init. VideoSDK's ContentGeneration automatically passes
        ``@function_tool`` methods to ``chat(tools=...)``.  This adapter converts
        them to LangChain stubs for schema binding, then emits tool call metadata
        back to VideoSDK which handles dispatch and re-calls ``chat()`` with the
        result — exactly like ``OpenAILLM`` and ``GoogleLLM``.

    Mode B — LangChain-native tools (Tavily, Wikipedia, custom ``@tool`` functions):
        Pass LangChain tools at init via ``tools=[...]``.  The full tool-calling
        loop (call model → execute tool → feed result → repeat) runs entirely
        inside this adapter.  The voice pipeline only sees the final text stream.

    Both modes can also be used together on the same instance.

    Args:
        llm: Any LangChain ``BaseChatModel`` instance.
        tools: Optional list of LangChain tools executed internally (Mode B).
        max_tool_iterations: Safety cap on consecutive internal tool-call rounds.

    Example — VideoSDK tools (Mode A)::

        from videosdk.plugins.langchain import LangChainLLM
        from langchain_openai import ChatOpenAI

        llm = LangChainLLM(llm=ChatOpenAI(model="gpt-4o-mini"))

        # Tools come from @function_tool methods on the Agent subclass —
        # no extra configuration needed here.

    Example — LangChain tools (Mode B)::

        from videosdk.plugins.langchain import LangChainLLM
        from langchain_openai import ChatOpenAI
        from langchain_community.tools.tavily_search import TavilySearchResults

        llm = LangChainLLM(
            llm=ChatOpenAI(model="gpt-4o-mini"),
            tools=[TavilySearchResults(max_results=3)],
        )
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list | None = None,
        max_tool_iterations: int = 10,
    ) -> None:
        super().__init__()
        self._base_llm = llm
        self._langchain_tools: list = tools or []
        self._max_tool_iterations = max_tool_iterations
        self._cancelled = False
        self._lc_tool_map: dict[str, Any] = {t.name: t for t in self._langchain_tools}
        self._llm_with_lc_tools = (
            self._base_llm.bind_tools(self._langchain_tools)
            if self._langchain_tools
            else self._base_llm
        )

    def _convert_messages(self, ctx: ChatContext) -> List[BaseMessage]:
        """Convert a VideoSDK ChatContext into a LangChain message list."""
        lc_messages: List[BaseMessage] = []
        for item in ctx.items:
            if item is None:
                continue
            if isinstance(item, ChatMessage):
                content = item.content
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content if isinstance(c, str))
                if item.role == ChatRole.SYSTEM:
                    lc_messages.append(SystemMessage(content=content))
                elif item.role == ChatRole.USER:
                    lc_messages.append(HumanMessage(content=content))
                elif item.role == ChatRole.ASSISTANT:
                    lc_messages.append(AIMessage(content=content))
            elif isinstance(item, FunctionCall):
                try:
                    args = (
                        json.loads(item.arguments)
                        if isinstance(item.arguments, str)
                        else item.arguments
                    )
                except json.JSONDecodeError:
                    args = {}
                lc_messages.append(
                    AIMessage(
                        content="",
                        tool_calls=[{"id": item.call_id, "name": item.name, "args": args, "type": "tool_call"}],
                    )
                )
            elif isinstance(item, FunctionCallOutput):
                lc_messages.append(
                    ToolMessage(content=item.output, tool_call_id=item.call_id, name=item.name)
                )
        return lc_messages

    async def _stream_round(
        self, llm: BaseChatModel, lc_messages: List[BaseMessage], **kwargs: Any
    ):
        """Stream one model call. Yields (content_chunk, tool_call_chunks) per chunk."""
        async for chunk in llm.astream(lc_messages, **kwargs):
            if self._cancelled:
                return
            if not isinstance(chunk, AIMessageChunk):
                continue
            yield chunk


    async def _chat_videosdk_tools(
        self,
        lc_messages: List[BaseMessage],
        videosdk_tool_names: set[str],
        llm: BaseChatModel,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        """Single-shot stream: text chunks yielded, tool calls emitted as metadata."""
        pending_tool_calls: dict[int, dict] = {}

        async for chunk in self._stream_round(llm, lc_messages, **kwargs):
            if chunk.content:
                yield LLMResponse(content=chunk.content, role=ChatRole.ASSISTANT)

            if chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    idx = tc.get("index", 0)
                    if idx not in pending_tool_calls:
                        pending_tool_calls[idx] = {
                            "id": tc.get("id") or "",
                            "name": tc.get("name") or "",
                            "args": tc.get("args") or "",
                        }
                    else:
                        existing = pending_tool_calls[idx]
                        if tc.get("name"):
                            existing["name"] += tc["name"]
                        if tc.get("args"):
                            existing["args"] += tc["args"]
                        if tc.get("id") and not existing["id"]:
                            existing["id"] = tc["id"]

        for tc_data in pending_tool_calls.values():
            try:
                args = json.loads(tc_data["args"]) if tc_data["args"] else {}
            except json.JSONDecodeError:
                args = {}
            yield LLMResponse(
                content="",
                role=ChatRole.ASSISTANT,
                metadata={"function_call": {"name": tc_data["name"], "arguments": args, "id": tc_data["id"]}},
            )

    async def _chat_lc_tools_loop(
        self,
        lc_messages: List[BaseMessage],
        llm: BaseChatModel,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        """Full internal tool-calling loop until a text-only response is produced."""
        for iteration in range(self._max_tool_iterations):
            if self._cancelled:
                return

            full_content = ""
            pending_tool_calls: dict[int, dict] = {}

            async for chunk in self._stream_round(llm, lc_messages, **kwargs):
                if chunk.content:
                    full_content += chunk.content
                    yield LLMResponse(content=chunk.content, role=ChatRole.ASSISTANT)
                if chunk.tool_call_chunks:
                    for tc in chunk.tool_call_chunks:
                        idx = tc.get("index", 0)
                        if idx not in pending_tool_calls:
                            pending_tool_calls[idx] = {
                                "id": tc.get("id") or "",
                                "name": tc.get("name") or "",
                                "args": tc.get("args") or "",
                            }
                        else:
                            existing = pending_tool_calls[idx]
                            if tc.get("name"):
                                existing["name"] += tc["name"]
                            if tc.get("args"):
                                existing["args"] += tc["args"]
                            if tc.get("id") and not existing["id"]:
                                existing["id"] = tc["id"]

            if not pending_tool_calls:
                break

            parsed: list[dict] = []
            for tc_data in pending_tool_calls.values():
                try:
                    args = json.loads(tc_data["args"]) if tc_data["args"] else {}
                except json.JSONDecodeError:
                    args = {}
                parsed.append({"id": tc_data["id"], "name": tc_data["name"], "args": args, "type": "tool_call"})

            lc_messages.append(AIMessage(content=full_content, tool_calls=parsed))

            for tc in parsed:
                if self._cancelled:
                    return
                tool = self._lc_tool_map.get(tc["name"])
                if tool is None:
                    result = f"Tool '{tc['name']}' not found."
                    logger.warning(result)
                else:
                    try:
                        result = await tool.ainvoke(tc["args"])
                    except Exception as exc:
                        result = f"Tool execution error: {exc}"
                        logger.error("Error executing tool %s: %s", tc["name"], exc)

                lc_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])
                )

            logger.debug("LangChainLLM tool iteration %d complete", iteration + 1)

    async def chat(
        self,
        messages: ChatContext,
        tools: list[FunctionTool] | None = None,
        conversational_graph: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        """Stream a response from the LangChain model.

        - If VideoSDK ``@function_tool`` tools are passed (via ``tools``): the model
          receives their schemas, and any tool calls are emitted as ``LLMResponse``
          metadata for VideoSDK ContentGeneration to dispatch (Mode A).
        - If only LangChain-native tools were given at init (``tools`` is empty):
          the internal tool-calling loop runs until a text answer is produced (Mode B).

        Args:
            messages: VideoSDK ChatContext with full conversation history.
            tools: VideoSDK FunctionTools from the Agent's ``@function_tool`` methods.
            conversational_graph: Accepted for API compatibility; not used by LangChain.
            **kwargs: Forwarded to the underlying model's ``astream`` call.

        Yields:
            LLMResponse chunks — text content and/or tool call metadata.
        """
        self._cancelled = False
        lc_messages = self._convert_messages(messages)

        try:
            videosdk_tools = [vt for vt in (tools or []) if is_function_tool(vt)]
            if videosdk_tools:
                videosdk_tool_names = set()
                stub_tools: list[StructuredTool] = []
                for vt in videosdk_tools:
                    info = get_tool_info(vt)
                    schema = build_openai_schema(vt)
                    videosdk_tool_names.add(info.name)
                    stub_tools.append(
                        _make_stub_tool(
                            name=info.name,
                            description=info.description or info.name,
                            parameters=schema.get("parameters", {}),
                        )
                    )
                active_llm = self._base_llm.bind_tools(stub_tools + self._langchain_tools)
                async for chunk in self._chat_videosdk_tools(
                    lc_messages, videosdk_tool_names, active_llm, **kwargs
                ):
                    yield chunk
                return

            async for chunk in self._chat_lc_tools_loop(
                lc_messages, self._llm_with_lc_tools, **kwargs
            ):
                yield chunk

        except Exception as exc:
            if not self._cancelled:
                self.emit("error", exc)
            raise

    async def cancel_current_generation(self) -> None:
        self._cancelled = True

    async def aclose(self) -> None:
        await super().aclose()
