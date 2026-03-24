from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, List

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from videosdk.agents import LLM, LLMResponse, ChatContext, ChatRole
from videosdk.agents.llm.chat_context import ChatMessage, FunctionCall, FunctionCallOutput

logger = logging.getLogger(__name__)

_SUPPORTED_MODES: set[str] = {"messages", "custom"}


class LangGraphLLM(LLM):
    """VideoSDK LLM adapter that wraps a compiled LangGraph StateGraph.

    The entire graph — its nodes, edges, tool nodes, conditional routing,
    and internal state — runs as the "LLM" from VideoSDK's perspective.
    The STT → LLM → TTS pipeline stays unchanged; this adapter bridges
    VideoSDK's ChatContext into the graph's message state and streams the
    graph's AI output back as ``LLMResponse`` chunks.

    Args:
        graph: A compiled LangGraph graph (``StateGraph.compile()``).
            Must accept ``{"messages": list[BaseMessage]}`` as input and
            use ``MessagesState`` (or a compatible schema).
        output_node: If provided, only text chunks emitted by this node
            name are forwarded to the voice pipeline.  Use this to
            restrict output to a dedicated synthesis/response node and
            suppress intermediate researcher/planner node text.
            Only applies when ``stream_mode="messages"``.
            Defaults to ``None`` (all AI text chunks are forwarded).
        config: Optional LangGraph ``RunnableConfig`` dict (e.g. for
            thread IDs, recursion limits, or custom callbacks).
        stream_mode: LangGraph streaming mode.  ``"messages"`` (default)
            streams ``AIMessageChunk`` tokens; ``"custom"`` streams
            arbitrary objects emitted by graph nodes via ``graph.send()``.
            Pass a list to enable both simultaneously.
        subgraphs: If ``True``, stream tokens from nested subgraphs as
            well as the top-level graph.  Requires LangGraph ≥ 0.2.
        context: Optional LangGraph 2.0 context object injected into the
            graph at runtime.  Useful for passing dependencies (database
            connections, session objects) that should not live in state.

    Example — basic usage::

        from videosdk.plugins.langchain import LangGraphLLM

        graph = build_my_research_graph()   # returns CompiledStateGraph
        llm = LangGraphLLM(graph=graph, output_node="synthesizer")

    Example — nested subgraphs::

        llm = LangGraphLLM(graph=graph, subgraphs=True)

    Example — custom streaming (nodes emit strings via graph.send)::

        llm = LangGraphLLM(graph=graph, stream_mode="custom")
    """

    def __init__(
        self,
        graph: Any, 
        output_node: str | None = None,
        config: dict | None = None,
        stream_mode: str | list[str] = "messages",
        subgraphs: bool = False,
        context: Any | None = None,
    ) -> None:
        super().__init__()
        modes = {stream_mode} if isinstance(stream_mode, str) else set(stream_mode)
        unsupported = modes - _SUPPORTED_MODES
        if unsupported:
            raise ValueError(
                f"Unsupported stream_mode(s): {unsupported}. "
                f"Supported: {_SUPPORTED_MODES}"
            )
        self._graph = graph
        self._output_node = output_node
        self._config = config or {}
        self._stream_mode = stream_mode
        self._subgraphs = subgraphs
        self._context = context
        self._cancelled = False

    def _convert_messages(self, ctx: ChatContext) -> List[BaseMessage]:
        """Convert a VideoSDK ChatContext into a LangChain message list."""
        lc_messages: List[BaseMessage] = []

        for item in ctx.items:
            if item is None:
                continue

            if isinstance(item, ChatMessage):
                content = item.content
                if isinstance(content, list):
                    content = " ".join(
                        str(c) for c in content if isinstance(c, str)
                    )
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
                        tool_calls=[
                            {
                                "id": item.call_id,
                                "name": item.name,
                                "args": args,
                                "type": "tool_call",
                            }
                        ],
                    )
                )

            elif isinstance(item, FunctionCallOutput):
                lc_messages.append(
                    ToolMessage(
                        content=item.output,
                        tool_call_id=item.call_id,
                        name=item.name,
                    )
                )

        return lc_messages


    def _extract_message_chunk(self, item: Any) -> BaseMessageChunk | str | None:
        """Normalise a raw item from ``graph.astream(stream_mode='messages')``.

        LangGraph can yield tokens in several shapes depending on version and
        whether ``subgraphs=True`` is used:

        - ``(token, metadata)``                     — standard single-graph
        - ``(namespace, (token, metadata))``         — with subgraphs=True
        - ``(mode, (token, metadata))``              — multi-mode list variant
        - ``(namespace, mode, (token, metadata))``   — future-proof extension
        """
        if isinstance(item, (BaseMessageChunk, str)):
            return item

        if not isinstance(item, tuple):
            return None

        if len(item) == 2 and not isinstance(item[1], tuple):
            return item[0]

        if len(item) == 2 and isinstance(item[1], tuple):
            inner = item[1]
            if len(inner) == 2:
                return inner[0]

        if len(item) == 3 and isinstance(item[2], tuple):
            inner = item[2]
            if len(inner) == 2:
                return inner[0]

        return None

    def _chunk_to_response(self, raw: Any) -> LLMResponse | None:
        """Convert a raw token/payload to an LLMResponse, or None to discard."""
        content: str | None = None

        if isinstance(raw, str):
            content = raw
        elif isinstance(raw, BaseMessageChunk):
            content = raw.content if isinstance(raw.content, str) else None
        elif isinstance(raw, dict):
            c = raw.get("content")
            content = c if isinstance(c, str) else None
        elif hasattr(raw, "content"):
            c = raw.content
            content = c if isinstance(c, str) else None

        if not content:
            return None
        return LLMResponse(content=content, role=ChatRole.ASSISTANT)


    async def chat(
        self,
        messages: ChatContext,
        tools: list | None = None,
        conversational_graph: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        """Run the LangGraph graph and stream its AI text output.

        The full conversation history from ``messages`` is injected as the
        initial ``{"messages": [...]}`` graph state.  The graph runs through
        all its nodes and this method streams the output back to the
        VideoSDK voice pipeline.

        Args:
            messages: VideoSDK ChatContext with full conversation history.
            tools: Ignored — tool execution is handled inside the graph.
            conversational_graph: Accepted for API compatibility; not used.
            **kwargs: Forwarded to ``graph.astream()``.

        Yields:
            LLMResponse text chunks as the graph streams its output.
        """
        self._cancelled = False
        lc_messages = self._convert_messages(messages)

        run_config = self._config or None
        is_multi_mode = isinstance(self._stream_mode, list)

        try:

            astream_kwargs: dict[str, Any] = {
                "stream_mode": self._stream_mode,
            }
            if run_config:
                astream_kwargs["config"] = run_config
            if self._subgraphs:
                astream_kwargs["subgraphs"] = True
            if self._context is not None:
                astream_kwargs["context"] = self._context
            astream_kwargs.update(kwargs)

            try:
                aiter = self._graph.astream({"messages": lc_messages}, **astream_kwargs)
            except TypeError:
                safe_kwargs: dict[str, Any] = {"stream_mode": self._stream_mode}
                if run_config:
                    safe_kwargs["config"] = run_config
                safe_kwargs.update(kwargs)
                aiter = self._graph.astream({"messages": lc_messages}, **safe_kwargs)

            async for item in aiter:
                if self._cancelled:
                    return

                if is_multi_mode and isinstance(item, tuple) and len(item) == 2:
                    mode, data = item
                    if not isinstance(mode, str):
                        continue
                    if mode == "custom":
                        resp = self._chunk_to_response(data)
                        if resp:
                            yield resp
                    elif mode == "messages":
                        token = self._extract_message_chunk(data)
                        if token is None:
                            continue
                        if not isinstance(token, AIMessageChunk) or not token.content:
                            continue
                        if self._output_node is not None:
                            meta = data[1] if isinstance(data, tuple) and len(data) == 2 else {}
                            node = meta.get("langgraph_node", "") if isinstance(meta, dict) else ""
                            if node != self._output_node:
                                continue
                        resp = self._chunk_to_response(token)
                        if resp:
                            yield resp
                    continue

                if self._stream_mode == "custom":
                    resp = self._chunk_to_response(item)
                    if resp:
                        yield resp

                elif self._stream_mode == "messages":
                    token = self._extract_message_chunk(item)
                    if token is None:
                        continue
                    if not isinstance(token, AIMessageChunk) or not token.content:
                        continue
                    if self._output_node is not None:
                        meta = item[1] if isinstance(item, tuple) and len(item) == 2 else {}
                        node = meta.get("langgraph_node", "") if isinstance(meta, dict) else ""
                        if node != self._output_node:
                            continue
                    resp = self._chunk_to_response(token)
                    if resp:
                        yield resp

        except Exception as exc:
            if not self._cancelled:
                self.emit("error", exc)
            raise
        
    async def cancel_current_generation(self) -> None:
        self._cancelled = True

    async def aclose(self) -> None:
        await super().aclose()
