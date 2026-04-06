from __future__ import annotations

from typing import AsyncIterator, Literal,Any
import asyncio
import json
import time
import logging
from .event_emitter import EventEmitter
from .llm.llm import LLM, ResponseChunk
from .llm.chat_context import ChatRole,FunctionCallOutput
from .utils import is_function_tool, get_tool_info, UserState, AgentState
from .agent import Agent
from .metrics import metrics_collector

logger = logging.getLogger(__name__)


class ContentGeneration(EventEmitter[Literal["generation_started", "generation_chunk", "generation_complete", "tool_called", "agent_switched"]]):
    """
    Handles LLM processing and content generation.
    
    Events:
    - generation_started: LLM generation begins
    - generation_chunk: Streaming chunk received
    - generation_complete: Generation finished
    - tool_called: Function tool invoked
    - agent_switched: Agent switching occurred
    """
    
    def __init__(
        self,
        agent: Agent | None = None,
        llm: LLM | None = None,
        conversational_graph: Any | None = None,
        context_window: Any | None = None,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.llm = llm
        self.conversational_graph = conversational_graph
        self.context_window = context_window
        self.llm_lock = asyncio.Lock()
        self._is_interrupted = False

    @property
    def max_tool_calls_per_turn(self) -> int:
        """Get max_tool_calls_per_turn from context_window, or default to 10."""
        if self.context_window and hasattr(self.context_window, 'max_tool_calls_per_turn'):
            return self.context_window.max_tool_calls_per_turn
        return 10
    
    async def start(self) -> None:
        """Start the content generation component"""
        logger.info("ContentGeneration started")
    
    async def generate(self, user_text: str, context_prefix: str | None = None) -> AsyncIterator[ResponseChunk]:
        """
        Process user text with LLM and yield response chunks.

        Args:
            user_text: User input text
            context_prefix: Optional temporary context (e.g. KB results) injected
                           for this LLM call only, not persisted in chat history.

        Yields:
            ResponseChunk objects with generated content
        """
        async with self.llm_lock:
            if not self.llm:
                logger.warning("No LLM available for content generation")
                return
            
            if not self.agent or not getattr(self.agent, "chat_context", None):
                logger.warning("Agent not available for LLM processing")
                return
            
            # Context window handles compression + truncation in one call
            if self.context_window and self.llm:
                await self.context_window.manage(self.agent.chat_context, self.llm)
            
            self.emit("generation_started", {
                "user_text": user_text,
                "context_size": len(self.agent.chat_context.items)
            })

            metrics_collector.on_llm_start()
            metrics_collector.set_llm_input(user_text)

            first_chunk_received = False
            _total_tool_calls = 0
            _max_total_tool_calls = self.max_tool_calls_per_turn
            _call_id_counter = 0 

            _prefix_original_content = None
            _prefix_target_msg = None
            if context_prefix:
                msgs = self.agent.chat_context.messages()
                _prefix_target_msg = next(
                    (m for m in reversed(msgs) if m.role == ChatRole.USER), None
                )
                if _prefix_target_msg:
                    _prefix_original_content = _prefix_target_msg.content
                    clean_text = (
                        _prefix_original_content[0]
                        if isinstance(_prefix_original_content, list)
                        else _prefix_original_content
                    )
                    _prefix_target_msg.content = [f"{context_prefix}\n\nUser: {clean_text}"]

            agent_session = getattr(self.agent, "session", None)
            if agent_session:
                agent_session._emit_user_state(UserState.IDLE)
                agent_session._emit_agent_state(AgentState.THINKING)

            async for llm_chunk_resp in self.llm.chat(
                self.agent.chat_context,
                tools=self.agent._tools,
                conversational_graph=self.conversational_graph if self.conversational_graph else None
            ):
                if llm_chunk_resp.metadata and "usage" in llm_chunk_resp.metadata:
                    metrics_collector.set_llm_usage(llm_chunk_resp.metadata["usage"])
                    self.emit("usage_tracked", llm_chunk_resp.metadata["usage"])

                if self._is_interrupted:
                    logger.info("LLM processing interrupted")
                    break

                if not self.agent or not getattr(self.agent, "chat_context", None):
                    logger.warning("Agent context unavailable, stopping LLM processing")
                    break

                if not first_chunk_received:
                    first_chunk_received = True
                    metrics_collector.on_llm_first_token()
                    self.emit("first_chunk", {})

                if llm_chunk_resp.metadata and "function_call" in llm_chunk_resp.metadata:
                    func_call = llm_chunk_resp.metadata["function_call"]

                    _total_tool_calls += 1
                    if _total_tool_calls > _max_total_tool_calls:
                        logger.warning(f"Tool call limit reached ({_max_total_tool_calls}), skipping")
                        continue

                    logger.info(f"Tool call: {func_call['name']} ({_total_tool_calls}/{_max_total_tool_calls})")

                    metrics_collector.add_function_tool_call(
                        tool_name=func_call["name"],
                        tool_params=func_call["arguments"],
                    )
                    self.emit("tool_called", {
                        "name": func_call["name"],
                        "arguments": func_call["arguments"]
                    })
                    
                    chat_context = getattr(self.agent, "chat_context", None)
                    if not chat_context:
                        logger.warning("Chat context missing while handling function call")
                        return

                    _call_id_counter += 1
                    func_call_id = func_call.get("call_id") or f"call_{int(time.time())}_{_call_id_counter}"

                    chat_context.add_function_call(
                        name=func_call["name"],
                        arguments=json.dumps(func_call["arguments"]),
                        call_id=func_call_id
                    )
                    
                    try:
                        if not self.agent:
                            logger.warning("Agent cleaned up before selecting tool")
                            return
                        
                        tool = next(
                            (t for t in self.agent.tools if is_function_tool(t) and get_tool_info(t).name == func_call["name"]),
                            None
                        )
                    except Exception as e:
                        logger.error(f"Error while selecting tool: {e}")
                        continue
                    
                    if tool:
                        agent_session = getattr(self.agent, "session", None)
                        if agent_session:
                            agent_session._is_executing_tool = True

                        try:
                            result = await tool(**func_call["arguments"])

                            if isinstance(result, Agent):
                                new_agent = result
                                current_session = self.agent.session

                                logger.info(f"Agent handoff: {type(self.agent).__name__} \u2192 {type(new_agent).__name__}")

                                chat_context.add_function_output(
                                    name=func_call["name"],
                                    output=f"Transferred to {type(new_agent).__name__}",
                                    call_id=func_call_id
                                )

                                if getattr(new_agent, 'inherit_context', True):
                                    new_agent.chat_context = self.agent.chat_context
                                    new_agent.chat_context.add_message(
                                        role=ChatRole.SYSTEM,
                                        content=new_agent.instructions,
                                        replace=True
                                    )

                                new_agent.session = current_session
                                self.agent = new_agent
                                current_session.agent = new_agent

                                if hasattr(current_session.pipeline, 'set_agent'):
                                    current_session.pipeline.set_agent(new_agent)

                                if hasattr(new_agent, 'on_enter') and asyncio.iscoroutinefunction(new_agent.on_enter):
                                    await new_agent.on_enter()

                                self.emit("agent_switched", {
                                    "old_agent": type(result).__name__,
                                    "new_agent": type(new_agent).__name__
                                })

                                return

                            chat_context = getattr(self.agent, "chat_context", None)
                            if not chat_context:
                                logger.warning("Chat context missing after tool execution")
                                return

                            chat_context.add_function_output(
                                name=func_call["name"],
                                output=json.dumps(result),
                                call_id=func_call_id
                            )

                            max_tool_rounds = _max_total_tool_calls
                            _tool_loop_text_yielded = False  
                            for _round in range(max_tool_rounds):
                                logger.debug(f"Post-tool LLM round {_round + 1}/{max_tool_rounds}")

                                pending_calls = []
                                buffered_text = []

                                async for new_resp in self.llm.chat(
                                    chat_context,
                                    tools=self.agent.tools,
                                    conversational_graph=self.conversational_graph if self.conversational_graph else None
                                ):
                                    if self._is_interrupted:
                                        break
                                    if not new_resp:
                                        continue

                                    if new_resp.metadata and "function_call" in new_resp.metadata:
                                        _total_tool_calls += 1
                                        if _total_tool_calls > _max_total_tool_calls:
                                            logger.warning(f"Tool call limit reached ({_max_total_tool_calls}), stopping")
                                            break
                                        next_call = new_resp.metadata["function_call"]
                                        _call_id_counter += 1
                                        next_call_id = next_call.get("call_id") or f"call_{int(time.time())}_{_call_id_counter}"
                                        logger.info(f"Chained tool call: {next_call['name']} (round {_round + 1}, total {_total_tool_calls}/{_max_total_tool_calls})")
                                        pending_calls.append((next_call, next_call_id))
                                    elif new_resp.content:
                                        buffered_text.append((new_resp.content, new_resp.metadata, new_resp.role))

                                if not pending_calls and buffered_text:
                                    _tool_loop_text_yielded = True
                                    for content, metadata, role in buffered_text:
                                        self.emit("generation_chunk", {"content": content, "metadata": metadata})
                                        yield ResponseChunk(content, metadata, role)
                                elif pending_calls and buffered_text:
                                    logger.debug("Discarding intermediate text (tool calls pending)")

                                if self._is_interrupted or not pending_calls:
                                    break

                                for next_call, next_call_id in pending_calls:
                                    chat_context.add_function_call(
                                        name=next_call["name"],
                                        arguments=json.dumps(next_call["arguments"]),
                                        call_id=next_call_id
                                    )

                                if len(pending_calls) == 1:
                                    next_call, next_call_id = pending_calls[0]
                                    next_tool = next(
                                        (t for t in self.agent.tools
                                         if is_function_tool(t) and get_tool_info(t).name == next_call["name"]),
                                        None
                                    )
                                    if next_tool:
                                        next_result = await next_tool(**next_call["arguments"])
                                        chat_context.add_function_output(
                                            name=next_call["name"],
                                            output=json.dumps(next_result),
                                            call_id=next_call_id
                                        )
                                    else:
                                        chat_context.add_function_output(
                                            name=next_call["name"],
                                            output=json.dumps({"error": f"Tool '{next_call['name']}' not found"}),
                                            call_id=next_call_id,
                                            is_error=True
                                        )
                                else:
                                    logger.info(f"Executing {len(pending_calls)} tools in parallel")

                                    async def _exec_tool(call_info):
                                        nc, nc_id = call_info
                                        t = next(
                                            (t for t in self.agent.tools
                                             if is_function_tool(t) and get_tool_info(t).name == nc["name"]),
                                            None
                                        )
                                        if t:
                                            return nc, nc_id, await t(**nc["arguments"]), False
                                        return nc, nc_id, {"error": f"Tool '{nc['name']}' not found"}, True

                                    results = await asyncio.gather(
                                        *[_exec_tool(pc) for pc in pending_calls],
                                        return_exceptions=True
                                    )

                                    for r in results:
                                        if isinstance(r, Exception):
                                            logger.error(f"Parallel tool error: {r}")
                                            continue
                                        nc, nc_id, output, is_err = r
                                        chat_context.add_function_output(
                                            name=nc["name"],
                                            output=json.dumps(output),
                                            call_id=nc_id,
                                            is_error=is_err
                                        )

                            last_item = chat_context.items[-1] if chat_context.items else None
                            if not self._is_interrupted and not _tool_loop_text_yielded and isinstance(last_item, FunctionCallOutput):
                                logger.info("Tool loop exhausted, forcing final text response")
                                async for final_resp in self.llm.chat(
                                    chat_context,
                                    tools=None,
                                    conversational_graph=self.conversational_graph if self.conversational_graph else None
                                ):
                                    if self._is_interrupted or not final_resp:
                                        break
                                    if final_resp.content:
                                        self.emit("generation_chunk", {
                                            "content": final_resp.content,
                                            "metadata": final_resp.metadata
                                        })
                                        yield ResponseChunk(final_resp.content, final_resp.metadata, final_resp.role)

                            if self._is_interrupted:
                                last_item = chat_context.items[-1] if chat_context.items else None
                                if isinstance(last_item, FunctionCallOutput):
                                    logger.info(f"Tool execution interrupted after '{last_item.name}', adding closure")
                                    msg = chat_context.add_message(
                                        role=ChatRole.ASSISTANT,
                                        content=f"[Used {last_item.name} tool — response interrupted]"
                                    )
                                    msg.interrupted = True

                        except Exception as e:
                            logger.error(f"Error executing function {func_call['name']}: {e}")
                            continue

                        finally:
                            if agent_session:
                                agent_session._is_executing_tool = False

                    if _tool_loop_text_yielded:
                        break
                else:
                    has_content = llm_chunk_resp and llm_chunk_resp.content
                    has_graph_response = (
                        llm_chunk_resp
                        and llm_chunk_resp.metadata
                        and llm_chunk_resp.metadata.get("graph_response")
                    )
                    if has_content or has_graph_response:
                        self.emit("generation_chunk", {
                            "content": llm_chunk_resp.content,
                            "metadata": llm_chunk_resp.metadata
                        })
                        yield ResponseChunk(llm_chunk_resp.content, llm_chunk_resp.metadata, llm_chunk_resp.role)

            if _prefix_target_msg and _prefix_original_content is not None:
                _prefix_target_msg.content = _prefix_original_content

            if not self._is_interrupted:
                metrics_collector.on_llm_complete()
                self.emit("generation_complete", {})
    
    def interrupt(self) -> None:
        """Interrupt the current generation"""
        self._is_interrupted = True
    
    def reset_interrupt(self) -> None:
        """Reset interrupt flag"""
        self._is_interrupted = False
    
    async def cancel(self) -> None:
        """Cancel LLM generation"""
        if self.llm:
            try:
                await self.llm.cancel_current_generation()
            except Exception as e:
                logger.error(f"LLM cancellation failed: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup content generation resources"""
        logger.info("Cleaning up content generation")
        
        self.llm = None
        self.agent = None
        self.conversational_graph = None
        
        logger.info("Content generation cleaned up")