from __future__ import annotations

from typing import AsyncIterator, Literal, TYPE_CHECKING, Any
import asyncio
import json
import time
import logging
from .event_emitter import EventEmitter
from .llm.llm import LLM, ResponseChunk
from .llm.chat_context import ChatRole
from .utils import is_function_tool, get_tool_info, UserState, AgentState
from .agent import Agent

if TYPE_CHECKING:
    from .knowledge_base.base import KnowledgeBase

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
        max_context_items: int | None = None,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.llm = llm
        self.conversational_graph = conversational_graph
        self.max_context_items = max_context_items
        self.llm_lock = asyncio.Lock()
        self._is_interrupted = False
    
    async def start(self) -> None:
        """Start the content generation component"""
        logger.info("[start] ContentGeneration started")
    
    async def generate(self, user_text: str) -> AsyncIterator[ResponseChunk]:
        """
        Process user text with LLM and yield response chunks.
        
        Args:
            user_text: User input text
            knowledge_base: Optional knowledge base for context enrichment
            
        Yields:
            ResponseChunk objects with generated content
        """
        async with self.llm_lock:
            if not self.llm:
                logger.warning("[generate] No LLM available for content generation")
                return
            
            if not self.agent or not getattr(self.agent, "chat_context", None):
                logger.warning("[generate] Agent not available for LLM processing")
                return
            
            if self.max_context_items:
                current_items = len(self.agent.chat_context.items)
                if current_items > self.max_context_items:
                    try:
                        logger.info(f"[generate] Truncating context from {current_items} to {self.max_context_items} items")
                        self.agent.chat_context.truncate(self.max_context_items)
                        logger.info(f"[generate] Truncation complete. Final size: {len(self.agent.chat_context.items)} items")
                    except Exception as e:
                        logger.error(f"[generate] Error during truncation: {e}", exc_info=True)
            
            self.emit("generation_started", {
                "user_text": user_text,
                "context_size": len(self.agent.chat_context.items)
            })
            
            first_chunk_received = False
            
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
                    self.emit("usage_tracked", llm_chunk_resp.metadata["usage"])
                
                if self._is_interrupted:
                    logger.info("[generate][CONTENT_GENERATION] - LLM processing interrupted")
                    break
                
                if not self.agent or not getattr(self.agent, "chat_context", None):
                    logger.warning("[generate][CONTENT_GENERATION] Agent context unavailable, stopping LLM processing")
                    break
                
                if not first_chunk_received:
                    first_chunk_received = True
                    self.emit("first_chunk", {})
                
                if llm_chunk_resp.metadata and "function_call" in llm_chunk_resp.metadata:
                    func_call = llm_chunk_resp.metadata["function_call"]
                    
                    self.emit("tool_called", {
                        "name": func_call["name"],
                        "arguments": func_call["arguments"]
                    })
                    
                    chat_context = getattr(self.agent, "chat_context", None)
                    if not chat_context:
                        logger.warning("[generate] Chat context missing while handling function call")
                        return
                    
                    chat_context.add_function_call(
                        name=func_call["name"],
                        arguments=json.dumps(func_call["arguments"]),
                        call_id=func_call.get("call_id", f"call_{int(time.time())}")
                    )
                    
                    try:
                        if not self.agent:
                            logger.warning("[generate] Agent cleaned up before selecting tool")
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
                                
                                logger.info(f"[generate] Switching from {type(self.agent).__name__} to {type(new_agent).__name__}")
                                
                                if getattr(new_agent, 'inherit_context', True):
                                    logger.info(f"[generate] Inheriting context from {type(self.agent).__name__}")
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
                                logger.warning("[generate] Chat context missing after tool execution")
                                return
                            
                            chat_context.add_function_output(
                                name=func_call["name"],
                                output=json.dumps(result),
                                call_id=func_call.get("call_id", f"call_{int(time.time())}")
                            )
                            
                            async for new_resp in self.llm.chat(
                                chat_context,
                                tools=self.agent.tools,
                                conversational_graph=self.conversational_graph if self.conversational_graph else None
                            ):
                                if self._is_interrupted:
                                    break
                                if new_resp:
                                    self.emit("generation_chunk", {
                                        "content": new_resp.content,
                                        "metadata": new_resp.metadata
                                    })
                                    yield ResponseChunk(new_resp.content, new_resp.metadata, new_resp.role)
                        
                        except Exception as e:
                            logger.error(f"[generate] Error executing function {func_call['name']}: {e}")
                            continue
                        
                        finally:
                            if agent_session:
                                agent_session._is_executing_tool = False
                else:
                    if llm_chunk_resp:
                        self.emit("generation_chunk", {
                            "content": llm_chunk_resp.content,
                            "metadata": llm_chunk_resp.metadata
                        })
                        yield ResponseChunk(llm_chunk_resp.content, llm_chunk_resp.metadata, llm_chunk_resp.role)
            
            if not self._is_interrupted:
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
        logger.debug("[cleanup] Cleaning up content generation")
        
        self.llm = None
        self.agent = None
        self.conversational_graph = None
        
        logger.info("[cleanup] Content generation cleaned up")
