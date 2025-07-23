from __future__ import annotations

from typing import Any, Callable, Optional
import asyncio

from .agent import Agent
from .llm.chat_context import ChatRole
from .conversation_flow import ConversationFlow
from .pipeline import Pipeline

class AgentSession:
    """
    Manages an agent session with its associated conversation flow and pipeline.
    """
    
    def __init__(
        self,
        agent: Agent,
        pipeline: Pipeline,
        conversation_flow: Optional[ConversationFlow] = None,
        wake_up: Optional[int] = None,
    ) -> None:
        """
        Initialize an agent session.
        
        Args:
            agent: Instance of an Agent class that handles the core logic
            pipeline: Pipeline instance to process the agent's operations
            conversation_flow: ConversationFlow instance to manage conversation state
            wake_up: Time in seconds after which to trigger wake-up callback if no speech detected
        """
        self.agent = agent
        self.pipeline = pipeline
        self.conversation_flow = conversation_flow
        self.agent.session = self
        self.wake_up = wake_up
        self.on_wake_up: Optional[Callable[[], None] | Callable[[], Any]] = None
        self._wake_up_task: Optional[asyncio.Task] = None
        self._wake_up_timer_active = False
        
        if hasattr(self.pipeline, 'set_agent'):
            self.pipeline.set_agent(self.agent)
        if hasattr(self.pipeline, 'set_conversation_flow') and self.conversation_flow is not None:
            self.pipeline.set_conversation_flow(self.conversation_flow)
        if hasattr(self.pipeline, 'set_wake_up_callback'):
            self.pipeline.set_wake_up_callback(self._reset_wake_up_timer)

    def _start_wake_up_timer(self) -> None:
        if self.wake_up is not None and self.on_wake_up is not None:
            self._wake_up_timer_active = True
            self._wake_up_task = asyncio.create_task(self._wake_up_timer_loop())
    
    def _reset_wake_up_timer(self) -> None:
        if self.wake_up is not None and self.on_wake_up is not None:
            if self._wake_up_task and not self._wake_up_task.done():
                self._wake_up_task.cancel()
            if self._wake_up_timer_active:
                self._wake_up_task = asyncio.create_task(self._wake_up_timer_loop())
    
    def _cancel_wake_up_timer(self) -> None:
        if self._wake_up_task and not self._wake_up_task.done():
            self._wake_up_task.cancel()
        self._wake_up_timer_active = False
    
    async def _wake_up_timer_loop(self) -> None:
        try:
            await asyncio.sleep(self.wake_up)
            if self._wake_up_timer_active and self.on_wake_up:
                if asyncio.iscoroutinefunction(self.on_wake_up):
                    await self.on_wake_up()
                else:
                    self.on_wake_up()
        except asyncio.CancelledError:
            pass

    async def start(self, **kwargs: Any) -> None:
        """
        Start the agent session.
        This will:
        1. Initialize the agent (including MCP tools if configured)
        2. Call the agent's on_enter hook
        3. Start the pipeline processing
        4. Start wake-up timer if configured (but only if callback is set)
        
        Args:
            **kwargs: Additional arguments to pass to the pipeline start method
        """       
        await self.agent.initialize_mcp()
        if hasattr(self.pipeline, 'set_agent'):
            self.pipeline.set_agent(self.agent)
        
        await self.pipeline.start()
        await self.agent.on_enter()
        if self.on_wake_up is not None:
            self._start_wake_up_timer()
        
    async def say(self, message: str) -> None:
        """
        Send an initial message to the agent.
        """
        self.agent.chat_context.add_message(role=ChatRole.ASSISTANT, content=message)
        await self.pipeline.send_message(message)
    
    async def close(self) -> None:
        """
        Close the agent session.
        """
        self._cancel_wake_up_timer()
        await self.agent.on_exit()
        await self.pipeline.cleanup()
    
    async def leave(self) -> None:
        """
        Leave the agent session.
        """
        await self.pipeline.leave()
        