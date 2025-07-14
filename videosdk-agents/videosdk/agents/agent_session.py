from __future__ import annotations

from typing import Any, Optional

from .agent import Agent
from .llm.chat_context import ChatMessage, ChatRole
from .conversation_flow import ConversationFlow
from .pipeline import Pipeline
import os
from .metrics import metrics_collector

class AgentSession:
    """
    Manages an agent session with its associated conversation flow and pipeline.
    """
    
    def __init__(
        self,
        agent: Agent,
        pipeline: Pipeline,
        conversation_flow: Optional[ConversationFlow] = None,
    ) -> None:
        """
        Initialize an agent session.
        
        Args:
            agent: Instance of an Agent class that handles the core logic
            flow: ConversationFlow instance to manage conversation state
            pipeline: Pipeline instance to process the agent's operations
        """
        self.agent = agent
        self.pipeline = pipeline
        self.conversation_flow = conversation_flow
        self.agent.session = self
        
        if hasattr(self.pipeline, 'set_agent'):
            self.pipeline.set_agent(self.agent)
        if hasattr(self.pipeline, 'set_conversation_flow') and self.conversation_flow is not None:
            self.pipeline.set_conversation_flow(self.conversation_flow)

    async def start(self, **kwargs: Any) -> None:
        """
        Start the agent session.
        This will:
        1. Initialize the agent (including MCP tools if configured)
        2. Call the agent's on_enter hook
        3. Start the pipeline processing
        
        Args:
            **kwargs: Additional arguments to pass to the pipeline start method
        Raises:
        ValueError: If meetingId is not provided in the context
        """       
        traces_flow_manager = metrics_collector.traces_flow_manager
        if traces_flow_manager:
            config_attributes = {
                "system_instructions": self.agent.instructions,
                "tools": [tool.name for tool in self.agent.tools] if self.agent.tools else [],
                "pipeline": self.pipeline.__class__.__name__,
            }
            if hasattr(self.pipeline, 'stt') and self.pipeline.stt:
                config_attributes["stt_provider"] = self.pipeline.stt.label
            if hasattr(self.pipeline, 'llm') and self.pipeline.llm:
                config_attributes["llm_provider"] = self.pipeline.llm.label
            if hasattr(self.pipeline, 'tts') and self.pipeline.tts:
                config_attributes["tts_provider"] = self.pipeline.tts.label
            

            await traces_flow_manager.start_agent_session_config(config_attributes)
            await traces_flow_manager.start_agent_session({})

        await self.agent.initialize_mcp()
        if hasattr(self.pipeline, 'set_agent'):
            self.pipeline.set_agent(self.agent)
        
        await self.pipeline.start()
        await self.agent.on_enter()
        
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
        traces_flow_manager = metrics_collector.traces_flow_manager
        if traces_flow_manager:
            await traces_flow_manager.start_agent_session_closed({})
            traces_flow_manager.end_agent_session_closed()

        await self.agent.on_exit()
        await self.pipeline.cleanup()
    
    async def leave(self) -> None:
        """
        Leave the agent session.
        """
        await self.pipeline.leave()
        