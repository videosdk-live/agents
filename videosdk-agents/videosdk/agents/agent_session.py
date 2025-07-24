from __future__ import annotations

from typing import Any, Optional

from .agent import Agent
from .llm.chat_context import ChatMessage, ChatRole
from .conversation_flow import ConversationFlow
from .pipeline import Pipeline
import os
from .metrics import metrics_collector
from .metrics.realtime_collector import realtime_metrics_collector
from .realtime_pipeline import RealTimePipeline


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
        await self.agent.initialize_mcp()

        if isinstance(self.pipeline, RealTimePipeline):
            await realtime_metrics_collector.start_session(self.agent, self.pipeline)
        else:
            traces_flow_manager = metrics_collector.traces_flow_manager
            if traces_flow_manager:
                config_attributes = {
                    "system_instructions": self.agent.instructions,
                    "function_tools": [
                        getattr(tool, "name", tool.__name__ if callable(tool) else str(tool))
                        for tool in (
                            [tool for tool in self.agent.tools if tool not in self.agent.mcp_manager.tools]
                            if self.agent.mcp_manager else self.agent.tools
                        )
                    ] if self.agent.tools else [],

                    "mcp_tools": [
                        tool._tool_info.name
                        for tool in self.agent.mcp_manager.tools
                    ] if self.agent.mcp_manager else [],

                    "pipeline": self.pipeline.__class__.__name__,
                    **({
                        "stt_provider": self.pipeline.stt.__class__.__name__ if self.pipeline.stt else None,
                        "tts_provider": self.pipeline.tts.__class__.__name__ if self.pipeline.tts else None, 
                        "llm_provider": self.pipeline.llm.__class__.__name__ if self.pipeline.llm else None,
                        "stt_model": self.pipeline.get_component_configs()['stt'].get('model') if hasattr(self.pipeline, 'get_component_configs') and self.pipeline.stt else None,
                        "llm_model": self.pipeline.get_component_configs()['llm'].get('model') if hasattr(self.pipeline, 'get_component_configs') and self.pipeline.llm else None,
                        "tts_model": self.pipeline.get_component_configs()['tts'].get('model') if hasattr(self.pipeline, 'get_component_configs') and self.pipeline.tts else None
                    } if self.pipeline.__class__.__name__ == "CascadingPipeline" else {}),
                }
                await traces_flow_manager.start_agent_session_config(config_attributes)
                await traces_flow_manager.start_agent_session({})

            if self.pipeline.__class__.__name__ == "CascadingPipeline":
                configs = self.pipeline.get_component_configs() if hasattr(self.pipeline, 'get_component_configs') else {}
                metrics_collector.set_provider_info(
                    llm_provider=self.pipeline.llm.__class__.__name__ if self.pipeline.llm else "",
                    llm_model=configs.get('llm', {}).get('model', "") if self.pipeline.llm else "",
                    stt_provider=self.pipeline.stt.__class__.__name__ if self.pipeline.stt else "",
                    stt_model=configs.get('stt', {}).get('model', "") if self.pipeline.stt else "",
                    tts_provider=self.pipeline.tts.__class__.__name__ if self.pipeline.tts else "",
                    tts_model=configs.get('tts', {}).get('model', "") if self.pipeline.tts else ""
                )
        
        if hasattr(self.pipeline, 'set_agent'):
            self.pipeline.set_agent(self.agent)
        
        await self.pipeline.start()
        await self.agent.on_enter()
        
    async def say(self, message: str) -> None:
        """
        Send an initial message to the agent.
        """
        if not isinstance(self.pipeline, RealTimePipeline):
            traces_flow_manager = metrics_collector.traces_flow_manager
            if traces_flow_manager:
                traces_flow_manager.agent_say_called(message)
        self.agent.chat_context.add_message(role=ChatRole.ASSISTANT, content=message)
        await self.pipeline.send_message(message)
    
    async def close(self) -> None:
        """
        Close the agent session.
        """
        if isinstance(self.pipeline, RealTimePipeline):
            realtime_metrics_collector.finalize_session()
            traces_flow_manager = realtime_metrics_collector.traces_flow_manager
            if traces_flow_manager:
                await traces_flow_manager.start_agent_session_closed({})
                traces_flow_manager.end_agent_session_closed()
        else:
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
        