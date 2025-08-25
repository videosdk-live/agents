from __future__ import annotations
from typing import Any, Callable, Optional
import asyncio

from .agent import Agent
from .llm.chat_context import ChatRole
from .conversation_flow import ConversationFlow
from .pipeline import Pipeline
from .metrics import cascading_metrics_collector, realtime_metrics_collector
from .realtime_pipeline import RealTimePipeline
from .utils import get_tool_info
import time
from .job import get_current_job_context
from .event_bus import global_event_emitter
import logging
logger = logging.getLogger(__name__)
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
        self._closed: bool = False
        if hasattr(self.pipeline, 'set_agent'):
            self.pipeline.set_agent(self.agent)
        if (
            hasattr(self.pipeline, "set_conversation_flow")
            and self.conversation_flow is not None
        ):
            self.pipeline.set_conversation_flow(self.conversation_flow)
        if hasattr(self.pipeline, 'set_wake_up_callback'):
            self.pipeline.set_wake_up_callback(self._reset_wake_up_timer)

        try:
            job_ctx = get_current_job_context()
            if job_ctx:
                job_ctx.add_shutdown_callback(self.close)
        except Exception:
            pass

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

        if isinstance(self.pipeline, RealTimePipeline):
            await realtime_metrics_collector.start_session(self.agent, self.pipeline)
        else:
            traces_flow_manager = cascading_metrics_collector.traces_flow_manager
            if traces_flow_manager:
                config_attributes = {
                    "system_instructions": self.agent.instructions,
                    "function_tools": [
                        get_tool_info(tool).name
                        for tool in (
                            [tool for tool in self.agent.tools if tool not in self.agent.mcp_manager.tools]
                            if self.agent.mcp_manager else self.agent.tools
                        )
                    ] if self.agent.tools else [],

                    "mcp_tools": [
                        get_tool_info(tool).name
                        for tool in self.agent.mcp_manager.tools
                    ] if self.agent.mcp_manager else [],

                    "pipeline": self.pipeline.__class__.__name__,
                    **({
                        "stt_provider": self.pipeline.stt.__class__.__name__ if self.pipeline.stt else None,
                        "tts_provider": self.pipeline.tts.__class__.__name__ if self.pipeline.tts else None, 
                        "llm_provider": self.pipeline.llm.__class__.__name__ if self.pipeline.llm else None,
                        "vad_provider": self.pipeline.vad.__class__.__name__ if hasattr(self.pipeline, 'vad') and self.pipeline.vad else None,
                        "eou_provider": self.pipeline.turn_detector.__class__.__name__ if hasattr(self.pipeline, 'turn_detector') and self.pipeline.turn_detector else None,
                        "stt_model": self.pipeline.get_component_configs()['stt'].get('model') if hasattr(self.pipeline, 'get_component_configs') and self.pipeline.stt else None,
                        "llm_model": self.pipeline.get_component_configs()['llm'].get('model') if hasattr(self.pipeline, 'get_component_configs') and self.pipeline.llm else None,
                        "tts_model": self.pipeline.get_component_configs()['tts'].get('model') if hasattr(self.pipeline, 'get_component_configs') and self.pipeline.tts else None,
                        "vad_model": self.pipeline.get_component_configs()['vad'].get('model') if hasattr(self.pipeline, 'get_component_configs') and hasattr(self.pipeline, 'vad') and self.pipeline.vad else None,
                        "eou_model": self.pipeline.get_component_configs()['eou'].get('model') if hasattr(self.pipeline, 'get_component_configs') and hasattr(self.pipeline, 'turn_detector') and self.pipeline.turn_detector else None
                    } if self.pipeline.__class__.__name__ == "CascadingPipeline" else {}),
                }
                start_time = time.perf_counter()
                config_attributes["start_time"] = start_time
                await traces_flow_manager.start_agent_session_config(config_attributes)
                await traces_flow_manager.start_agent_session({"start_time": start_time})

            if self.pipeline.__class__.__name__ == "CascadingPipeline":
                configs = self.pipeline.get_component_configs() if hasattr(self.pipeline, 'get_component_configs') else {}
                cascading_metrics_collector.set_provider_info(
                    llm_provider=self.pipeline.llm.__class__.__name__ if self.pipeline.llm else "",
                    llm_model=configs.get('llm', {}).get('model', "") if self.pipeline.llm else "",
                    stt_provider=self.pipeline.stt.__class__.__name__ if self.pipeline.stt else "",
                    stt_model=configs.get('stt', {}).get('model', "") if self.pipeline.stt else "",
                    tts_provider=self.pipeline.tts.__class__.__name__ if self.pipeline.tts else "",
                    tts_model=configs.get('tts', {}).get('model', "") if self.pipeline.tts else "",
                    vad_provider=self.pipeline.vad.__class__.__name__ if hasattr(self.pipeline, 'vad') and self.pipeline.vad else "",
                    vad_model=configs.get('vad', {}).get('model', "") if hasattr(self.pipeline, 'vad') and self.pipeline.vad else "",
                    eou_provider=self.pipeline.turn_detector.__class__.__name__ if hasattr(self.pipeline, 'turn_detector') and self.pipeline.turn_detector else "",
                    eou_model=configs.get('eou', {}).get('model', "") if hasattr(self.pipeline, 'turn_detector') and self.pipeline.turn_detector else ""
                )
        
        if hasattr(self.pipeline, 'set_agent'):
            self.pipeline.set_agent(self.agent)

        await self.pipeline.start()
        await self.agent.on_enter()
        global_event_emitter.emit("AGENT_STARTED", {"session": self})
        if self.on_wake_up is not None:
            self._start_wake_up_timer()
    async def say(self, message: str) -> None:
        """
        Send an initial message to the agent.
        """
        if not isinstance(self.pipeline, RealTimePipeline):
            traces_flow_manager = cascading_metrics_collector.traces_flow_manager
            if traces_flow_manager:
                traces_flow_manager.agent_say_called(message)
        self.agent.chat_context.add_message(role=ChatRole.ASSISTANT, content=message)
        await self.pipeline.send_message(message)

    async def close(self) -> None:
        """
        Close the agent session.
        """
        if self._closed:
            logger.info("Agent session already closed")
            return
        self._closed = True
        if isinstance(self.pipeline, RealTimePipeline):
            realtime_metrics_collector.finalize_session()
            traces_flow_manager = realtime_metrics_collector.traces_flow_manager
            if traces_flow_manager:
                start_time = time.perf_counter()
                await traces_flow_manager.start_agent_session_closed({"start_time": start_time})
                traces_flow_manager.end_agent_session_closed()
        else:
            traces_flow_manager = cascading_metrics_collector.traces_flow_manager
            if traces_flow_manager:
                start_time = time.perf_counter()
                await traces_flow_manager.start_agent_session_closed({"start_time": start_time})
                traces_flow_manager.end_agent_session_closed()

        self._cancel_wake_up_timer()
        await self.agent.on_exit()
        await self.pipeline.cleanup()

    async def leave(self) -> None:
        """
        Leave the agent session.
        """
        await self.pipeline.leave()
