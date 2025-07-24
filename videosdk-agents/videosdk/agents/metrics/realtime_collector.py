from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from dataclasses import asdict

from .models import RealtimeInteractionData, TimelineEvent
from .analytics import AnalyticsClient
from .traces_flow import TracesFlowManager


if TYPE_CHECKING:
    from videosdk.agents.agent import Agent
    from videosdk.agents.pipeline import Pipeline


class RealtimeMetricsCollector:

    _agent_info: Dict[str, Any] = {
        "provider_class_name": None,
        "provider_model_name": None,
        "system_instructions": None,
        "function_tools": [],
        "mcp_tools": []
    }

    def __init__(self) -> None:
        self.interactions: List[RealtimeInteractionData] = []
        self.current_interaction: Optional[RealtimeInteractionData] = None
        self.lock = asyncio.Lock()
        self.agent_speech_end_timer: Optional[asyncio.TimerHandle] = None
        self.analytics_client = AnalyticsClient()
        self.session_id: Optional[str] = None
        self.traces_flow_manager: Optional[TracesFlowManager] = None

    def set_session_id(self, session_id: str):
        """Set the session ID for metrics tracking"""
        self.session_id = session_id
        self.analytics_client.set_session_id(session_id)

    def set_traces_flow_manager(self, manager: TracesFlowManager):
        """Set the TracesFlowManager instance for realtime tracing"""
        self.traces_flow_manager = manager
        
    def _transform_to_camel_case(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Converts snake_case to camelCase for analytics reporting."""
        
        def to_camel_case(snake_str: str) -> str:
            parts = snake_str.split('_')
            return parts[0] + ''.join(x.title() for x in parts[1:])

        if isinstance(data, list):
            return [self._transform_to_camel_case(item) for item in data]
        if isinstance(data, dict):
            return {to_camel_case(k): self._transform_to_camel_case(v) for k, v in data.items()}
        return data

    async def start_session(self, agent: Agent, pipeline: Pipeline) -> None:
        RealtimeMetricsCollector._agent_info = {
            "provider_class_name": pipeline.model.__class__.__name__,
            "provider_model_name": getattr(pipeline.model, "model", None),
            "system_instructions": agent.instructions,
            "function_tools": [
                getattr(tool, "name", tool.__name__ if callable(tool) else str(tool))
                for tool in (
                    [tool for tool in agent.tools if tool not in agent.mcp_manager.tools]
                    if agent.mcp_manager else agent.tools
                )
            ] if agent.tools else [],
            "mcp_tools": [
                tool._tool_info.name
                for tool in agent.mcp_manager.tools
            ] if agent.mcp_manager else [],
        }
        self.interactions = []
        self.current_interaction = None
        from . import metrics_collector as _metrics_collector

        traces_flow_manager = _metrics_collector.traces_flow_manager
        if traces_flow_manager:
            self.set_traces_flow_manager(traces_flow_manager)

            config_attributes = {
                **RealtimeMetricsCollector._agent_info,
                "pipeline": pipeline.__class__.__name__,
                "llm_provider": pipeline.model.__class__.__name__,
            }

            await traces_flow_manager.start_agent_session_config(config_attributes)
            await traces_flow_manager.start_agent_session({})

    async def _start_new_interaction(self) -> None:
        async with self.lock:
            self.current_interaction = RealtimeInteractionData(
                interaction_id=str(uuid.uuid4()),
                **RealtimeMetricsCollector._agent_info
            )
            self.interactions.append(self.current_interaction)

    async def set_user_speech_start(self) -> None:
        if self.current_interaction:
            self._finalize_interaction_and_send()
        
        await self._start_new_interaction()
        if self.current_interaction and self.current_interaction.user_speech_start_time is None:
            self.current_interaction.user_speech_start_time = time.perf_counter()
            await self.start_timeline_event("user_speech")

    async def set_user_speech_end(self) -> None:
        if self.current_interaction and self.current_interaction.user_speech_end_time is None:
            self.current_interaction.user_speech_end_time = time.perf_counter()
            await self.end_timeline_event("user_speech")

    async def set_agent_speech_start(self) -> None:
        if not self.current_interaction:
            await self._start_new_interaction()
        elif self.current_interaction.user_speech_start_time is not None and self.current_interaction.user_speech_end_time is None:
            self.current_interaction.user_speech_end_time = time.perf_counter()
            await self.end_timeline_event("user_speech")

        if self.current_interaction and self.current_interaction.agent_speech_start_time is None:
            self.current_interaction.agent_speech_start_time = time.perf_counter()
            await self.start_timeline_event("agent_speech")
            if self.agent_speech_end_timer:
                self.agent_speech_end_timer.cancel()

    async def set_agent_speech_end(self, timeout: float = 1.0) -> None:
        if self.current_interaction:
            if self.agent_speech_end_timer:
                self.agent_speech_end_timer.cancel()
            
            loop = asyncio.get_event_loop()
            self.agent_speech_end_timer = loop.call_later(timeout, self._finalize_interaction_and_send)
            await self.end_timeline_event("agent_speech")

    def _finalize_agent_speech(self) -> None:
        if not self.current_interaction or self.current_interaction.agent_speech_end_time is not None:
            return
        
        is_valid_interaction = self.current_interaction.user_speech_start_time is not None or \
                               self.current_interaction.agent_speech_start_time is not None

        if not is_valid_interaction:
            return

        self.current_interaction.agent_speech_end_time = time.perf_counter()
        self.agent_speech_end_timer = None

    def _finalize_interaction_and_send(self) -> None:
        if not self.current_interaction:
            return

        self._finalize_agent_speech()

        if self.current_interaction.user_speech_start_time and not self.current_interaction.user_speech_end_time:
            self.current_interaction.user_speech_end_time = time.perf_counter()

        current_time = time.perf_counter()
        for event in self.current_interaction.timeline:
            if event.end_time is None:
                event.end_time = current_time
                event.duration_ms = round((current_time - event.start_time) * 1000, 4)

        is_valid_interaction = self.current_interaction.user_speech_start_time is not None or \
                               self.current_interaction.agent_speech_start_time is not None

        if not is_valid_interaction:
            self.current_interaction = None
            return

        self.current_interaction.compute_latencies()
        if not hasattr(self, 'traces_flow_manager') or self.traces_flow_manager is None:
            try:
                from . import metrics_collector as _metrics_collector
                self.traces_flow_manager = _metrics_collector.traces_flow_manager
            except Exception:
                pass
        if self.traces_flow_manager:
            self.traces_flow_manager.create_realtime_interaction_trace(self.current_interaction)
        interaction_data = asdict(self.current_interaction)
        
        if len(self.interactions) > 1:
            fields_to_remove = [
                "provider_class_name", "provider_model_name", "system_instructions",
                "function_tools", "mcp_tools"
            ]
            for field in fields_to_remove:
                if field in interaction_data:
                    del interaction_data[field]
                    
        transformed_data = self._transform_to_camel_case(interaction_data)
        self.analytics_client.send_interaction_analytics_safe({
            "sessionId": self.session_id,
            "data": [transformed_data]
        })
        self.interactions.append(self.current_interaction)
        self.current_interaction = None

    async def add_timeline_event(self, event: TimelineEvent) -> None:
        if self.current_interaction:
            self.current_interaction.timeline.append(event)

    async def start_timeline_event(self, event_type: str) -> None:
        """Start a timeline event with a precise start time"""
        if self.current_interaction:
            event = TimelineEvent(
                event_type=event_type,
                start_time=time.perf_counter()
            )
            self.current_interaction.timeline.append(event)

    async def end_timeline_event(self, event_type: str) -> None:
        """End a timeline event and calculate duration"""
        if self.current_interaction:
            end_time = time.perf_counter()
            for event in reversed(self.current_interaction.timeline):
                if event.event_type == event_type and event.end_time is None:
                    event.end_time = end_time
                    event.duration_ms = round((end_time - event.start_time) * 1000, 4)
                    break

    async def update_timeline_event_text(self, event_type: str, text: str) -> None:
        """Update timeline event with text content"""
        if self.current_interaction:
            for event in reversed(self.current_interaction.timeline):
                if event.event_type == event_type and not event.text:
                    event.text = text
                    break

    async def add_tool_call(self, tool_name: str) -> None:
        if self.current_interaction and tool_name not in self.current_interaction.function_tools_called:
            self.current_interaction.function_tools_called.append(tool_name)

    async def set_user_transcript(self, text: str) -> None:
        """Set the user transcript for the current interaction and update timeline"""
        if self.current_interaction:
            if self.current_interaction.user_speech_start_time is None:
                self.current_interaction.user_speech_start_time = time.perf_counter()
                await self.start_timeline_event("user_speech")
            if self.current_interaction.user_speech_end_time is None:
                self.current_interaction.user_speech_end_time = time.perf_counter()
                await self.end_timeline_event("user_speech")
            await self.update_timeline_event_text("user_speech", text)

    async def set_agent_response(self, text: str) -> None:
        """Set the agent response for the current interaction and update timeline"""
        if self.current_interaction:
            if self.current_interaction.agent_speech_start_time is None:
                self.current_interaction.agent_speech_start_time = time.perf_counter()
                await self.start_timeline_event("agent_speech")
            await self.update_timeline_event_text("agent_speech", text)

    def set_realtime_model_error(self, error: Dict[str, Any]) -> None:
        """Set a realtime model-specific error for the current interaction"""
        if self.current_interaction:
            self.current_interaction.realtime_model_errors.append(error)

    async def set_interrupted(self) -> None:
        if self.current_interaction:
            self.current_interaction.interrupted = True

    def finalize_session(self) -> None:
        asyncio.run_coroutine_threadsafe(self._start_new_interaction(), asyncio.get_event_loop())

realtime_metrics_collector = RealtimeMetricsCollector() 