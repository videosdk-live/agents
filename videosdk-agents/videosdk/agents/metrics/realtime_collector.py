from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from dataclasses import asdict

from .models import RealtimeInteractionData, TimelineEvent
from .analytics import AnalyticsClient


if TYPE_CHECKING:
    from videosdk.agents.agent import Agent
    from videosdk.agents.pipeline import Pipeline


class RealtimeMetricsCollector:
    def __init__(self) -> None:
        self.interactions: List[RealtimeInteractionData] = []
        self.current_interaction: Optional[RealtimeInteractionData] = None
        self.agent_info: Dict[str, Any] = {}
        self.lock = asyncio.Lock()
        self.agent_speech_end_timer: Optional[asyncio.TimerHandle] = None
        self.analytics_client = AnalyticsClient()
        self.session_id: Optional[str] = None

    def set_session_id(self, session_id: str):
        """Set the session ID for metrics tracking"""
        self.session_id = session_id
        self.analytics_client.set_session_id(session_id)

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

    def start_session(self, agent: Agent, pipeline: Pipeline) -> None:
        self.agent_info = {
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

    async def _start_new_interaction(self) -> None:
        async with self.lock:
            if self.current_interaction:
                if self.current_interaction.user_speech_start_time and not self.current_interaction.agent_speech_end_time:
                    self.current_interaction.agent_speech_end_time = time.perf_counter()
                
                self.current_interaction.compute_latencies()
                interaction_data = asdict(self.current_interaction)
                transformed_data = self._transform_to_camel_case(interaction_data)
                self.analytics_client.send_interaction_analytics_safe({
                    "sessionId": self.session_id,
                    "data": [transformed_data]
                })

            self.current_interaction = RealtimeInteractionData(
                interaction_id=str(uuid.uuid4()),
                **self.agent_info
            )
            self.interactions.append(self.current_interaction)

    async def set_user_speech_start(self) -> None:
        await self._start_new_interaction()
        if self.current_interaction and self.current_interaction.user_speech_start_time is None:
            self.current_interaction.user_speech_start_time = time.perf_counter()

    async def set_user_speech_end(self) -> None:
        if self.current_interaction:
            self.current_interaction.user_speech_end_time = time.perf_counter()

    async def set_agent_speech_start(self) -> None:
        if self.current_interaction and self.current_interaction.agent_speech_start_time is None:
            self.current_interaction.agent_speech_start_time = time.perf_counter()
            if self.agent_speech_end_timer:
                self.agent_speech_end_timer.cancel()

    async def set_agent_speech_end(self, timeout: float = 1.0) -> None:
        if self.current_interaction:
            if self.agent_speech_end_timer:
                self.agent_speech_end_timer.cancel()
            
            loop = asyncio.get_event_loop()
            self.agent_speech_end_timer = loop.call_later(timeout, self._finalize_agent_speech)

    def _finalize_agent_speech(self) -> None:
        if not self.current_interaction or self.current_interaction.agent_speech_end_time is not None:
            return
        if self.current_interaction.user_speech_start_time is None:
            return
        self.current_interaction.agent_speech_end_time = time.perf_counter()
        self.agent_speech_end_timer = None
        
        self.current_interaction.compute_latencies()
        interaction_data = asdict(self.current_interaction)
        transformed_data = self._transform_to_camel_case(interaction_data)
        self.analytics_client.send_interaction_analytics_safe({
            "sessionId": self.session_id,
            "data": [transformed_data]
        })

        self.current_interaction = None

    async def add_timeline_event(self, event: TimelineEvent) -> None:
        if self.current_interaction:
            self.current_interaction.timeline.append(event)
    
    async def add_tool_call(self, tool_name: str) -> None:
        if self.current_interaction and tool_name not in self.current_interaction.function_tools_called:
            self.current_interaction.function_tools_called.append(tool_name)

    async def set_interrupted(self) -> None:
        if self.current_interaction:
            self.current_interaction.interrupted = True

    def finalize_session(self) -> None:
        if self.current_interaction:
            if self.current_interaction.user_speech_start_time and not self.current_interaction.agent_speech_end_time:
                self.current_interaction.agent_speech_end_time = time.perf_counter()
            
            self.current_interaction.compute_latencies()
            interaction_data = asdict(self.current_interaction)
            transformed_data = self._transform_to_camel_case(interaction_data)
            self.analytics_client.send_interaction_analytics_safe({
                "sessionId": self.session_id,
                "data": [transformed_data]
            })

    def is_collecting(self) -> bool:
        return bool(self.agent_info)

realtime_metrics_collector = RealtimeMetricsCollector() 