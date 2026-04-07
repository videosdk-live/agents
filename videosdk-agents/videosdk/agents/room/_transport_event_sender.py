import asyncio
import logging
import datetime
from videosdk import (
    Agent as TransportAgent,
    AgentState as TransportAgentState,
)
from ..event_bus import global_event_emitter
from ..utils import format_metrics

logger = logging.getLogger(__name__)

class TransportEventSender:
    """Handles agent signaling over the transport layer by listening to global events."""

    _AGENT_STATE_MAP = {
        "idle": TransportAgentState.IDLE,
        "speaking": TransportAgentState.SPEAKING,
        "listening": TransportAgentState.LISTENING,
        "thinking": TransportAgentState.THINKING,
    }

    def __init__(self, room_handler):
        self.room_handler = room_handler
        self._transport_agent_cache: TransportAgent | None = None

        global_event_emitter.on("AGENT_STATE_CHANGED", self._on_agent_state_changed)
        global_event_emitter.on("USER_TRANSCRIPT_ADDED", self._on_user_transcript_added)
        global_event_emitter.on("AGENT_TRANSCRIPT_ADDED", self._on_agent_transcript_added)
        global_event_emitter.on("TURN_METRICS_ADDED", self._on_turn_metrics_added)

    def _get_transport_agent(self) -> TransportAgent | None:
        """Get the transport-level Agent instance (from videosdk SDK)."""
        if self._transport_agent_cache is not None:
            return self._transport_agent_cache
        
        if self.room_handler.meeting and self.room_handler.meeting.local_participant:
            participant = self.room_handler.meeting.local_participant
            if isinstance(participant, TransportAgent):
                self._transport_agent_cache = participant
                return participant
        return None

    def _on_agent_state_changed(self, data: dict):
        state_value = data.get("state")
        if state_value:
            asyncio.create_task(self.send_agent_state(state_value))

    def _on_user_transcript_added(self, data: dict):
        text = data.get("text")
        if text:
            peer_id = ""
            if hasattr(self.room_handler, "participants_data") and self.room_handler.participants_data:
                peer_id = next(iter(self.room_handler.participants_data.keys()))
            timestamp = int(datetime.datetime.now().timestamp() * 1000)
            asyncio.create_task(self.send_agent_transcript(text, peer_id,timestamp))

    def _on_agent_transcript_added(self, data: dict):
        text = data.get("text")
        if text:
            peer_id = ""
            if self.room_handler.meeting and self.room_handler.meeting.local_participant:
                peer_id = self.room_handler.meeting.local_participant.id
            timestamp = int(datetime.datetime.now().timestamp() * 1000)
            asyncio.create_task(self.send_agent_transcript(text, peer_id,timestamp))

    def _on_turn_metrics_added(self, data: dict):
        metrics = data.get("metrics")
        if metrics:
            asyncio.create_task(self.send_agent_metrics(metrics))

    async def send_agent_state(self, state_value: int) -> None:
        """Send agent state change via the transport signaling channel."""
        agent = self._get_transport_agent()
        if agent:
            transport_state = self._AGENT_STATE_MAP.get(state_value)
            if transport_state:
                try:
                    await agent.async_send_state_changed(transport_state)
                except Exception as e:
                    logger.error(f"Error sending agent state via transport: {e}")

    async def send_agent_transcript(self, text: str, peer_id: str, timestamp: int) -> None:
        """Send agent transcript via the transport signaling channel."""
        agent = self._get_transport_agent()
        if agent:
            try:
                await agent.async_send_transcript(text, peer_id, timestamp)
            except Exception as e:
                logger.error(f"Error sending agent transcript via transport: {e}")

    async def send_agent_metrics(self, data: dict) -> None:
        """Send agent metrics via the transport signaling channel."""
        agent = self._get_transport_agent()
        if agent:
            try:
                payload = format_metrics(data)
                await agent.async_send_metrics(payload)
                print(f"HERE LOOK HERE METRICS SENT>>>>>>>>>>>>>>>>> ?????????: {payload}")
            except Exception as e:
                logger.error(f"Error sending agent metrics via transport: {e}")

    def cleanup(self):
        """Cleanup event listeners."""
        global_event_emitter.off("AGENT_STATE_CHANGED", self._on_agent_state_changed)
        global_event_emitter.off("USER_TRANSCRIPT_ADDED", self._on_user_transcript_added)
        global_event_emitter.off("AGENT_TRANSCRIPT_ADDED", self._on_agent_transcript_added)
        global_event_emitter.off("TURN_METRICS_ADDED", self._on_turn_metrics_added)