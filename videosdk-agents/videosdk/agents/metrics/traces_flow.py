from typing import Dict, Any, Optional
from opentelemetry.trace import Span, StatusCode
from opentelemetry import trace
from .integration import create_span, complete_span, create_log
from .models import InteractionMetrics
from contextlib import contextmanager
import asyncio

class TracesFlowManager:
    """
    Manages the flow of OpenTelemetry traces for agent interactions,
    ensuring correct parent-child relationships between spans.
    """

    def __init__(self, room_id: str, session_id: Optional[str] = None):
        self.room_id = room_id
        self.session_id = session_id
        self.root_span: Optional[Span] = None
        self.agent_session_span: Optional[Span] = None
        self.main_interaction_span: Optional[Span] = None
        self.agent_session_config_span: Optional[Span] = None
        self.agent_session_closed_span: Optional[Span] = None
        self._interaction_count = 0
        self.root_span_ready = asyncio.Event()
        self.a2a_span: Optional[Span] = None
        self._a2a_interaction_count = 0

    def set_session_id(self, session_id: str):
        """Set the session ID for the trace manager."""
        self.session_id = session_id

    def start_agent_joined_meeting(self, attributes: Dict[str, Any]):
        """Starts the root span for the agent joining a meeting."""
        if self.root_span:
            print("Root span 'Agent Joined Meeting' already exists.")
            return
        
        self.root_span = create_span("Agent Joined Meeting", attributes)
        if self.root_span:
            self.root_span_ready.set()
            with trace.use_span(self.root_span):
                create_log("Agent joining confirmed", "INFO", { "meeting_id": self.room_id })

    async def start_agent_session_config(self, attributes: Dict[str, Any]):
        """Starts the span for the agent's session config, child of the root span."""
        await self.root_span_ready.wait()
        if not self.root_span:
            print("Cannot start agent session config span without a root span.")
            return

        if self.agent_session_config_span:
            print("Agent session config span already exists.")
            return

        self.agent_session_config_span = create_span("Agent Session Config", attributes, parent_span=self.root_span)
        if self.agent_session_config_span:
            with trace.use_span(self.agent_session_config_span):
                create_log("Agent session config created", "INFO", attributes)

    def end_agent_session_config(self):
        """Completes the agent session config span."""
        self.end_span(self.agent_session_config_span, "Agent session config ended")
        self.agent_session_config_span = None

    async def start_agent_session_closed(self, attributes: Dict[str, Any]):
        """Starts the span for agent session closed."""
        await self.root_span_ready.wait()
        if not self.root_span:
            print("Cannot start agent session closed span without a root span.")
            return

        if self.agent_session_closed_span:
            print("Agent session closed span already exists.")
            return

        self.agent_session_closed_span = create_span("Agent Session Closed", attributes, parent_span=self.root_span)
        if self.agent_session_closed_span:
            with trace.use_span(self.agent_session_closed_span):
                create_log("Agent session closed span created", "INFO", attributes)

    def end_agent_session_closed(self):
        """Completes the agent session closed span."""
        self.end_span(self.agent_session_closed_span, "Agent session closed")
        self.agent_session_closed_span = None

    async def start_agent_session(self, attributes: Dict[str, Any]):
        """Starts the span for the agent's session, child of the root span."""
        await self.root_span_ready.wait()
        if not self.root_span:
            print("Cannot start agent session span without a root span.")
            return

        if self.agent_session_span:
            print("Agent session span already exists.")
            return

        self.agent_session_span = create_span("Agent Session Started", attributes, parent_span=self.root_span)
        if self.agent_session_span:
            with trace.use_span(self.agent_session_span):
                create_log("Agent session started", "INFO", {
                    "session_id": self.session_id,
                })
        
        self.start_main_interaction()

    def start_main_interaction(self):
        """Starts a parent span for all user-agent interactions."""
        if not self.agent_session_span:
            print("Cannot start main interaction span without an agent session span.")
            return

        if self.main_interaction_span:
            print("Main interaction span already exists.")
            return
            
        self.main_interaction_span = create_span("User & Agent Interactions", parent_span=self.agent_session_span)
        if self.main_interaction_span:
            with trace.use_span(self.main_interaction_span):
                create_log("Main interaction span started, ready for user interactions.", "INFO")

    def create_interaction_trace(self, interaction_data: InteractionMetrics):
        """
        Creates a full trace for a single interaction from its collected metrics data.
        This includes the parent interaction span and all its processing child spans.
        """
        if not self.main_interaction_span:
            print("ERROR: Cannot create interaction trace without a main interaction span.")
            return

        self._interaction_count += 1
        interaction_name = f"Interaction {self._interaction_count}"
        
        interaction_span = create_span(interaction_name, 
                                       {"interaction_id": interaction_data.interaction_id}, 
                                       parent_span=self.main_interaction_span)
        if interaction_span:
            with trace.use_span(interaction_span):
                create_log(f"Interaction STARTed  {interaction_name} started", "INFO", {
                    "interaction_id": interaction_data.interaction_id
                })

        if not interaction_span:
            return

        with trace.use_span(interaction_span, end_on_exit=False):

            if interaction_data.stt_latency is not None:
                stt_span = create_span("STT Processing Time", {"duration_ms": interaction_data.stt_latency}, parent_span=interaction_span)
                self.end_span(stt_span)

            if interaction_data.llm_latency is not None:
                llm_span = create_span("LLM Processing Time", {"duration_ms": interaction_data.llm_latency, "llm_latency": interaction_data.llm_latency}, parent_span=interaction_span)
                self.end_span(llm_span)

            if interaction_data.function_tools_called is not None:
                for tool in interaction_data.function_tools_called:
                    tool_span = create_span("Function Tool Called", {"tool_name": tool}, parent_span=interaction_span)
                    self.end_span(tool_span)

            if interaction_data.tts_latency is not None:
                tts_span = create_span("TTS Processing Time ", {"duration_ms": interaction_data.tts_latency, "ttfb_ms": interaction_data.ttfb, "tts_latency": interaction_data.tts_latency}, parent_span=interaction_span)
                self.end_span(tts_span)

            if interaction_data.timeline:
                for event in interaction_data.timeline:
                    if event.event_type == "user_speech":
                        user_speech_span = create_span("User Speech", {"duration_ms": event.duration_ms, "text": event.text}, parent_span=interaction_span)
                        self.end_span(user_speech_span)
                    elif event.event_type == "agent_speech":
                        agent_speech_span = create_span("Agent Speech", {"duration_ms": event.duration_ms, "text": event.text}, parent_span=interaction_span)
                        self.end_span(agent_speech_span)    

            if interaction_data.e2e_latency is not None:
                e2e_span = create_span("E2E Processing Time", {"duration_ms": interaction_data.e2e_latency, "e2e_latency": interaction_data.e2e_latency}, parent_span=interaction_span)
                self.end_span(e2e_span)
        
        self.end_span(interaction_span, message="Interaction trace created from data.")

    def end_main_interaction(self):
        """Completes the main interaction span."""
        self.end_span(self.main_interaction_span, "All interactions processed")
        self.main_interaction_span = None

    def agent_say_called(self, message: str):
        """Creates a span for the agent's say method."""
        if not self.agent_session_span:
            print("Cannot create agent say span without an agent session span.")
            return

        current_span = trace.get_current_span()

        agent_say_span = create_span(
            "Agent Say",
            {"Agent Say Message": message},
            parent_span=current_span if current_span else self.agent_session_span
        )

        self.end_span(agent_say_span, "Agent say span created")    

    def create_a2a_trace(self, name: str, attributes: Dict[str, Any]) -> Optional[Span]:
        """Creates an A2A trace under the main interaction span."""
        if not self.main_interaction_span:
            print("Cannot create A2A trace without main interaction span.")
            return None

        if not self.a2a_span:
            self.a2a_span = create_span(
                "Agent-to-Agent Communications",
                {"total_a2a_interactions": self._a2a_interaction_count},
                parent_span=self.main_interaction_span
            )
            if self.a2a_span:
                with trace.use_span(self.a2a_span):
                    create_log("A2A communication started", "INFO")

        if not self.a2a_span:
            print("Failed to create A2A parent span")
            return None

        self._a2a_interaction_count += 1
        span_name = f"A2A {self._a2a_interaction_count}: {name}"
        
        a2a_span = create_span(
            span_name, 
            {
                **attributes,
                "a2a_interaction_number": self._a2a_interaction_count,
                "parent_span": "Agent-to-Agent Communications"
            }, 
            parent_span=self.a2a_span
        )
        
        if a2a_span:
            with trace.use_span(a2a_span):
                create_log(f"A2A event: {name}", "INFO", attributes)
        
        return a2a_span

    def end_a2a_trace(self, span: Optional[Span], message: str = ""):
        """Ends an A2A trace span."""
        if span:
            with trace.use_span(span):
                if message:
                    create_log(message, "INFO")
            complete_span(span, StatusCode.OK)

    def end_a2a_communication(self):
        """Ends the A2A communication parent span."""
        if self.a2a_span:
            with trace.use_span(self.a2a_span):
                create_log(f"A2A communication ended with {self._a2a_interaction_count} interactions", "INFO")
            complete_span(self.a2a_span, StatusCode.OK)
            self.a2a_span = None
            self._a2a_interaction_count = 0  

    def end_agent_session(self):
        """Completes the agent session span."""
        if self.main_interaction_span:
            self.end_main_interaction()
        self.end_span(self.agent_session_span, "Agent session ended")
        self.agent_session_span = None

    def end_agent_joined_meeting(self):
        """Completes the root span."""
        if self.agent_session_span:
            self.end_agent_session()
        if self.agent_session_config_span:
            self.end_agent_session_config()
        if self.agent_session_closed_span:
            self.end_agent_session_closed()
        self.end_span(self.root_span, "Agent left meeting")
        self.root_span = None
            
    def end_span(self, span: Optional[Span], message: str = "", status_code: StatusCode = StatusCode.OK):
        """Completes a given span with a status."""
        if span:
            complete_span(span, status_code, message) 