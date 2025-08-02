from typing import Dict, Any, Optional
from opentelemetry.trace import Span, StatusCode
from opentelemetry import trace
from .integration import create_span, complete_span, create_log
from .models import InteractionMetrics, RealtimeInteractionData, TimelineEvent
import asyncio
import time

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
        
        agent_name = attributes.get('agent_name', 'UnknownAgent')
        agent_id = attributes.get('peerId', 'UnknownID')

        span_name = f"agent_{agent_name}_agentId_{agent_id}"

        self.root_span = create_span(span_name, attributes)

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
                create_log(f"Interaction Started: {interaction_name}", "INFO", {
                    "interaction_id": interaction_data.interaction_id
                })

        if not interaction_span:
            return

        with trace.use_span(interaction_span, end_on_exit=False):

            stt_errors = [e for e in interaction_data.errors if e['source'] == 'STT']
            if interaction_data.stt_start_time is not None or interaction_data.stt_end_time is not None or stt_errors:

                stt_span_name = "STT Processing"
                if interaction_data.stt_latency:
                    stt_span_name = f"STT Processing (took {interaction_data.stt_latency}ms)"
                    
                stt_attrs = {}
                if interaction_data.stt_start_time:
                    stt_attrs["stt_start_timestamp"] = interaction_data.stt_start_time
                    stt_attrs["stt_start_time_readable"] = time.strftime("%H:%M:%S", time.localtime(interaction_data.stt_start_time))
                if interaction_data.stt_end_time:
                    stt_attrs["stt_end_timestamp"] = interaction_data.stt_end_time
                    stt_attrs["stt_end_time_readable"] = time.strftime("%H:%M:%S", time.localtime(interaction_data.stt_end_time))
                if interaction_data.stt_latency:
                    stt_attrs["duration_ms"] = interaction_data.stt_latency
                    
                stt_span = create_span(stt_span_name, stt_attrs, parent_span=interaction_span)

                if stt_span:
                    if interaction_data.stt_start_time:
                        audio_input_attrs = {
                            "exact_timestamp": interaction_data.stt_start_time,
                            "readable_time": time.strftime("%H:%M:%S", time.localtime(interaction_data.stt_start_time))
                        }
                        audio_input_span = create_span("Taking input started", audio_input_attrs, parent_span=stt_span)
                        self.end_span(audio_input_span)

                    if interaction_data.stt_end_time:
                        convert_text_attrs = {
                            "exact_timestamp": interaction_data.stt_end_time,
                            "readable_time": time.strftime("%H:%M:%S", time.localtime(interaction_data.stt_end_time))
                        }
                        if interaction_data.stt_latency is not None:
                            convert_text_attrs["convert_text_latency"] = interaction_data.stt_latency
                        convert_span = create_span("Text Conversion Completed", convert_text_attrs, parent_span=stt_span)
                        self.end_span(convert_span)

                    for error in stt_errors:
                        stt_span.add_event("error", attributes={
                            "message": error["message"],
                            "timestamp": error["timestamp"]
                        })

                    status = StatusCode.ERROR if stt_errors else StatusCode.OK
                    self.end_span(stt_span, status_code=status)
            
            eou_errors = [e for e in interaction_data.errors if e['source'] == 'TURN-D']
            if interaction_data.eou_start_time is not None or interaction_data.eou_end_time is not None or eou_errors:

                eou_span_name = "EOU Processing"
                if interaction_data.eou_latency:
                    eou_span_name = f"EOU Processing (took {interaction_data.eou_latency}ms)"
                    
                eou_attrs = {}
                if interaction_data.eou_start_time:
                    eou_attrs["eou_start_timestamp"] = interaction_data.eou_start_time
                    eou_attrs["eou_start_time_readable"] = time.strftime("%H:%M:%S", time.localtime(interaction_data.eou_start_time))
                if interaction_data.eou_end_time:
                    eou_attrs["eou_end_timestamp"] = interaction_data.eou_end_time
                    eou_attrs["eou_end_time_readable"] = time.strftime("%H:%M:%S", time.localtime(interaction_data.eou_end_time))
                if interaction_data.eou_latency:
                    eou_attrs["duration_ms"] = interaction_data.eou_latency
                    
                eou_span = create_span(eou_span_name, eou_attrs, parent_span=interaction_span)

                if eou_span:
                    if interaction_data.eou_start_time:
                        eou_start_attrs = {
                            "exact_timestamp": interaction_data.eou_start_time,
                            "readable_time": time.strftime("%H:%M:%S", time.localtime(interaction_data.eou_start_time))
                        }
                        eou_start_span = create_span("EOU Detection Started", eou_start_attrs, parent_span=eou_span)
                        self.end_span(eou_start_span)

                    if interaction_data.eou_end_time:
                        eou_complete_attrs = {
                            "exact_timestamp": interaction_data.eou_end_time,
                            "readable_time": time.strftime("%H:%M:%S", time.localtime(interaction_data.eou_end_time))
                        }
                        if interaction_data.eou_latency is not None:
                            eou_complete_attrs["eou_detection_latency"] = interaction_data.eou_latency
                        eou_complete_span = create_span("EOU Detection Completed", eou_complete_attrs, parent_span=eou_span)
                        self.end_span(eou_complete_span)

                    for error in eou_errors:
                        eou_span.add_event("error", attributes={
                            "message": error["message"],
                            "timestamp": error["timestamp"]
                        })

                    eou_status = StatusCode.ERROR if eou_errors else StatusCode.OK
                    self.end_span(eou_span, status_code=eou_status)
                else:
                    eou_span = None

            llm_errors = [e for e in interaction_data.errors if e['source'] == 'LLM']
            if interaction_data.llm_start_time is not None or interaction_data.llm_end_time is not None or llm_errors:

                llm_span_name = "LLM Processing"
                if interaction_data.llm_latency:
                    llm_span_name = f"LLM Processing (took {interaction_data.llm_latency}ms)"
                    
                llm_attrs = {}
                if interaction_data.llm_start_time:
                    llm_attrs["llm_start_timestamp"] = interaction_data.llm_start_time
                    llm_attrs["llm_start_time_readable"] = time.strftime("%H:%M:%S", time.localtime(interaction_data.llm_start_time))
                if interaction_data.llm_end_time:
                    llm_attrs["llm_end_timestamp"] = interaction_data.llm_end_time
                    llm_attrs["llm_end_time_readable"] = time.strftime("%H:%M:%S", time.localtime(interaction_data.llm_end_time))
                if interaction_data.llm_latency:
                    llm_attrs["duration_ms"] = interaction_data.llm_latency
                
                llm_span = create_span(llm_span_name, llm_attrs, parent_span=interaction_span)

                if llm_span:

                    if interaction_data.llm_start_time:
                        query_start_attrs = {
                            "exact_timestamp": interaction_data.llm_start_time,
                            "readable_time": time.strftime("%H:%M:%S", time.localtime(interaction_data.llm_start_time))
                        }
                        query_start_span = create_span("Query Process start", query_start_attrs, parent_span=llm_span)
                        self.end_span(query_start_span)

                    if interaction_data.llm_end_time:
                        generate_output_attrs = {
                            "exact_timestamp": interaction_data.llm_end_time,
                            "readable_time": time.strftime("%H:%M:%S", time.localtime(interaction_data.llm_end_time))
                        }
                        generate_output_span = create_span("Generated Output", generate_output_attrs, parent_span=llm_span)
                        self.end_span(generate_output_span)

                    if interaction_data.function_tool_timestamps:
                        for tool_data in interaction_data.function_tool_timestamps:
                            tool_attrs = {
                                "tool_name": tool_data["tool_name"],
                                "exact_timestamp": tool_data["timestamp"],
                                "readable_time": tool_data.get("readable_time", time.strftime("%H:%M:%S", time.localtime(tool_data["timestamp"])))
                            }
                            tool_span = create_span(f"Tool Calling: {tool_data['tool_name']}", tool_attrs, parent_span=llm_span)
                            self.end_span(tool_span)

                    for error in llm_errors:
                        llm_span.add_event("error", attributes={
                            "message": error["message"],
                            "timestamp": error["timestamp"]
                        })

                    llm_status = StatusCode.ERROR if llm_errors else StatusCode.OK
                    self.end_span(llm_span, status_code=llm_status)

            tts_errors = [e for e in interaction_data.errors if e['source'] == 'TTS']
            if interaction_data.tts_start_time is not None or interaction_data.tts_end_time is not None or tts_errors:

                tts_span_name = "TTS Processing"
                if interaction_data.tts_latency:
                    tts_span_name = f"TTS Processing (took {interaction_data.tts_latency}ms)"
                    
                tts_attrs = {}
                if interaction_data.tts_start_time:
                    tts_attrs["tts_start_timestamp"] = interaction_data.tts_start_time
                    tts_attrs["tts_start_time_readable"] = time.strftime("%H:%M:%S", time.localtime(interaction_data.tts_start_time))
                if interaction_data.tts_end_time:
                    tts_attrs["tts_end_timestamp"] = interaction_data.tts_end_time
                    tts_attrs["tts_end_time_readable"] = time.strftime("%H:%M:%S", time.localtime(interaction_data.tts_end_time))
                if interaction_data.tts_latency:
                    tts_attrs["duration_ms"] = interaction_data.tts_latency
                if interaction_data.ttfb:
                    tts_attrs["ttfb_ms"] = interaction_data.ttfb
                
                tts_span = create_span(tts_span_name, tts_attrs, parent_span=interaction_span)

                if tts_span:
                    if interaction_data.tts_start_time:
                        input_start_attrs = {
                            "exact_timestamp": interaction_data.tts_start_time,
                            "readable_time": time.strftime("%H:%M:%S", time.localtime(interaction_data.tts_start_time))
                        }
                        input_audio_span = create_span("Taking input started", input_start_attrs, parent_span=tts_span)
                        self.end_span(input_audio_span)

                    if interaction_data.tts_end_time:
                        audio_complete_attrs = {
                            "exact_timestamp": interaction_data.tts_end_time,
                            "readable_time": time.strftime("%H:%M:%S", time.localtime(interaction_data.tts_end_time))
                        }
                        speak_span = create_span("Audio Generation Completed", audio_complete_attrs, parent_span=tts_span)
                        self.end_span(speak_span)

                    for error in tts_errors:
                        tts_span.add_event("error", attributes={
                            "message": error["message"],
                            "timestamp": error["timestamp"]
                        })

                    tts_status = StatusCode.ERROR if tts_errors else StatusCode.OK
                    self.end_span(tts_span, status_code=tts_status)

            if interaction_data.timeline:
                for event in interaction_data.timeline:
                    if event.event_type == "user_speech":
                        user_speech_span = create_span("User Speech", {"duration_ms": event.duration_ms, "text": event.text}, parent_span=interaction_span)
                        self.end_span(user_speech_span)
                    elif event.event_type == "agent_speech":
                        agent_speech_span = create_span("Agent Speech", {"duration_ms": event.duration_ms, "text": event.text}, parent_span=interaction_span)
                        self.end_span(agent_speech_span)    

        if interaction_data.errors:
            vad_turn_errors = [e for e in interaction_data.errors if e['source'] in ['VAD']]
            
            if vad_turn_errors:
                span_name = "VAD Processing Error"

                vad_turn_span = create_span(span_name, parent_span=interaction_span)
                if vad_turn_span:
                    for error in vad_turn_errors:
                        vad_turn_span.add_event("error", attributes={
                            "message": error["message"],
                            "timestamp": error["timestamp"],
                            "source": error["source"]
                        })
                    
                    status = StatusCode.ERROR
                    self.end_span(vad_turn_span, status_code=status)
        
        # TODO: Add proper interruption span.
        if interaction_data.interrupted:
            interrupted_span = create_span("Interrupted", parent_span=interaction_span)
            self.end_span(interrupted_span, message="Agent was interrupted") 
        
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

    def create_realtime_interaction_trace(self, interaction_data: RealtimeInteractionData):
        """
        Creates a full trace for a single realtime interaction from its collected metrics data.
        This includes the parent interaction span and child spans for speech events, tools, and latencies.
        """
        if not self.main_interaction_span:
            print("ERROR: Cannot create realtime interaction trace without a main interaction span.")
            return

        self._interaction_count += 1
        interaction_name = f"Interaction {self._interaction_count}"
        
        interaction_span = create_span(interaction_name, 
                                       {"interaction_id": interaction_data.interaction_id}, 
                                       parent_span=self.main_interaction_span)
        if interaction_span:
            with trace.use_span(interaction_span):
                create_log(f"Realtime Interaction {interaction_name} started", "INFO", {
                    "interaction_id": interaction_data.interaction_id
                })

        if not interaction_span:
            return

        with trace.use_span(interaction_span, end_on_exit=False):

            if interaction_data.timeline:
                user_speech_count = 1
                agent_speech_count = 1
                for event in interaction_data.timeline:
                    if event.event_type == "user_speech":
                        span_name = f"User Speech {user_speech_count}"
                        user_speech_span = create_span(span_name, {
                            "duration_ms": event.duration_ms, 
                            "text": event.text
                        }, parent_span=interaction_span)
                        self.end_span(user_speech_span)
                        user_speech_count += 1
                    elif event.event_type == "agent_speech":
                        span_name = f"Agent Speech {agent_speech_count}"
                        agent_speech_span = create_span(span_name, {
                            "duration_ms": event.duration_ms, 
                            "text": event.text
                        }, parent_span=interaction_span)
                        self.end_span(agent_speech_span)
                        agent_speech_count += 1

            if interaction_data.function_tools_called:
                for i, tool in enumerate(interaction_data.function_tools_called, 1):
                    tool_span = create_span(f"Function Tool Called {i}", {
                        "tool_name": tool
                    }, parent_span=interaction_span)
                    self.end_span(tool_span)

            if interaction_data.ttfw is not None:
                ttfw_span = create_span("TTFW", {"duration_ms": interaction_data.ttfw}, parent_span=interaction_span)
                self.end_span(ttfw_span)

            if interaction_data.thinking_delay is not None:
                thinking_span = create_span("Thinking Delay", {"duration_ms": interaction_data.thinking_delay}, parent_span=interaction_span)
                self.end_span(thinking_span)
            if interaction_data.realtime_model_errors:
                for error in interaction_data.realtime_model_errors:
                    interaction_span.add_event(
                        name="model_error",
                        attributes={
                            "message": error.get("message", "Unknown error"),
                            "timestamp": error.get("timestamp", "N/A"),
                        }
                    )
                model_status = StatusCode.ERROR
            else:
                model_status = StatusCode.OK

            if interaction_data.e2e_latency is not None:
                e2e_span = create_span("E2E Latency", {"duration_ms": interaction_data.e2e_latency}, parent_span=interaction_span)
                self.end_span(e2e_span)
            
        self.end_span(interaction_span, message="Realtime interaction trace created from data.", status_code=model_status) 