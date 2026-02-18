from typing import Any, Dict, Optional
from opentelemetry.trace import Span, StatusCode
from opentelemetry import trace
from .integration import create_span, complete_span, create_log
from .models import CascadingTurnData, RealtimeTurnData
from .metrics_schema import TurnMetrics
import asyncio
import time

class TracesFlowManager:
    """
    Manages the flow of OpenTelemetry traces for agent Turns,
    ensuring correct parent-child relationships between spans.
    """

    def __init__(self, room_id: str, session_id: Optional[str] = None):
        self.room_id = room_id
        self.session_id = session_id
        self.root_span: Optional[Span] = None
        self.agent_session_span: Optional[Span] = None
        self.main_turn_span: Optional[Span] = None
        self.agent_session_config_span: Optional[Span] = None
        self.agent_session_closed_span: Optional[Span] = None
        self._turn_count = 0
        self.root_span_ready = asyncio.Event()
        self.a2a_span: Optional[Span] = None
        self._a2a_turn_count = 0

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

        span_name = f"Agent Session: agentName_{agent_name}_agentId_{agent_id}"

        start_time = attributes.get('start_time', time.perf_counter())
        self.root_span = create_span(span_name, attributes, start_time=start_time)

        if self.root_span:
            self.root_span_ready.set()
            with trace.use_span(self.root_span):
                create_log("Agent Session Started", "INFO", { "meeting_id": self.room_id })

    async def start_agent_session_config(self, attributes: Dict[str, Any]):
        """Starts the span for the agent's session configuration, child of the root span."""
        await self.root_span_ready.wait()
        if not self.root_span:
            print("Cannot start agent session config span without a root span.")
            return

        if self.agent_session_config_span:
            print("Agent session config span already exists.")
            return

        start_time = attributes.get('start_time', time.perf_counter())
        self.agent_session_config_span = create_span("Session Configuration", attributes, parent_span=self.root_span, start_time=start_time)
        if self.agent_session_config_span:
            with trace.use_span(self.agent_session_config_span):
                create_log("Agent session config created", "INFO", attributes)

    def end_agent_session_config(self):
        """Completes the agent session config span."""
        end_time = time.perf_counter()
        self.end_span(self.agent_session_config_span, "Agent session config ended", end_time=end_time)
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

        start_time = attributes.get('start_time', time.perf_counter())
        self.agent_session_closed_span = create_span("Agent Session Closed", attributes, parent_span=self.root_span, start_time=start_time)
        if self.agent_session_closed_span:
            with trace.use_span(self.agent_session_closed_span):
                create_log("Agent session closed span created", "INFO", attributes)

    def end_agent_session_closed(self):
        """Completes the agent session closed span."""
        end_time = time.perf_counter()
        self.end_span(self.agent_session_closed_span, "Agent session closed", end_time=end_time)
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

        start_time = attributes.get('start_time', time.perf_counter())
        self.agent_session_span = create_span("Session Started", attributes, parent_span=self.root_span, start_time=start_time)
        if self.agent_session_span:
            with trace.use_span(self.agent_session_span):
                create_log("Agent session started", "INFO", {
                    "session_id": self.session_id,
                })
        
        self.start_main_turn()

    def start_main_turn(self):
        """Starts a parent span for all user-agent turn."""
        if not self.agent_session_span:
            print("Cannot start main turn span without an agent session span.")
            return

        if self.main_turn_span:
            print("Main turn span already exists.")
            return
            
        start_time = time.perf_counter()
        self.main_turn_span = create_span("User & Agent Turns", parent_span=self.agent_session_span, start_time=start_time)
        if self.main_turn_span:
            with trace.use_span(self.main_turn_span):
                create_log("Main Turn span started, ready for user turns.", "INFO")

    def create_cascading_turn_trace(self, cascading_turn_data: CascadingTurnData):
        """
        Creates a full trace for a single turn from its collected metrics data.
        This includes the parent turn span and all its processing child spans.
        """
        if not self.main_turn_span:
            print("ERROR: Cannot create cascading turn trace without a main turn span.")
            return

        self._turn_count += 1
        turn_name = f"Turn #{self._turn_count}"

        if self._turn_count == 1:
            turn_span_start_time = cascading_turn_data.tts_start_time if cascading_turn_data.tts_start_time else None
        else: 
            turn_span_start_time = cascading_turn_data.user_speech_start_time if cascading_turn_data.user_speech_start_time else None
       
        turn_span = create_span(turn_name,parent_span=self.main_turn_span, start_time=turn_span_start_time)
        if turn_span:
            with trace.use_span(turn_span):
                create_log(f"Turn Started: {turn_name}", "INFO")

        if not turn_span:
            return

        with trace.use_span(turn_span, end_on_exit=False):
            stt_errors = [e for e in cascading_turn_data.errors if e['source'] == 'STT']
            if cascading_turn_data.stt_start_time is not None or cascading_turn_data.stt_end_time is not None or stt_errors:
                create_log(f"{cascading_turn_data.stt_provider_class}: Speech to Text Processing Started", "INFO")
                stt_span_name = f"{cascading_turn_data.stt_provider_class}: Speech to Text Processing"

                stt_attrs = {}
                if cascading_turn_data.stt_provider_class:
                    stt_attrs["provider_class"] = cascading_turn_data.stt_provider_class
                if cascading_turn_data.stt_model_name:
                    stt_attrs["model_name"] = cascading_turn_data.stt_model_name
                if cascading_turn_data.stt_latency:
                    stt_attrs["duration_ms"] = cascading_turn_data.stt_latency
                if cascading_turn_data.stt_start_time:
                    stt_attrs["start_timestamp"] = cascading_turn_data.stt_start_time
                if cascading_turn_data.stt_end_time:
                    stt_attrs["end_timestamp"] = cascading_turn_data.stt_end_time
                
                stt_span = create_span(stt_span_name, stt_attrs, parent_span=turn_span, start_time=cascading_turn_data.stt_start_time)

                if stt_span:
                    for error in stt_errors:
                        stt_span.add_event("error", attributes={
                            "message": error["message"],
                            "timestamp": error["timestamp"]
                        })

                    status = StatusCode.ERROR if stt_errors else StatusCode.OK
                    create_log(f"{cascading_turn_data.stt_provider_class}: Speech to Text Processing Ended with status {status}", "INFO")
                    self.end_span(stt_span, status_code=status, end_time=cascading_turn_data.stt_end_time)
            
            eou_errors = [e for e in cascading_turn_data.errors if e['source'] == 'TURN-D']
            if cascading_turn_data.eou_start_time is not None or cascading_turn_data.eou_end_time is not None or eou_errors:
                create_log(f"{cascading_turn_data.eou_provider_class}: End-Of-Utterence Detection Started", "INFO")
                eou_span_name = f"{cascading_turn_data.eou_provider_class}: End-Of-Utterence Detection"
      
                eou_attrs = {}
                if cascading_turn_data.eou_provider_class:
                    eou_attrs["provider_class"] = cascading_turn_data.eou_provider_class
                if cascading_turn_data.eou_model_name:
                    eou_attrs["model_name"] = cascading_turn_data.eou_model_name
                if cascading_turn_data.eou_latency:
                    eou_attrs["duration_ms"] = cascading_turn_data.eou_latency
                if cascading_turn_data.eou_start_time:
                    eou_attrs["start_timestamp"] = cascading_turn_data.eou_start_time
                if cascading_turn_data.eou_end_time:
                    eou_attrs["end_timestamp"] = cascading_turn_data.eou_end_time
                    
                eou_span = create_span(eou_span_name, eou_attrs, parent_span=turn_span, start_time=cascading_turn_data.eou_start_time)

                if eou_span:
                    for error in eou_errors:
                        eou_span.add_event("error", attributes={
                            "message": error["message"],
                            "timestamp": error["timestamp"]
                        })

                    eou_status = StatusCode.ERROR if eou_errors else StatusCode.OK
                    create_log(f"{cascading_turn_data.eou_provider_class}: End-Of-Utterence Detection Ended with status {eou_status}", "INFO")
                    self.end_span(eou_span, status_code=eou_status, end_time=cascading_turn_data.eou_end_time)
                else:
                    eou_span = None

            llm_errors = [e for e in cascading_turn_data.errors if e['source'] == 'LLM']
            if cascading_turn_data.llm_start_time is not None or cascading_turn_data.llm_end_time is not None or llm_errors:
                create_log(f"{cascading_turn_data.llm_provider_class}: LLM Processing Started", "INFO")
                llm_span_name = f"{cascading_turn_data.llm_provider_class}: LLM Processing"

                llm_attrs = {}
                if cascading_turn_data.llm_provider_class:
                    llm_attrs["provider_class"] = cascading_turn_data.llm_provider_class
                if cascading_turn_data.llm_model_name:
                    llm_attrs["model_name"] = cascading_turn_data.llm_model_name
                if cascading_turn_data.llm_duration:
                    llm_attrs["duration_ms"] = cascading_turn_data.llm_duration
                if cascading_turn_data.llm_start_time:
                    llm_attrs["start_timestamp"] = cascading_turn_data.llm_start_time
                if cascading_turn_data.llm_end_time:
                    llm_attrs["end_timestamp"] = cascading_turn_data.llm_end_time
                
                llm_span = create_span(llm_span_name, llm_attrs, parent_span=turn_span, start_time=cascading_turn_data.llm_start_time)

                if llm_span:

                    if cascading_turn_data.function_tool_timestamps:
                        for tool_data in cascading_turn_data.function_tool_timestamps:
                            tool_timestamp = tool_data["timestamp"]
                            tool_span = create_span(f"Invoked Tool: {tool_data['tool_name']}", parent_span=llm_span, start_time=tool_timestamp)
                            self.end_span(tool_span, end_time=tool_timestamp)

                    for error in llm_errors:
                        llm_span.add_event("error", attributes={
                            "message": error["message"],
                            "timestamp": error["timestamp"]
                        })
                    ttft_span = create_span(
                            "Time to First Token", 
                            attributes={"llm_ttft": cascading_turn_data.llm_ttft}, 
                            parent_span=llm_span, 
                            start_time=cascading_turn_data.llm_start_time
                        )
                    ttft_end_timestamp = cascading_turn_data.llm_start_time + (cascading_turn_data.llm_ttft/1000)
                    self.end_span(ttft_span, end_time=ttft_end_timestamp)

                    llm_status = StatusCode.ERROR if llm_errors else StatusCode.OK
                    create_log(f"{cascading_turn_data.llm_provider_class}: LLM Processing Ended with status {llm_status}", "INFO")
                    self.end_span(llm_span, status_code=llm_status, end_time=cascading_turn_data.llm_end_time)

            tts_errors = [e for e in cascading_turn_data.errors if e['source'] == 'TTS']
            if cascading_turn_data.tts_start_time is not None or cascading_turn_data.tts_end_time is not None or tts_errors:
                create_log(f"{cascading_turn_data.tts_provider_class}: Text to Speech Processing Started", "INFO")
                tts_span_name = f"{cascading_turn_data.tts_provider_class}: Text to Speech Processing"

                tts_attrs = {}
                if cascading_turn_data.tts_provider_class:
                    tts_attrs["provider_class"] = cascading_turn_data.tts_provider_class
                if cascading_turn_data.tts_model_name:
                    tts_attrs["model_name"] = cascading_turn_data.tts_model_name
                    
                tts_span = create_span(tts_span_name, tts_attrs, parent_span=turn_span, start_time=cascading_turn_data.tts_start_time)

                if tts_span:
                    
                    if cascading_turn_data.tts_end_time is not None:
                        ttfb_span = create_span("Time to First Byte", parent_span=tts_span, start_time=cascading_turn_data.tts_start_time)
                        self.end_span(ttfb_span, end_time=cascading_turn_data.tts_end_time)

                    for error in tts_errors:
                        tts_span.add_event("error", attributes={
                            "message": error["message"],
                            "timestamp": error["timestamp"]
                        })

                    tts_status = StatusCode.ERROR if tts_errors else StatusCode.OK
                    create_log(f"{cascading_turn_data.tts_provider_class}: Text to Speech Processing Ended with status {tts_status}", "INFO")
                    self.end_span(tts_span, status_code=tts_status, end_time=cascading_turn_data.tts_end_time)

            if cascading_turn_data.timeline:
                for event in cascading_turn_data.timeline:
                    if event.event_type == "user_speech":
                        create_log(f"User Input Speech Detected", "INFO")
                        user_speech_span = create_span("User Input Speech", {"Transcript": event.text}, parent_span=turn_span, start_time=event.start_time)
                        self.end_span(user_speech_span, end_time=event.end_time)
                    elif event.event_type == "agent_speech":
                        create_log(f"Agent Output Speech Detected", "INFO")
                        agent_speech_span = create_span("Agent Output Speech", {"Transcript": event.text}, parent_span=turn_span, start_time=event.start_time)
                        self.end_span(agent_speech_span, end_time=event.end_time)    

        if cascading_turn_data.errors:
            vad_turn_errors = [e for e in cascading_turn_data.errors if e['source'] in ['VAD']]
            
            if vad_turn_errors:
                span_name = f"{cascading_turn_data.vad_provider_class}: VAD Processing Error"
                
                vad_attrs = {}
                if cascading_turn_data.vad_provider_class:
                    vad_attrs["provider_class"] = cascading_turn_data.vad_provider_class
                if cascading_turn_data.vad_model_name:
                    vad_attrs["model_name"] = cascading_turn_data.vad_model_name

                vad_turn_span = create_span(span_name, vad_attrs, parent_span=turn_span)
                if vad_turn_span:
                    for error in vad_turn_errors:
                        vad_turn_span.add_event("error", attributes={
                            "message": error["message"],
                            "timestamp": error["timestamp"],
                            "source": error["source"]
                        })
                    
                    status = StatusCode.ERROR
                    self.end_span(vad_turn_span, status_code=status)
        
        if cascading_turn_data.interrupted:
            interrupted_span = create_span("Turn Interrupted", parent_span=turn_span)
            self.end_span(interrupted_span, message="Agent was interrupted") 

        turn_end_time = None
        if cascading_turn_data.tts_end_time:
            turn_end_time = cascading_turn_data.tts_end_time
        elif cascading_turn_data.llm_end_time:
            turn_end_time = cascading_turn_data.llm_end_time 
        
        self.end_span(turn_span, message="End of Cascading turn trace.", end_time=turn_end_time)

    def create_unified_turn_trace(self, turn: TurnMetrics, session: Any = None) -> None:
        """
        Creates a full trace for a single turn from the unified TurnMetrics schema.
        Handles both cascading and realtime component spans based on what data is present.
        """
        if not self.main_turn_span:
            print("ERROR: Cannot create unified turn trace without a main turn span.")
            return

        self._turn_count += 1
        turn_name = f"Turn #{self._turn_count}"

        # Determine turn start time
        turn_span_start_time = None
        if self._turn_count == 1 and turn.tts_metrics:
            tts = turn.tts_metrics[-1]
            turn_span_start_time = tts.tts_start_time
        elif turn.user_speech_start_time:
            turn_span_start_time = turn.user_speech_start_time

        turn_span = create_span(turn_name, parent_span=self.main_turn_span, start_time=turn_span_start_time)
        if turn_span:
            with trace.use_span(turn_span):
                create_log(f"Turn Started: {turn_name}", "INFO")

        if not turn_span:
            return

        with trace.use_span(turn_span, end_on_exit=False):
            # --- STT spans ---
            stt_errors = [e for e in turn.errors if e.get("source") == "STT"]
            if turn.stt_metrics or stt_errors:
                stt = turn.stt_metrics[-1] if turn.stt_metrics else None
                stt_provider = stt.provider_class if stt else "STT"
                stt_span_name = f"{stt_provider}: Speech to Text Processing"
                create_log(f"{stt_provider}: Speech to Text Processing Started", "INFO")

                stt_attrs = {}
                if stt:
                    if stt.provider_class:
                        stt_attrs["provider_class"] = stt.provider_class
                    if stt.model_name:
                        stt_attrs["model_name"] = stt.model_name
                    if stt.stt_latency is not None:
                        stt_attrs["duration_ms"] = stt.stt_latency
                    if stt.stt_start_time:
                        stt_attrs["start_timestamp"] = stt.stt_start_time
                    if stt.stt_end_time:
                        stt_attrs["end_timestamp"] = stt.stt_end_time

                stt_span = create_span(
                    stt_span_name, stt_attrs,
                    parent_span=turn_span,
                    start_time=stt.stt_start_time if stt else None,
                )
                if stt_span:
                    for error in stt_errors:
                        stt_span.add_event("error", attributes={
                            "message": error.get("message", ""),
                            "timestamp": error.get("timestamp", ""),
                        })
                    status = StatusCode.ERROR if stt_errors else StatusCode.OK
                    create_log(f"{stt_provider}: Speech to Text Processing Ended with status {status}", "INFO")
                    self.end_span(stt_span, status_code=status, end_time=stt.stt_end_time if stt else None)

            # --- EOU spans ---
            eou_errors = [e for e in turn.errors if e.get("source") == "TURN-D"]
            if turn.eou_metrics or eou_errors:
                eou = turn.eou_metrics[-1] if turn.eou_metrics else None
                eou_provider = eou.provider_class if eou else "EOU"
                eou_span_name = f"{eou_provider}: End-Of-Utterance Detection"
                create_log(f"{eou_provider}: End-Of-Utterance Detection Started", "INFO")

                eou_attrs = {}
                if eou:
                    if eou.provider_class:
                        eou_attrs["provider_class"] = eou.provider_class
                    if eou.model_name:
                        eou_attrs["model_name"] = eou.model_name
                    if eou.eou_latency is not None:
                        eou_attrs["duration_ms"] = eou.eou_latency
                    if eou.eou_start_time:
                        eou_attrs["start_timestamp"] = eou.eou_start_time
                    if eou.eou_end_time:
                        eou_attrs["end_timestamp"] = eou.eou_end_time

                eou_span = create_span(
                    eou_span_name, eou_attrs,
                    parent_span=turn_span,
                    start_time=eou.eou_start_time if eou else None,
                )
                if eou_span:
                    for error in eou_errors:
                        eou_span.add_event("error", attributes={
                            "message": error.get("message", ""),
                            "timestamp": error.get("timestamp", ""),
                        })
                    eou_status = StatusCode.ERROR if eou_errors else StatusCode.OK
                    create_log(f"{eou_provider}: End-Of-Utterance Detection Ended with status {eou_status}", "INFO")
                    self.end_span(eou_span, status_code=eou_status, end_time=eou.eou_end_time if eou else None)

            # --- LLM spans ---
            llm_errors = [e for e in turn.errors if e.get("source") == "LLM"]
            if turn.llm_metrics or llm_errors:
                llm = turn.llm_metrics[-1] if turn.llm_metrics else None
                llm_provider = llm.provider_class if llm else "LLM"
                llm_span_name = f"{llm_provider}: LLM Processing"
                create_log(f"{llm_provider}: LLM Processing Started", "INFO")

                llm_attrs = {}
                if llm:
                    if llm.provider_class:
                        llm_attrs["provider_class"] = llm.provider_class
                    if llm.model_name:
                        llm_attrs["model_name"] = llm.model_name
                    if llm.llm_duration is not None:
                        llm_attrs["duration_ms"] = llm.llm_duration
                    if llm.llm_start_time:
                        llm_attrs["start_timestamp"] = llm.llm_start_time
                    if llm.llm_end_time:
                        llm_attrs["end_timestamp"] = llm.llm_end_time

                llm_span = create_span(
                    llm_span_name, llm_attrs,
                    parent_span=turn_span,
                    start_time=llm.llm_start_time if llm else None,
                )
                if llm_span:
                    # Tool call sub-spans
                    if turn.function_tool_timestamps:
                        for tool_data in turn.function_tool_timestamps:
                            tool_timestamp = tool_data.get("timestamp")
                            tool_span = create_span(
                                f"Invoked Tool: {tool_data.get('tool_name', 'unknown')}",
                                parent_span=llm_span,
                                start_time=tool_timestamp,
                            )
                            self.end_span(tool_span, end_time=tool_timestamp)

                    for error in llm_errors:
                        llm_span.add_event("error", attributes={
                            "message": error.get("message", ""),
                            "timestamp": error.get("timestamp", ""),
                        })

                    # TTFT sub-span
                    if llm and llm.llm_ttft is not None and llm.llm_start_time is not None:
                        ttft_span = create_span(
                            "Time to First Token",
                            attributes={"llm_ttft": llm.llm_ttft},
                            parent_span=llm_span,
                            start_time=llm.llm_start_time,
                        )
                        ttft_end = llm.llm_start_time + (llm.llm_ttft / 1000)
                        self.end_span(ttft_span, end_time=ttft_end)

                    llm_status = StatusCode.ERROR if llm_errors else StatusCode.OK
                    create_log(f"{llm_provider}: LLM Processing Ended with status {llm_status}", "INFO")
                    self.end_span(llm_span, status_code=llm_status, end_time=llm.llm_end_time if llm else None)

            # --- TTS spans ---
            tts_errors = [e for e in turn.errors if e.get("source") == "TTS"]
            if turn.tts_metrics or tts_errors:
                tts = turn.tts_metrics[-1] if turn.tts_metrics else None
                tts_provider = tts.provider_class if tts else "TTS"
                tts_span_name = f"{tts_provider}: Text to Speech Processing"
                create_log(f"{tts_provider}: Text to Speech Processing Started", "INFO")

                tts_attrs = {}
                if tts:
                    if tts.provider_class:
                        tts_attrs["provider_class"] = tts.provider_class
                    if tts.model_name:
                        tts_attrs["model_name"] = tts.model_name

                tts_span = create_span(
                    tts_span_name, tts_attrs,
                    parent_span=turn_span,
                    start_time=tts.tts_start_time if tts else None,
                )
                if tts_span:
                    # TTFB sub-span
                    if tts and tts.tts_first_byte_time is not None:
                        ttfb_span = create_span(
                            "Time to First Byte",
                            parent_span=tts_span,
                            start_time=tts.tts_start_time,
                        )
                        self.end_span(ttfb_span, end_time=tts.tts_first_byte_time)

                    for error in tts_errors:
                        tts_span.add_event("error", attributes={
                            "message": error.get("message", ""),
                            "timestamp": error.get("timestamp", ""),
                        })

                    tts_status = StatusCode.ERROR if tts_errors else StatusCode.OK
                    create_log(f"{tts_provider}: Text to Speech Processing Ended with status {tts_status}", "INFO")
                    self.end_span(tts_span, status_code=tts_status, end_time=tts.tts_end_time if tts else None)

            # --- Realtime spans (for S2S modes) ---
            if turn.realtime_metrics:
                rt = turn.realtime_metrics[-1]
                rt_provider = rt.provider_class or "Realtime"

                # Realtime tool calls
                if turn.function_tools_called:
                    for tool_name in turn.function_tools_called:
                        tool_span = create_span(
                            f"Invoked Tool: {tool_name}",
                            parent_span=turn_span,
                            start_time=time.perf_counter(),
                        )
                        self.end_span(tool_span, end_time=time.perf_counter())

                # TTFB span for realtime
                if turn.e2e_latency is not None:
                    ttfb_span = create_span(
                        "Time to First Word",
                        {"duration_ms": turn.e2e_latency},
                        parent_span=turn_span,
                        start_time=time.perf_counter(),
                    )
                    self.end_span(ttfb_span, end_time=time.perf_counter())

            # --- Realtime model errors ---
            rt_errors = [e for e in turn.errors if e.get("source") == "REALTIME"]
            model_status = StatusCode.ERROR if rt_errors else StatusCode.OK
            if rt_errors:
                for error in rt_errors:
                    turn_span.add_event("Errors", attributes={
                        "message": error.get("message", "Unknown error"),
                        "timestamp": error.get("timestamp", "N/A"),
                    })

            # --- Timeline events ---
            if turn.timeline_event_metrics:
                for event in turn.timeline_event_metrics:
                    if event.event_type == "user_speech":
                        create_log("User Input Speech Detected", "INFO")
                        user_speech_span = create_span(
                            "User Input Speech",
                            {"Transcript": event.text, "duration_ms": event.duration_ms},
                            parent_span=turn_span,
                            start_time=event.start_time,
                        )
                        self.end_span(user_speech_span, end_time=event.end_time)
                    elif event.event_type == "agent_speech":
                        create_log("Agent Output Speech Detected", "INFO")
                        agent_speech_span = create_span(
                            "Agent Output Speech",
                            {"Transcript": event.text, "duration_ms": event.duration_ms},
                            parent_span=turn_span,
                            start_time=event.start_time,
                        )
                        self.end_span(agent_speech_span, end_time=event.end_time)

            # --- VAD errors ---
            vad_errors = [e for e in turn.errors if e.get("source") == "VAD"]
            if vad_errors and turn.vad_metrics:
                vad = turn.vad_metrics[-1]
                span_name = f"{vad.provider_class}: VAD Processing Error"
                vad_attrs = {}
                if vad.provider_class:
                    vad_attrs["provider_class"] = vad.provider_class
                if vad.model_name:
                    vad_attrs["model_name"] = vad.model_name

                vad_span = create_span(span_name, vad_attrs, parent_span=turn_span)
                if vad_span:
                    for error in vad_errors:
                        vad_span.add_event("error", attributes={
                            "message": error.get("message", ""),
                            "timestamp": error.get("timestamp", ""),
                            "source": error.get("source", ""),
                        })
                    self.end_span(vad_span, status_code=StatusCode.ERROR)

            # --- Interruption span ---
            if turn.is_interrupted:
                interrupted_span = create_span("Turn Interrupted", parent_span=turn_span)
                self.end_span(interrupted_span, message="Agent was interrupted")

        # Determine turn end time
        turn_end_time = None
        if turn.tts_metrics and turn.tts_metrics[-1].tts_end_time:
            turn_end_time = turn.tts_metrics[-1].tts_end_time
        elif turn.llm_metrics and turn.llm_metrics[-1].llm_end_time:
            turn_end_time = turn.llm_metrics[-1].llm_end_time
        elif turn.agent_speech_end_time:
            turn_end_time = turn.agent_speech_end_time

        self.end_span(turn_span, message="End of turn trace.", end_time=turn_end_time)

    def end_main_turn(self):
        """Completes the main turn span."""
        self.end_span(self.main_turn_span, "All turns processed", end_time=time.perf_counter())
        self.main_turn_span = None

    def agent_say_called(self, message: str):
        """Creates a span for the agent's say method."""
        if not self.agent_session_span:
            print("Cannot create agent say span without an agent session span.")
            return

        current_span = trace.get_current_span()

        agent_say_span = create_span(
            "Agent Say",
            {"Agent Say Message": message},
            parent_span=current_span if current_span else self.agent_session_span,
            start_time=time.perf_counter()
        )

        self.end_span(agent_say_span, "Agent say span created", end_time=time.perf_counter())    

    def agent_reply_called(self, instructions: str):
        """Creates a span for an agent reply invocation."""
        if not self.agent_session_span:
            print("Cannot create agent reply span without an agent session span.")
            return

        current_span = trace.get_current_span()

        agent_reply_span = create_span(
            "Agent Reply",
            {"Agent Reply Instructions": instructions},
            parent_span=current_span if current_span else self.agent_session_span,
            start_time=time.perf_counter()
        )

        self.end_span(agent_reply_span, "Agent reply span created", end_time=time.perf_counter())

    def create_a2a_trace(self, name: str, attributes: Dict[str, Any]) -> Optional[Span]:
        """Creates an A2A trace under the main turn span."""
        if not self.main_turn_span:
            print("Cannot create A2A trace without main turn span.")
            return None

        if not self.a2a_span:
            self.a2a_span = create_span(
                "Agent-to-Agent Communications",
                {"total_a2a_turns": self._a2a_turn_count},
                parent_span=self.main_turn_span
            )
            if self.a2a_span:
                with trace.use_span(self.a2a_span):
                    create_log("A2A communication started", "INFO")

        if not self.a2a_span:
            print("Failed to create A2A parent span")
            return None

        self._a2a_turn_count += 1
        span_name = f"A2A {self._a2a_turn_count}: {name}"
        
        a2a_span = create_span(
            span_name, 
            {
                **attributes,
                "a2a_turn_number": self._a2a_turn_count,
                "parent_span": "Agent-to-Agent Communications"
            }, 
            parent_span=self.a2a_span,
            start_time=time.perf_counter()
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
            complete_span(span, StatusCode.OK, end_time=time.perf_counter())

    def end_a2a_communication(self):
        """Ends the A2A communication parent span."""
        if self.a2a_span:
            with trace.use_span(self.a2a_span, start_time=time.perf_counter()):
                create_log(f"A2A communication ended with {self._a2a_turn_count} turns", "INFO")
            complete_span(self.a2a_span, StatusCode.OK, end_time=time.perf_counter())
            self.a2a_span = None
            self._a2a_turn_count = 0  

    def end_agent_session(self):
        """Completes the agent session span."""
        if self.main_turn_span:
            self.end_main_turn()
        self.end_span(self.agent_session_span, "Agent session ended", end_time=time.perf_counter())
        self.agent_session_span = None

    def agent_meeting_end(self):
        """Completes the root span."""
        if self.agent_session_span:
            self.end_agent_session()
        if self.agent_session_config_span:
            self.end_agent_session_config()
        if self.agent_session_closed_span:
            self.end_agent_session_closed()
        self.end_span(self.root_span, "Agent left meeting", end_time=time.perf_counter())
        self.root_span = None
            
    def end_span(self, span: Optional[Span], message: str = "", status_code: StatusCode = StatusCode.OK, end_time: Optional[float] = None):
        """Completes a given span with a status."""
        if span:
            if end_time is None:
                end_time = time.perf_counter()
            desc = message if status_code == StatusCode.ERROR else ""
            complete_span(span, status_code, desc, end_time)

    def create_realtime_turn_trace(self, realtime_turn_data: RealtimeTurnData):
        """
        Creates a full trace for a single realtime turn from its collected metrics data.
        This includes the parent turn span and child spans for speech events, tools, and latencies.
        """
        if not self.main_turn_span:
            print("ERROR: Cannot create realtime turn trace without a main turn span.")
            return

        self._turn_count += 1
        turn_name = f"Turn#{self._turn_count}"
        
        turn_span = create_span(turn_name,parent_span=self.main_turn_span,start_time=time.perf_counter())
        if turn_span:
            with trace.use_span(turn_span):
                create_log(f"Realtime Turn {turn_name} started", "INFO")

        if not turn_span:
            return

        with trace.use_span(turn_span, end_on_exit=False):

            if realtime_turn_data.timeline:
                for event in realtime_turn_data.timeline:
                    if event.event_type == "user_speech":
                        span_name = f"User Input Speech"
                        user_speech_span = create_span(span_name, {
                            "duration_ms": event.duration_ms, 
                            "text": event.text
                        }, parent_span=turn_span,start_time=event.start_time)
                        self.end_span(user_speech_span,end_time=event.end_time)
                    elif event.event_type == "agent_speech":
                        span_name = f"Agent Output Speech"
                        agent_speech_span = create_span(span_name, {
                            "duration_ms": event.duration_ms, 
                            "text": event.text
                        }, parent_span=turn_span,start_time=event.start_time)
                        self.end_span(agent_speech_span,end_time=event.end_time)

            if realtime_turn_data.function_tools_called:
                for i, tool in enumerate(realtime_turn_data.function_tools_called, 1):
                    tool_span = create_span(f"Invoked Tool: {tool}", parent_span=turn_span,start_time=time.perf_counter())
                    self.end_span(tool_span,end_time=time.perf_counter())

            if realtime_turn_data.ttfb is not None:
                ttfb_span = create_span("Time to First Word", {"duration_ms": realtime_turn_data.ttfb}, parent_span=turn_span,start_time=time.perf_counter())
                self.end_span(ttfb_span,end_time=time.perf_counter())

            if realtime_turn_data.interrupted is not None:
                interrupted_span = create_span("Turn Interrupted", parent_span=turn_span,start_time=time.perf_counter())
                self.end_span(interrupted_span, message="Agent was interrupted", end_time=time.perf_counter())

            if realtime_turn_data.realtime_model_errors:
                for error in realtime_turn_data.realtime_model_errors:
                    turn_span.add_event(
                        name="Errors",
                        attributes={
                            "message": error.get("message", "Unknown error"),
                            "timestamp": error.get("timestamp", "N/A"),
                        }
                    )
                model_status = StatusCode.ERROR
            else:
                model_status = StatusCode.OK
            
        self.end_span(turn_span, message="End of Realtime Turn trace", status_code=model_status, end_time=time.perf_counter()) 