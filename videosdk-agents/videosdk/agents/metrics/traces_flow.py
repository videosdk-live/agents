from typing import Any, Dict, Optional, List
from opentelemetry.trace import Span, StatusCode
from opentelemetry import trace
from .integration import create_span, complete_span, create_log
from .metrics_schema import (
    TurnMetrics, 
    SttMetrics, LlmMetrics, TtsMetrics, 
    EouMetrics, VadMetrics, 
    InterruptionMetrics, FallbackEvent, KbMetrics, 
    RealtimeMetrics, 
    SessionMetrics, ParticipantMetrics,
)
import asyncio
from dataclasses import asdict
import time
import json
import logging

logger = logging.getLogger(__name__)

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
        self.session_metrics: Optional[SessionMetrics] = None
        self.participant_metrics: Optional[List[ParticipantMetrics]] = []

    def set_session_metrics(self, session_metrics: SessionMetrics):
        """Set the session metrics for the trace manager."""
        self.session_metrics = session_metrics

    def set_session_id(self, session_id: str):
        """Set the session ID for the trace manager."""
        self.session_id = session_id

    def start_agent_joined_meeting(self, attributes: Dict[str, Any]):
        """Starts the root span for the agent joining a meeting."""
        if self.root_span:
            return
        
        agent_name = attributes.get('agent_name', 'UnknownAgent')
        agent_id = attributes.get('peerId', 'UnknownID')

        span_name = f"Agent Session: agentName_{agent_name}_agentId_{agent_id}"

        start_time = attributes.get('start_time', time.perf_counter())
        self.root_span = create_span(span_name, attributes, start_time=start_time)

        # Always set root_span_ready so downstream awaits don't hang
        # when telemetry is not initialized (create_span returns None)
        self.root_span_ready.set()

        if self.root_span:
            with trace.use_span(self.root_span):
                create_log("Agent Session Started", "INFO", { "meeting_id": self.room_id })

    async def start_agent_session_config(self, attributes: Dict[str, Any]):
        """Starts the span for the agent's session configuration, child of the root span."""
        await self.root_span_ready.wait()
        if not self.root_span:
            return

        if self.agent_session_config_span:
            return

        start_time = attributes.get('start_time', time.perf_counter())
        self.agent_session_config_span = create_span("Session Configuration", attributes, parent_span=self.root_span, start_time=start_time)
        if self.agent_session_config_span:
            with trace.use_span(self.agent_session_config_span):
                self.end_agent_session_config()

    def end_agent_session_config(self):
        """Completes the agent session config span."""
        end_time = time.perf_counter()
        self.end_span(self.agent_session_config_span, "Agent session config ended", end_time=end_time)
        self.agent_session_config_span = None

    async def start_agent_session_closed(self, attributes: Dict[str, Any]):
        """Starts the span for agent session closed."""
        await self.root_span_ready.wait()
        if not self.root_span:
            return

        if self.agent_session_closed_span:
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
            return

        if self.agent_session_span:
            return

        start_time = attributes.get('start_time', time.perf_counter())
        p_m =[]
        a_p_m = []
        for p in self.participant_metrics:
            if p.kind == "user":
                p_m.append(asdict(p))
            else:
                a_p_m.append(asdict(p))
        attributes["participant_metrics"] = p_m
        attributes["agent_participant_metrics"] = a_p_m
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
            return

        if self.main_turn_span:
            return
            
        start_time = time.perf_counter()
        self.main_turn_span = create_span("User & Agent Turns", parent_span=self.agent_session_span, start_time=start_time)
        if self.main_turn_span:
            with trace.use_span(self.main_turn_span):
                create_log("Main Turn span started, ready for user turns.", "INFO")
    
    def create_unified_turn_trace(self, turn: TurnMetrics, session: Any = None) -> None:
        """
        Creates a full trace for a single turn from the unified TurnMetrics schema.
        Handles both cascading and realtime component spans based on what data is present.
        """
        if not self.main_turn_span:
            return
        self._turn_count += 1
        turn_name = f"Turn #{self._turn_count}"

        # Determine turn start time dynamically to encompass all child spans
        start_times = []
        if turn.user_speech_start_time:
            start_times.append(turn.user_speech_start_time)
        if turn.stt_metrics and turn.stt_metrics[0].stt_start_time:
            start_times.append(turn.stt_metrics[0].stt_start_time)
        if turn.llm_metrics and turn.llm_metrics[0].llm_start_time:
            start_times.append(turn.llm_metrics[0].llm_start_time)
        if turn.tts_metrics and turn.tts_metrics[0].tts_start_time:
            start_times.append(turn.tts_metrics[0].tts_start_time)
        if turn.eou_metrics and turn.eou_metrics[0].eou_start_time:
            start_times.append(turn.eou_metrics[0].eou_start_time)
        if turn.timeline_event_metrics:
            for ev in turn.timeline_event_metrics:
                if ev.start_time:
                    start_times.append(ev.start_time)
                    
        turn_span_start_time = min(start_times) if start_times else None

        turn_span = create_span(turn_name, parent_span=self.main_turn_span, start_time=turn_span_start_time)
        if turn_span:
            with trace.use_span(turn_span):
                create_log(f"Turn Started: {turn_name}", "INFO")

        if not turn_span:
            return

        with trace.use_span(turn_span, end_on_exit=False):

            # --- VAD errors ---
            def create_vad_span(vad: VadMetrics):
                vad_errors = [e for e in turn.errors if e.get("source") == "VAD"]
                if vad_errors or turn.vad_metrics:
                    vad_class = turn.session_metrics.provider_per_component.get("vad", {}).get("provider_class")
                    vad_model = turn.session_metrics.provider_per_component.get("vad", {}).get("model_name")
                    vad_span_name = f"{vad_class}: VAD Processing"
                    
                    vad_attrs = {}
                    if not vad_class:
                        return
                    if vad_class:
                        vad_attrs["provider_class"] = vad_class
                    if vad_model:
                        vad_attrs["model_name"] = vad_model
                    if vad.vad_min_silence_duration:
                        vad_attrs["min_silence_duration"] = vad.vad_min_silence_duration
                    if vad.vad_min_speech_duration:
                        vad_attrs["min_speech_duration"] = vad.vad_min_speech_duration
                    if vad.vad_threshold:
                        vad_attrs["threshold"] = vad.vad_threshold

                # Calculate span start time: end_of_speech_time - min_silence_duration
                if vad.user_speech_start_time is None and vad.user_speech_end_time is not None:
                    vad.user_speech_start_time = vad.user_speech_end_time - vad.vad_min_silence_duration
                elif vad.user_speech_start_time is not None and vad.user_speech_end_time is None:
                    vad.user_speech_end_time = vad.user_speech_start_time + vad.vad_min_silence_duration

                vad_start_time = vad.user_speech_start_time
                vad_end_time = vad.user_speech_end_time
                vad_span = create_span(vad_span_name, vad_attrs, parent_span=turn_span, start_time=vad_start_time)
                
                if vad_span:
                    for error in vad_errors:
                        vad_span.add_event("error", attributes={
                            "message": error["message"],
                            "timestamp": error["timestamp"],
                            "source": error["source"]
                        })
                        with trace.use_span(vad_span):
                            vad_error_span = create_span("VAD Error", {"message": error["message"]}, parent_span=vad_span, start_time=error["timestamp"])
                            self.end_span(vad_error_span, end_time=error["timestamp"]+0.100)
                            vad_errors.remove(error)
                    
                    vad_status = StatusCode.ERROR if vad_errors else StatusCode.OK
                    self.end_span(vad_span, status_code=vad_status, end_time=vad_end_time)

            vad_list = turn.vad_metrics if turn.vad_metrics else None
            if vad_list:
                for vad in vad_list:
                    try:
                        create_vad_span(vad)
                    except Exception as e:
                        logger.error(f"Error creating VAD span: {e}")

            # --- Interruption span ---
            def create_interruption_span(interruption: InterruptionMetrics):
                if interruption.false_interrupt_start_time and interruption.false_interrupt_end_time:
                    false_interrupt_attrs = {}
                    if interruption.interrupt_mode:
                        false_interrupt_attrs["interrupt_mode"] = interruption.interrupt_mode
                    if interruption.false_interrupt_pause_duration:
                        false_interrupt_attrs["pause_duration_config"] = interruption.false_interrupt_pause_duration
                    if interruption.false_interrupt_duration:
                        false_interrupt_attrs["false_interrupt_duration"] = interruption.false_interrupt_duration
                    if interruption.resumed_after_false_interrupt:
                        false_interrupt_attrs["resumed_after_false_interrupt"] = True
                    if interruption.false_interrupt_duration:
                        false_interrupt_attrs["actual_duration"] = interruption.false_interrupt_duration
                    
                    false_interrupt_end = interruption.false_interrupt_end_time
                    if false_interrupt_end is None:
                        # False interrupt was followed by true interrupt
                        false_interrupt_end = interruption.interrupt_start_time
                    
                    false_interrupt_span_name = "False Interruption (Resumed)" if interruption.resumed_after_false_interrupt else "False Interruption (Escalated)"
                    false_interrupt_span = create_span(false_interrupt_span_name, false_interrupt_attrs, parent_span=turn_span, start_time=interruption.false_interrupt_start_time)
                    self.end_span(false_interrupt_span, message="False interruption detected", end_time=false_interrupt_end)

                if turn.is_interrupted:
                    interrupted_attrs = {}
                    if interruption.interrupt_mode:
                        interrupted_attrs["interrupt_mode"] = interruption.interrupt_mode
                    if interruption.interrupt_min_duration:
                        interrupted_attrs["interrupt_min_duration"] = interruption.interrupt_min_duration
                    if interruption.interrupt_min_words:
                        interrupted_attrs["interrupt_min_words"] = interruption.interrupt_min_words
                    if interruption.false_interrupt_pause_duration:
                        interrupted_attrs["false_interrupt_pause_duration"] = interruption.false_interrupt_pause_duration
                    if interruption.resume_on_false_interrupt:
                        interrupted_attrs["resume_on_false_interrupt"] = interruption.resume_on_false_interrupt
                    if interruption.interrupt_reason:
                        interrupted_attrs["interrupt_reason"] = interruption.interrupt_reason
                    if interruption.interrupt_words:
                        interrupted_attrs["interrupt_words"] = interruption.interrupt_words
                    if interruption.interrupt_duration:
                        interrupted_attrs["interrupt_duration"] = interruption.interrupt_duration
                    # Mark if this was preceded by a false interrupt
                    if interruption.false_interrupt_start_time is not None:
                        interrupted_attrs["preceded_by_false_interrupt"] = True
                    
                    interrupted_span = create_span("Turn Interrupted", interrupted_attrs, parent_span=turn_span, start_time=interruption.interrupt_start_time)
            
                # Calculate interrupt end time with proper None checks
                if interruption.interrupt_start_time is not None:
                    if interruption.interrupt_duration is not None:
                        interruption.interrupt_end_time = interruption.interrupt_start_time + interruption.interrupt_duration
                    elif interruption.interrupt_min_duration is not None:
                        interruption.interrupt_end_time = interruption.interrupt_start_time + interruption.interrupt_min_duration
                    else:
                        interruption.interrupt_end_time = interruption.interrupt_start_time
            
                self.end_span(interrupted_span, message="Agent was interrupted", end_time=interruption.interrupt_end_time) 

            
            if turn.interruption_metrics:
                try:
                    create_interruption_span(turn.interruption_metrics)
                except Exception as e:
                    logger.error(f"Error creating interruption span: {e}")
                
            
            # --- Fallback spans ---
            def create_fallback_span(fallback: FallbackEvent):
                is_recovery = fallback.is_recovery
                if is_recovery:
                    fallback_span_name = f"Recovery: {fallback.component_type}"
                    fallback_attrs = {
                        "temporary_disable_sec": fallback.temporary_disable_sec,
                        "permanent_disable_after_attempts": fallback.permanent_disable_after_attempts,
                        "recovery_attempt": fallback.recovery_attempt,
                        "message": fallback.message,
                        "restored_provider": fallback.new_provider_label,
                        "previous_provider": fallback.original_provider_label,
                    }
                    span_time = fallback.start_time
                    recovery_span = create_span(fallback_span_name, fallback_attrs, parent_span=turn_span, start_time=span_time)
                    if recovery_span:
                        self.end_span(recovery_span, message="Recovery completed", end_time=fallback.end_time)
                        return
                
                fallback_span_name = f"Fallback: {fallback.component_type}"
                
                fallback_attrs = {
                    "temporary_disable_sec": fallback.temporary_disable_sec,
                    "permanent_disable_after_attempts": fallback.permanent_disable_after_attempts,
                    "recovery_attempt": fallback.recovery_attempt,
                    "message": fallback.message,
                }
                
                # Use same start_time for all spans (instant spans)
                span_time = fallback.start_time
                fallback_span = create_span(fallback_span_name, fallback_attrs, parent_span=turn_span, start_time=span_time)
                
                if fallback_span:
                    # Child trace for original connection attempt (if exists)
                    if fallback.original_provider_label:
                        original_conn_attrs = {
                            "provider": fallback.original_provider_label,
                            "status": "failed"
                        }
                        
                        original_conn_span = create_span(
                            f"Connection: {fallback.original_provider_label}",
                            original_conn_attrs,
                            parent_span=fallback_span,
                            start_time=span_time
                        )
                        self.end_span(original_conn_span, status_code=StatusCode.ERROR, end_time=span_time)
                    
                    # Child trace for new connection attempt (if switched successfully)
                    if fallback.new_provider_label:
                        new_conn_attrs = {
                            "provider": fallback.new_provider_label,
                            "status": "success"
                        }
                        
                        new_conn_span = create_span(
                            f"Connection: {fallback.new_provider_label}",
                            new_conn_attrs,
                            parent_span=fallback_span,
                            start_time=span_time
                        )
                        self.end_span(new_conn_span, status_code=StatusCode.OK, end_time=span_time)
                    
                    # End the fallback span - status depends on whether we successfully switched
                    fallback_status = StatusCode.OK if fallback.new_provider_label else StatusCode.ERROR
                    self.end_span(fallback_span, status_code=fallback_status, end_time=span_time)

            if turn.fallback_events:
                for fallback in turn.fallback_events:
                    try:
                        create_fallback_span(fallback)
                    except Exception as e:
                        logger.error(f"Error creating fallback span: {e}")

            # --- STT spans ---
            def create_stt_span(stt: SttMetrics):
                stt_errors = [e for e in turn.errors if e.get("source") == "STT"]
                if stt or stt_errors:

                    stt_attrs = {}
                    if stt:
                        stt_class = turn.session_metrics.provider_per_component.get("stt", {}).get("provider_class")
                        if stt_class:
                            stt_attrs["provider_class"] = stt_class
                        
                        stt_model = turn.session_metrics.provider_per_component.get("stt", {}).get("model_name")
                        stt_attrs["input"] = "N/A"
                        if stt_model:
                            stt_attrs["model_name"] = stt_model
                        if stt.stt_latency is not None:
                            stt_attrs["duration_ms"] = stt.stt_latency
                        if stt.stt_start_time:
                            stt_attrs["start_timestamp"] = stt.stt_start_time
                        if stt.stt_end_time:
                            stt_attrs["end_timestamp"] = stt.stt_end_time
                        if stt.stt_transcript:
                            stt_attrs["output"] = stt.stt_transcript
                        if stt_class =="DeepgramSTTV2" and turn.preemtive_generation_enabled:
                            stt_attrs["stt_preemptive_generation_enabled"] = turn.preemtive_generation_enabled
                    
                    stt_span_name = f"{stt_class}: Speech to Text Processing"
                    stt_span = create_span(
                        stt_span_name, stt_attrs,
                        parent_span=turn_span,
                        start_time=stt.stt_start_time if stt else None,
                    )

                    if stt.stt_preemptive_generation_enabled:
                        with trace.use_span(stt_span):
                            preemptive_attributes = {
                                "preemptive_generation_occurred": stt.stt_preemptive_generation_occurred,
                                "partial_text": stt.stt_preflight_transcript,
                                "final_text": stt.stt_transcript,
                            }
                        if stt.stt_preemptive_generation_occurred:
                            preemptive_attributes["preemptive_generation_latency"] = stt.stt_preflight_latency
                        preemptive_span = create_span("Preemptive Generation", preemptive_attributes, parent_span=stt_span, start_time=stt.stt_start_time)
                        preemptive_end_time = stt.stt_preflight_end_time or stt.stt_end_time
                        self.end_span(preemptive_span, end_time=preemptive_end_time)


                    if stt_span:
                        for error in stt_errors:
                            stt_span.add_event("error", attributes={
                                "message": error.get("message", ""),
                                "timestamp": error.get("timestamp", ""),
                            })
                            if stt.stt_start_time <= error.get("timestamp") <= stt.stt_end_time:
                                with trace.use_span(stt_span):
                                    stt_error_span = create_span("STT Error", {"message": error.get("message", "")}, parent_span=stt_span, start_time=error.get("timestamp"))
                                    self.end_span(stt_error_span, end_time=error.get("timestamp")+0.100)
                                    stt_errors.remove(error)
                        status = StatusCode.ERROR if stt_errors else StatusCode.OK
                        create_log(f"{stt_class}: Speech to Text Processing Ended with status {status}", "INFO")
                        self.end_span(stt_span, status_code=status, end_time=stt.stt_end_time if stt else None)
            
            stt_list = turn.stt_metrics if turn.stt_metrics else None
            if stt_list:
                for stt in stt_list:
                    try:
                        create_stt_span(stt)
                    except Exception as e:
                        logger.error(f"Error creating STT span: {e}")

            # --- EOU spans ---
            def create_eou_span(eou: EouMetrics):
                eou_errors = [e for e in turn.errors if e.get("source") == "TURN-D"]

                eou_attrs = {}
                if eou:
                    eou_class = turn.session_metrics.provider_per_component.get("eou", {}).get("provider_class")
                    eou_model = turn.session_metrics.provider_per_component.get("eou", {}).get("model_name")
                    
                    if eou_class:
                        eou_attrs["provider_class"] = eou_class
                    if eou_model:
                        eou_attrs["model_name"] = eou_model
                    if turn.user_speech:
                        eou_attrs["input"] = turn.user_speech
                    if eou.eou_latency is not None:
                        eou_attrs["duration_ms"] = eou.eou_latency
                    if eou.eou_start_time:
                        eou_attrs["start_timestamp"] = eou.eou_start_time
                    if eou.eou_end_time:
                        eou_attrs["end_timestamp"] = eou.eou_end_time
                    if eou.waited_for_additional_speech:
                        eou_attrs["waited_for_additional_speech"] = eou.waited_for_additional_speech
                    if eou.eou_probability:
                        eou_attrs["eou_probability"] = round(eou.eou_probability, 4)
                    if turn.session_metrics.eou_config.get("min_speech_wait_timeout"):
                        eou_attrs["min_speech_wait_timeout"] = turn.session_metrics.eou_config.get("min_speech_wait_timeout")
                    if turn.session_metrics.eou_config.get("max_speech_wait_timeout"):
                        eou_attrs["max_speech_wait_timeout"] = turn.session_metrics.eou_config.get("max_speech_wait_timeout")

                    eou_span_name = f"{eou_class}: End-Of-Utterance Detection"

                eou_span = create_span(
                    eou_span_name, eou_attrs,
                    parent_span=turn_span,
                    start_time=eou.eou_start_time if eou else None,
                )
                if eou.waited_for_additional_speech:
                    delay = round(eou.wait_for_additional_speech_duration, 4)
                    with trace.use_span(eou_span):
                        wait_for_additional_speech_span =create_span (
                            "Wait for Additional Speech",
                            {
                                "wait_for_additional_speech_duration":delay,
                                "eou_probability": round(eou.eou_probability, 4),
                            },
                            start_time=eou.eou_end_time,
                        )
                        self.end_span(wait_for_additional_speech_span, status_code=StatusCode.OK, end_time=eou.eou_end_time + delay)
                
                if eou_span:
                    for error in eou_errors:
                        eou_span.add_event("error", attributes={
                            "message": error.get("message", ""),
                            "timestamp": error.get("timestamp", ""),
                        })
                        with trace.use_span(eou_span):
                            eou_error_span = create_span("EOU Error", {"message": error.get("message", "")}, parent_span=eou_span, start_time=error.get("timestamp"))
                            self.end_span(eou_error_span, end_time=error.get("timestamp")+0.100)
                            eou_errors.remove(error)
                    eou_status = StatusCode.ERROR if eou_errors else StatusCode.OK
                    self.end_span(eou_span, status_code=eou_status, end_time=eou.eou_end_time if eou else None)

            eou_list = turn.eou_metrics if turn.eou_metrics else None
            if eou_list:
                for eou in eou_list:
                    try:
                        create_eou_span(eou)
                    except Exception as e:
                        logger.error(f"Error creating EOU span: {e}")


            # --- LLM spans ---
            def create_llm_span(llm: LlmMetrics):
                llm_errors = [e for e in turn.errors if e.get("source") == "LLM"]
                llm_attrs = {}
                if llm:
                    llm_class = turn.session_metrics.provider_per_component.get("llm", {}).get("provider_class")
                    if llm_class:
                        llm_attrs["provider_class"] = llm_class
                    llm_model = turn.session_metrics.provider_per_component.get("llm", {}).get("model_name")
                    if llm_model:
                        llm_attrs["model_name"] = llm_model
                    if llm.llm_input:
                        llm_attrs["input"] = llm.llm_input
                    if llm.llm_duration:
                        llm_attrs["duration_ms"] = llm.llm_duration
                    if llm.llm_start_time:
                        llm_attrs["start_timestamp"] = llm.llm_start_time
                    if llm.llm_end_time:
                        llm_attrs["end_timestamp"] = llm.llm_end_time
                    if turn.agent_speech:
                        llm_attrs["output"] = turn.agent_speech
                    if llm.prompt_tokens:
                        llm_attrs["input_tokens"] = llm.prompt_tokens
                    if llm.completion_tokens:
                        llm_attrs["output_tokens"] = llm.completion_tokens
                    if llm.prompt_cached_tokens:
                        llm_attrs["cached_input_tokens"] = llm.prompt_cached_tokens
                    if llm.total_tokens:
                        llm_attrs["total_tokens"] = llm.total_tokens

                llm_span_name = f"{llm_class}: LLM Processing"
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
                        with trace.use_span(llm_span):
                            llm_error_span = create_span("LLM Error", {"message": error.get("message", "")}, parent_span=llm_span, start_time=error.get("timestamp"))
                            self.end_span(llm_error_span, end_time=error.get("timestamp")+0.100)
                            llm_errors.remove(error)

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
                    self.end_span(llm_span, status_code=llm_status, end_time=llm.llm_end_time if llm else None)

            llm_list = turn.llm_metrics if turn.llm_metrics else None
            if llm_list:
                for llm in llm_list:
                    try:
                        create_llm_span(llm)
                    except Exception as e:
                        logger.error(f"Error creating LLM span: {e}")

            # --- TTS spans ---
            def create_tts_span(tts: TtsMetrics):
                tts_errors = [e for e in turn.errors if e.get("source") == "TTS"]
                tts_attrs = {}
                if tts:
                    tts_class = turn.session_metrics.provider_per_component.get("tts", {}).get("provider_class")
                    tts_model = turn.session_metrics.provider_per_component.get("tts", {}).get("model_name")
                    if tts_class:
                        tts_attrs["provider_class"] = tts_class
                    if tts_model:
                        tts_attrs["model_name"] = tts_model
                    if turn.agent_speech:
                        tts_attrs["input"] = turn.agent_speech
                    if tts.tts_duration:
                        tts_attrs["duration_ms"] = tts.tts_duration
                    if tts.tts_start_time:
                        tts_attrs["start_timestamp"] = tts.tts_start_time
                    if tts.tts_end_time:
                        tts_attrs["end_timestamp"] = tts.tts_end_time
                    if tts.tts_characters:
                        tts_attrs["characters"] = tts.tts_characters
                    if turn.agent_speech_duration:
                        tts_attrs["audio_duration_ms"] = turn.agent_speech_duration
                    tts_attrs["output"] = "N/A"

                tts_span_name = f"{tts_class}: Text to Speech Processing"
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
                        with trace.use_span(tts_span):
                            tts_error_span = create_span("TTS Error", {"message": error.get("message", "")}, parent_span=tts_span, start_time=error.get("timestamp"))
                            self.end_span(tts_error_span, end_time=error.get("timestamp")+0.100)
                            tts_errors.remove(error)

                    tts_status = StatusCode.ERROR if tts_errors else StatusCode.OK
                    self.end_span(tts_span, status_code=tts_status, end_time=tts.tts_end_time if tts else None)

            tts_list = turn.tts_metrics if turn.tts_metrics else None
            if tts_list:
                for tts in tts_list:
                    try:
                        create_tts_span(tts)
                    except Exception as e:
                        logger.error(f"Error creating TTS span: {e}")

            # --- KB spans ---
            def create_kb_span(kb: KbMetrics):
                kb_span_name = "Knowledge Base: Retrieval"
                kb_attrs = {}
                if turn.user_speech:
                    kb_attrs["input"] = turn.user_speech
                if kb.kb_retrieval_latency:
                    kb_attrs["retrieval_latency_ms"] = kb.kb_retrieval_latency
                if kb.kb_start_time:
                    kb_attrs["start_timestamp"] = kb.kb_start_time
                if kb.kb_end_time:
                    kb_attrs["end_timestamp"] = kb.kb_end_time
                if kb.kb_documents:
                    # Join documents as comma-separated string for readability
                    kb_attrs["documents"] = ", ".join(kb.kb_documents) if len(kb.kb_documents) <= 5 else f"{len(kb.kb_documents)} documents"
                    kb_attrs["document_count"] = len(kb.kb_documents)
                if kb.kb_scores:
                    # Include scores as comma-separated string
                    kb_attrs["scores"] = ", ".join([str(round(s, 4)) for s in kb.kb_scores[:5]])
                
                kb_span = create_span(kb_span_name, kb_attrs, parent_span=turn_span, start_time=kb.kb_start_time)
                if kb_span:
                    create_log("Knowledge Base: Retrieval Ended", "INFO")
                    self.end_span(kb_span, status_code=StatusCode.OK, end_time=kb.kb_end_time)

            if turn.kb_metrics:
                for kb in turn.kb_metrics:
                    try:
                        create_kb_span(kb)
                    except Exception as e:
                        logger.error(f"Error creating KB span: {e}")

            # --- Realtime spans (for S2S modes) ---
            def create_rt_span(rt: RealtimeMetrics):
                rt_errors = [e for e in turn.errors if e.get("source") == "REALTIME"]

                rt_attrs = {}
                if rt:
                    rt_class = turn.session_metrics.provider_per_component.get("realtime", {}).get("provider_class")
                    if rt_class:
                        rt_attrs["provider_class"] = rt_class
                    rt_model = turn.session_metrics.provider_per_component.get("realtime", {}).get("model_name")
                    if rt_model:
                        rt_attrs["model_name"] = rt_model

                rt_start_time = turn.user_speech_end_time if turn.user_speech_end_time else turn.agent_speech_start_time
                rt_end_time = turn.agent_speech_start_time
                
                # if turn.timeline_event_metrics:
                #     for event in turn.timeline_event_metrics:
                #         if event.event_type == "user_speech":
                #             rt_start_time = event.end_time
                #             break

                #     for event in turn.timeline_event_metrics:
                #         if event.event_type == "agent_speech":
                #             rt_end_time = event.start_time
                #             break
                
                
                rt_span_name = f"{rt_class}: Realtime Processing"
                rt_span = create_span(
                    rt_span_name, rt_attrs,
                    parent_span=turn_span,
                    start_time=rt_start_time,
                )
                if rt_span:
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
                        parent_span=rt_span,
                        start_time=rt_start_time,
                    )
                    self.end_span(ttfb_span, end_time=rt_end_time)

                # --- Realtime model errors ---
                rt_errors = [e for e in turn.errors if e.get("source") == "REALTIME"]
                if rt_errors:
                    for error in rt_errors:
                        turn_span.add_event("Errors", attributes={
                            "message": error.get("message", "Unknown error"),
                            "timestamp": error.get("timestamp", "N/A"),
                        })
                        with trace.use_span(turn_span):
                            rt_error_span = create_span("Realtime Error", {"message": error.get("message", "")}, parent_span=turn_span, start_time=error.get("timestamp"))
                            self.end_span(rt_error_span, end_time=error.get("timestamp")+0.100)
                            rt_errors.remove(error)
                self.end_span(rt_span, status_code=StatusCode.ERROR if rt_errors else StatusCode.OK, end_time=rt_end_time)
            

            rt_list = turn.realtime_metrics if turn.realtime_metrics else None
            if rt_list: 
                for rt in rt_list:
                    try:
                        create_rt_span(rt)
                    except Exception as e:
                        logger.error(f"Error creating RT span: {e}")

            def create_error_spans(errors:Dict[str, Any]):
                error_span_name = f"{errors.get('source', 'Unknown')} Error span"
                attr={}
                if errors.get('message'):
                    attr['error message'] = errors.get('message')
                span_start_time = errors.get('timestamp_perf')
                error_span = create_span(error_span_name, attributes=attr, parent_span=turn_span, start_time=span_start_time)
                self.end_span(error_span, status_code=StatusCode.ERROR, end_time=span_start_time + 0.001)
            
            for e in turn.errors:
                try:
                    create_error_spans(e)
                except Exception as e:
                    logger.error(f"Error creating error span: {e}")

            # Determine turn end time first for unbounded children spans
            end_times = []
            if turn.tts_metrics and turn.tts_metrics[-1].tts_end_time:
                end_times.append(turn.tts_metrics[-1].tts_end_time)
            if turn.llm_metrics and turn.llm_metrics[-1].llm_end_time:
                end_times.append(turn.llm_metrics[-1].llm_end_time)
            if turn.agent_speech_end_time:
                end_times.append(turn.agent_speech_end_time)
            if turn.eou_metrics and turn.eou_metrics[-1].eou_end_time:
                end_times.append(turn.eou_metrics[-1].eou_end_time)
            if turn.stt_metrics and turn.stt_metrics[-1].stt_end_time:
                end_times.append(turn.stt_metrics[-1].stt_end_time)
            if turn.user_speech_end_time:
                end_times.append(turn.user_speech_end_time)
            if turn.interruption_metrics and turn.interruption_metrics.false_interrupt_end_time:
                end_times.append(turn.interruption_metrics.false_interrupt_end_time)
                
            turn_end_time = max(end_times) if end_times else None

            if turn.is_interrupted or turn_end_time is None:
                turn_end_time = time.perf_counter()


                
                        

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
                        self.end_span(user_speech_span, end_time=event.end_time if event.end_time else turn_end_time)
                    elif event.event_type == "agent_speech":
                        create_log("Agent Output Speech Detected", "INFO")
                        agent_speech_span = create_span(
                            "Agent Output Speech",
                            {"Transcript": event.text, "duration_ms": event.duration_ms},
                            parent_span=turn_span,
                            start_time=event.start_time,
                        )
                        self.end_span(agent_speech_span, end_time=event.end_time if event.end_time else turn_end_time)



        self.end_span(turn_span, message="End of turn trace.", end_time=turn_end_time)

    def end_main_turn(self):
        """Completes the main turn span."""
        self.end_span(self.main_turn_span, "All turns processed", end_time=time.perf_counter())
        self.main_turn_span = None

    def agent_say_called(self, message: str):
        """Creates a span for the agent's say method."""
        if not self.agent_session_span:
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

    def create_background_audio_start_span(self, file_path: str = None, looping: bool = False, start_time: float = None):
        """Creates a 'Playing Background Audio' span at session level (same level as turn spans)."""
        if not self.main_turn_span:
            return None
        
        bg_audio_attrs = {}
        if file_path:
            bg_audio_attrs["file_path"] = file_path
        bg_audio_attrs["looping"] = looping
        bg_audio_attrs["event"] = "start"
        
        create_log("Playing Background Audio", "INFO")
        start_span = create_span("Playing Background Audio", bg_audio_attrs, parent_span=self.main_turn_span, start_time=start_time or time.perf_counter())
        # End immediately as a point-in-time event
        self.end_span(start_span, message="Background audio started", end_time=start_time or time.perf_counter())
        return start_span

    def create_background_audio_stop_span(self, file_path: str = None, looping: bool = False, end_time: float = None):
        """Creates a 'Stopped Background Audio' span at session level (same level as turn spans)."""
        if not self.main_turn_span:
            return None
        
        bg_audio_attrs = {}
        if file_path:
            bg_audio_attrs["file_path"] = file_path
        bg_audio_attrs["looping"] = looping
        bg_audio_attrs["event"] = "stop"
        
        create_log("Stopped Background Audio", "INFO")
        stop_span = create_span("Stopped Background Audio", bg_audio_attrs, parent_span=self.main_turn_span, start_time=end_time or time.perf_counter())
        # End immediately as a point-in-time event
        self.end_span(stop_span, message="Background audio stopped", end_time=end_time or time.perf_counter())
        return stop_span

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