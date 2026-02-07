import time
import hashlib
import os.path
from typing import Dict, Optional, Any, List, Literal
from dataclasses import asdict
from opentelemetry.trace import Span
from .models import TimelineEvent, CascadingTurnData, CascadingMetricsData
from .analytics import AnalyticsClient
from .traces_flow import TracesFlowManager
from ..playground_manager import PlaygroundManager
import logging

logger = logging.getLogger(__name__)
class CascadingMetricsCollector:
    """Collects and tracks performance metrics for AI agent turns"""
    
    def __init__(self):
        self.data = CascadingMetricsData()
        self.analytics_client = AnalyticsClient()
        self.traces_flow_manager: Optional[TracesFlowManager] = None
        self.active_spans: Dict[str, Span] = {}
        self.pending_user_start_time: Optional[float] = None
        self.playground = False

        
    def set_traces_flow_manager(self, manager: TracesFlowManager):
        """Set the TracesFlowManager instance"""
        self.traces_flow_manager = manager

    def _generate_interaction_id(self) -> str:
        """Generate a hash-based turn ID"""
        timestamp = str(time.time())
        session_id = self.data.session_id or "default"
        interaction_count = str(self.data.total_turns)
        
        hash_input = f"{timestamp}_{session_id}_{interaction_count}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _round_latency(self, latency: float) -> float:
        """Convert latency from seconds to milliseconds and round to 4 decimal places"""
        return round(latency * 1000, 4)
    
    def _transform_to_camel_case(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform snake_case field names to camelCase for analytics"""
        field_mapping = {
            # Speech Info
            'user_speech_start_time': 'userSpeechStartTime',
            'user_speech_end_time': 'userSpeechEndTime',
            'user_speech_duration': 'userSpeechDuration',
            'agent_speech_start_time': 'agentSpeechStartTime',
            'agent_speech_end_time': 'agentSpeechEndTime',
            'agent_speech_duration': 'agentSpeechDuration',

            # STT Metrics
            'stt_latency': 'sttLatency',
            'stt_start_time': 'sttStartTime',
            'stt_end_time': 'sttEndTime',
            'stt_preflight_latency': 'sttPreflightLatency',
            'stt_interim_latency': 'sttInterimLatency',
            'stt_duration': 'sttDuration',
            'stt_confidence': 'sttConfidence',

            'stt_preemptive_generation_occurred': 'sttPreemptiveGenerationOccurred',
            'stt_preemptive_generation_enabled': 'sttPreemptiveGenerationEnabled',

            # For OpenAISTT only
            'stt_input_tokens': 'sttInputTokens',
            'stt_output_tokens': 'sttOutputTokens',
            'stt_total_tokens': 'sttTotalTokens',
            
            # KB Metrics
            'kb_retrieval_latency': 'kbRetrievalLatency',
            'kb_documents': 'kbDocuments',
            'kb_scores': 'kbScores',

            # LLM Metrics
            'llm_input': 'llmInput',
            'llm_duration': 'llmDuration',
            'llm_start_time': 'llmStartTime',
            'llm_end_time': 'llmEndTime',
            'llm_ttft': 'ttft',
            'prompt_tokens': 'promptTokens',
            'completion_tokens': 'completionTokens',
            'total_tokens': 'totalTokens',
            'prompt_cached_tokens': 'promptCachedTokens',
            'tokens_per_second': 'tokensPerSecond',

            # TTS Metrics
            'tts_start_time': 'ttsStartTime',
            'tts_end_time': 'ttsEndTime',
            'tts_duration': 'ttsDuration',
            'tts_characters': 'ttsCharacters',
            'ttfb': 'ttfb',
            "tts_latency": "ttsLatency",

            # EOU Metrics
            'eou_latency': 'eouLatency',
            'eou_start_time': 'eouStartTime',
            'eou_end_time': 'eouEndTime',

            # Providers & Metadata
            'llm_provider_class': 'llmProviderClass',
            'llm_model_name': 'llmModelName',
            'stt_provider_class': 'sttProviderClass',
            'stt_model_name': 'sttModelName',
            'tts_provider_class': 'ttsProviderClass',
            'tts_model_name': 'ttsModelName',
            'vad_provider_class': 'vadProviderClass',
            'vad_model_name': 'vadModelName',
            'eou_provider_class': 'eouProviderClass',
            'eou_model_name': 'eouModelName',
            
            # Logic & Tools
            'e2e_latency': 'e2eLatency',
            'interrupted': 'interrupted',
            'timestamp': 'timestamp',
            'function_tools_called': 'functionToolsCalled',
            'function_tool_timestamps': 'functionToolTimestamps',
            'system_instructions': 'systemInstructions',
            'handoff_occurred': 'handOffOccurred',
            'is_a2a_enabled': 'isA2aEnabled',
            'errors': 'errors',

        }

        timeline_field_mapping = {
            'event_type': 'eventType',
            'start_time': 'startTime',
            'end_time': 'endTime',
            'duration_ms': 'durationInMs'
        }
        
        transformed_data = {}
        for key, value in interaction_data.items():
            camel_key = field_mapping.get(key, key)
            
            if key == 'timeline' and isinstance(value, list):
                transformed_timeline = []
                for event in value:
                    transformed_event = {}
                    for event_key, event_value in event.items():
                        camel_event_key = timeline_field_mapping.get(event_key, event_key)
                        transformed_event[camel_event_key] = event_value
                    transformed_timeline.append(transformed_event)
                transformed_data[camel_key] = transformed_timeline
            else:
                transformed_data[camel_key] = value
        
        return transformed_data

    def _remove_negatives(self, obj: Any) -> Any:
        """Recursively clamp any numeric value < 0 to 0 in dicts/lists."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (int, float)):
                    if v < 0:
                        obj[k] = 0
                elif isinstance(v, (dict, list)):
                    obj[k] = self._remove_negatives(v)
            return obj
        if isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, (int, float)):
                    if v < 0:
                        obj[i] = 0
                elif isinstance(v, (dict, list)):
                    obj[i] = self._remove_negatives(v)
            return obj
        return obj
    
    def _start_timeline_event(self, event_type: str, start_time: float) -> None:
        """Start a timeline event"""
        if self.data.current_turn:
            event = TimelineEvent(
                event_type=event_type,
                start_time=start_time
            )
            self.data.current_turn.timeline.append(event)
    
    def _end_timeline_event(self, event_type: str, end_time: float) -> None:
        """End a timeline event and calculate duration"""
        if self.data.current_turn:
            for event in reversed(self.data.current_turn.timeline):
                if event.event_type == event_type and event.end_time is None:
                    event.end_time = end_time
                    event.duration_ms = self._round_latency(end_time - event.start_time)
                    break
    
    def _update_timeline_event_text(self, event_type: str, text: str) -> None:
        """Update timeline event with text content"""
        if self.data.current_turn:
            for event in reversed(self.data.current_turn.timeline):
                if event.event_type == event_type and not event.text:
                    event.text = text
                    break
    
    def _calculate_e2e_metrics(self, turn: CascadingTurnData) -> None:
        """Calculate E2E and E2ET latencies based on individual component latencies"""
        e2e_components = []
        if turn.stt_latency:
            e2e_components.append(turn.stt_latency)
        if turn.eou_latency:
            e2e_components.append(turn.eou_latency)
        if turn.llm_ttft:
            e2e_components.append(turn.llm_ttft)
        if turn.tts_latency: 
            e2e_components.append(turn.tts_latency)
        
        if e2e_components:
            turn.e2e_latency = round(sum(e2e_components), 4)
        
    def _validate_interaction_has_required_latencies(self, turn: CascadingTurnData) -> bool:
        """
        Validate that the turn has at least one of the required latency metrics.
        Returns True if at least one latency is present, False if ALL are absent/None.
        """
        stt_present = turn.stt_latency is not None
        tts_present = turn.ttfb is not None  
        llm_present = turn.llm_ttft is not None
        eou_present = turn.eou_latency is not None
        
        if not any([stt_present, tts_present, llm_present, eou_present]):
            return False
        
        return True

    def set_session_id(self, session_id: str):
        """Set the session ID for metrics tracking"""
        self.data.session_id = session_id
        self.analytics_client.set_session_id(session_id)
    
    def set_system_instructions(self, instructions: str):
        """Set the system instructions for this session"""
        self.data.system_instructions = instructions
    
    def set_provider_info(self, llm_provider: str = "", llm_model: str = "", 
                        stt_provider: str = "", stt_model: str = "",
                        tts_provider: str = "", tts_model: str = "",
                        vad_provider: str = "", vad_model: str = "",
                        eou_provider: str = "", eou_model: str = ""):
        """Set the provider class and model information for this session"""
        self.data.llm_provider_class = llm_provider
        self.data.llm_model_name = llm_model
        self.data.stt_provider_class = stt_provider
        self.data.stt_model_name = stt_model
        self.data.tts_provider_class = tts_provider
        self.data.tts_model_name = tts_model
        self.data.vad_provider_class = vad_provider
        self.data.vad_model_name = vad_model
        self.data.eou_provider_class = eou_provider
        self.data.eou_model_name = eou_model
    
    def start_new_interaction(self, user_transcript: str = "") -> None:
        """Start tracking a new user-agent turn"""
        if self.data.current_turn:
            self.complete_current_turn()
        
        # Reset the lock for the new turn
        self.data.turn_timestamps_locked = False
        # Clear all start times to prevent stale timestamps from previous turns
        self.data.stt_start_time = None
        self.data.llm_start_time = None
        self.data.tts_start_time = None
        self.data.eou_start_time = None
        self.data.knowledge_base_start_time = None
        
        self.data.total_turns += 1
        
        self.data.current_turn = CascadingTurnData(
            system_instructions=self.data.system_instructions if self.data.total_turns == 1 else "",
            # Provider and model info should be included in every turn
            llm_provider_class=self.data.llm_provider_class,
            llm_model_name=self.data.llm_model_name,
            stt_provider_class=self.data.stt_provider_class,
            stt_model_name=self.data.stt_model_name,
            tts_provider_class=self.data.tts_provider_class,
            tts_model_name=self.data.tts_model_name,
            vad_provider_class=self.data.vad_provider_class,
            vad_model_name=self.data.vad_model_name,
            vad_min_silence_duration=self.data.vad_min_silence_duration,
            vad_min_speech_duration=self.data.vad_min_speech_duration,
            vad_threshold=self.data.vad_threshold,
            eou_provider_class=self.data.eou_provider_class,
            eou_model_name=self.data.eou_model_name,
            stt_preemptive_generation_enabled=self.data.stt_preemptive_generation_enabled,
            min_speech_wait_timeout=self.data.min_speech_wait_timeout,
            max_speech_wait_timeout=self.data.max_speech_wait_timeout,
            interrupt_mode=self.data.interrupt_mode,
            interrupt_min_duration=self.data.interrupt_min_duration,
            interrupt_min_words=self.data.interrupt_min_words,
            false_interrupt_pause_duration=self.data.false_interrupt_pause_duration,
            resume_on_false_interrupt=self.data.resume_on_false_interrupt
        )
        
        if self.pending_user_start_time is not None:
            self.data.current_turn.user_speech_start_time = self.pending_user_start_time
            self._start_timeline_event("user_speech", self.pending_user_start_time)

        if self.data.is_user_speaking and self.data.user_input_start_time:
            if self.data.current_turn.user_speech_start_time is None:
                self.data.current_turn.user_speech_start_time = self.data.user_input_start_time
                if not any(ev.event_type == "user_speech" for ev in self.data.current_turn.timeline):
                    self._start_timeline_event("user_speech", self.data.user_input_start_time)

        if user_transcript:
            self.set_user_transcript(user_transcript)
    
    def complete_current_turn(self) -> None:
        """Complete and store the current turn"""
        if self.data.current_turn:
            self._calculate_e2e_metrics(self.data.current_turn)

            if not self._validate_interaction_has_required_latencies(self.data.current_turn) and self.data.total_turns > 1:
                if self.data.current_turn.user_speech_start_time is not None:
                    if (self.pending_user_start_time is None or
                        self.data.current_turn.user_speech_start_time < self.pending_user_start_time):
                        self.pending_user_start_time = self.data.current_turn.user_speech_start_time
                        logger.info(f"[metrics] Caching earliest user start: {self.pending_user_start_time}")
                self.data.current_turn = None
                return

            if self.traces_flow_manager:
                self.traces_flow_manager.create_cascading_turn_trace(self.data.current_turn)

            self.data.turns.append(self.data.current_turn)
            if self.playground:
                self.playground_manager.send_cascading_metrics(metrics=self.data.current_turn, full_turn_data=True)
            interaction_data = asdict(self.data.current_turn)
            interaction_data['timeline'] = [asdict(event) for event in self.data.current_turn.timeline]
            transformed_data = self._transform_to_camel_case(interaction_data)
            # transformed_data = self._intify_latencies_and_timestamps(transformed_data)

            always_remove_fields = [
                'kb_start_time',
                'kb_end_time',
                'user_speech',
                'stt_preflight_end_time',
                'stt_interim_end_time',
                'errors',
                'functionToolTimestamps',
                'sttStartTime', 'sttEndTime',
                'ttsStartTime', 'ttsEndTime',
                'llmStartTime', 'llmEndTime',
                'eouStartTime', 'eouEndTime',
                'is_a2a_enabled',
                "interactionId",
                "timestamp",
            ]

            if not self.data.current_turn.is_a2a_enabled: 
                always_remove_fields.append("handOffOccurred")

            for field in always_remove_fields:
                if field in transformed_data:
                    del transformed_data[field]

            if len(self.data.turns) > 1: 
                provider_fields = [
                    'systemInstructions',
                    # 'llmProviderClass', 'llmModelName',
                    # 'sttProviderClass', 'sttModelName',
                    # 'ttsProviderClass', 'ttsModelName',
                    'eouProviderClass', 'eouModelName'
                    'vadProviderClass', 'vadModelName'
                ]
                for field in provider_fields:
                    if field in transformed_data:
                        del transformed_data[field]

            transformed_data = self._remove_negatives(transformed_data)

            interaction_payload = {
                "data": [transformed_data]               
            }
            
            self.analytics_client.send_interaction_analytics_safe(interaction_payload) 
            self.data.current_turn = None
            self.data.is_interrupted = False
            self.pending_user_start_time = None
    
    def on_interrupted(self):
        """Called when the user interrupts the agent"""
        if self.data.is_interrupted:
            return

        if self.data.is_agent_speaking:
            self.data.total_interruptions += 1
            self.data.is_interrupted = True
            if self.data.current_turn:
                self.data.current_turn.interrupted = True
                self.data.current_turn.interrupt_start_time = time.perf_counter()
                logger.info(f"User interrupted the agent. Total interruptions: {self.data.total_interruptions}")
                
                if self.playground:
                    self.playground_manager.send_cascading_metrics(metrics={"interrupted": self.data.current_turn.interrupted})

    def on_interrupt_trigger(self, word_count: Optional[int] = 0, duration: Optional[float] = 0):
        """Called when the user interrupts the agent"""
        if self.data.is_interrupted:
            return

        if self.data.is_agent_speaking:
            if self.data.current_turn:
                self.data.current_turn.interrupt_words = word_count
                self.data.current_turn.interrupt_duration = duration
                if self.data.current_turn.interrupt_words:
                    self.data.current_turn.interrupt_reason.append("STT")
                if self.data.current_turn.interrupt_duration:
                    self.data.current_turn.interrupt_reason.append("VAD")

    def set_interrupt_config(self, mode: Literal["VAD_ONLY", "STT_ONLY", "HYBRID"], min_duration: Optional[float] = None, min_words: Optional[int] = None, false_interrupt_pause_duration: Optional[float] = None, resume_on_false_interrupt: Optional[bool] = None):
        self.data.interrupt_mode = mode
        self.data.interrupt_min_duration = min_duration
        self.data.interrupt_min_words = min_words
        self.data.false_interrupt_pause_duration = false_interrupt_pause_duration
        self.data.resume_on_false_interrupt = resume_on_false_interrupt

    def on_false_interrupt_start(self, duration: float):
        """Called when false interrupt timer starts (potential resume scenario)"""
        if self.data.current_turn:
            self.data.current_turn.false_interrupt_start_time = time.perf_counter()
            logger.info(f"False interrupt started - waiting {duration}s to determine if real interrupt")

    def on_false_interrupt_resume(self):
        """Called when TTS resumes after false interrupt timeout (user didn't continue speaking)"""
        if self.data.current_turn:
            self.data.current_turn.is_false_interrupt = True
            self.data.current_turn.false_interrupt_end_time = time.perf_counter()
            if self.data.current_turn.false_interrupt_start_time:
                self.data.current_turn.false_interrupt_duration = self._round_latency(
                    self.data.current_turn.false_interrupt_end_time - self.data.current_turn.false_interrupt_start_time
                )
            self.data.current_turn.resumed_after_false_interrupt = True
            
            # Reset interrupted flag and data since this was NOT a true interrupt
            self.data.current_turn.interrupted = False
            self.data.current_turn.interrupt_start_time = None
            self.data.current_turn.interrupt_end_time = None
            self.data.current_turn.interrupt_words = None
            self.data.current_turn.interrupt_duration = None
            self.data.current_turn.interrupt_reason = []
            self.data.is_interrupted = False
            
            logger.info(f"False interrupt ended - TTS resumed after {self.data.current_turn.false_interrupt_duration}ms")

    def on_false_interrupt_escalated(self, word_count: Optional[int] = None):
        """Called when a false interrupt escalates to a true interrupt (user continued speaking)"""
        if self.data.current_turn:
            self.data.current_turn.is_false_interrupt = True
            self.data.current_turn.false_interrupt_end_time = time.perf_counter()
            if self.data.current_turn.false_interrupt_start_time:
                self.data.current_turn.false_interrupt_duration = self._round_latency(
                    self.data.current_turn.false_interrupt_end_time - self.data.current_turn.false_interrupt_start_time
                )
            self.data.current_turn.resumed_after_false_interrupt = False
            if word_count is not None:
                self.data.current_turn.false_interrupt_words = word_count
            logger.info(f"False interrupt escalated to true interrupt after {self.data.current_turn.false_interrupt_duration}ms")

    
    def on_user_speech_start(self):
        """Called when user starts speaking"""
        if self.data.is_user_speaking:
            return

        if not self.data.current_turn:
            self.start_new_interaction()

        self.data.is_user_speaking = True
        self.data.user_input_start_time = time.perf_counter()

        if self.data.current_turn:
            if self.data.current_turn.user_speech_start_time is None:
                self.data.current_turn.user_speech_start_time = self.data.user_input_start_time
            
            if not any(event.event_type == "user_speech" for event in self.data.current_turn.timeline):
                self._start_timeline_event("user_speech", self.data.user_input_start_time)
    
    def on_user_speech_end(self):
        """Called when user stops speaking"""
        self.data.is_user_speaking = False
        self.data.user_speech_end_time = time.perf_counter()
        
        if self.data.current_turn and self.data.current_turn.user_speech_start_time:
            self.data.current_turn.user_speech_end_time = self.data.user_speech_end_time
            self.data.current_turn.user_speech_duration = self._round_latency(self.data.current_turn.user_speech_end_time - self.data.current_turn.user_speech_start_time)
            self._end_timeline_event("user_speech", self.data.user_speech_end_time)
            logger.info(f"user speech duration: {self.data.current_turn.user_speech_duration}ms")

            if self.playground:
                self.playground_manager.send_cascading_metrics(metrics={"user_speech_duration": self.data.current_turn.user_speech_duration})
    
    def on_agent_speech_start(self):
        """Called when agent starts speaking (actual audio output)"""
        self.data.is_agent_speaking = True
        self.data.agent_speech_start_time = time.perf_counter()
        if self.data.current_turn:
            self.data.current_turn.agent_speech_start_time = self.data.agent_speech_start_time
            if not any(event.event_type == "agent_speech" and event.end_time is None for event in self.data.current_turn.timeline):
                self._start_timeline_event("agent_speech", self.data.agent_speech_start_time)
    
    def on_agent_speech_end(self):
        """Called when agent stops speaking"""
        self.data.is_agent_speaking = False
        agent_speech_end_time = time.perf_counter()
        
        if self.data.current_turn:
            self._end_timeline_event("agent_speech", agent_speech_end_time)
            self.data.current_turn.agent_speech_end_time = agent_speech_end_time


        if self.data.tts_start_time and self.data.tts_first_byte_time:
            total_tts_latency = self.data.tts_first_byte_time - self.data.tts_start_time
            if self.data.current_turn and self.data.current_turn.agent_speech_start_time:
                self.data.current_turn.tts_latency = self._round_latency(total_tts_latency)
                self.data.current_turn.agent_speech_duration = self._round_latency(agent_speech_end_time - self.data.current_turn.agent_speech_start_time)

                if self.playground:
                    self.playground_manager.send_cascading_metrics(metrics={"tts_latency": self.data.current_turn.tts_latency})
                    self.playground_manager.send_cascading_metrics(metrics={"agent_speech_duration": self.data.current_turn.agent_speech_duration})

            self.data.tts_start_time = None
            self.data.tts_first_byte_time = None
        elif self.data.tts_start_time:
            # If we have start time but no first byte time, just reset
            self.data.tts_start_time = None
            self.data.tts_first_byte_time = None
    
    def on_stt_response(self, duration: float, confidence: float):
        if not self.data.current_turn:
            self.start_new_interaction()
        
        if self.data.current_turn:
            self.data.current_turn.stt_duration = duration
            self.data.current_turn.stt_confidence = confidence
            logger.info(f"Stt duration {duration}, confidence {confidence}")

    def on_stt_metrics(self, metrics: Dict[str, Any]):
        if self.data.current_turn:
            self.data.current_turn.stt_input_tokens = metrics.get("input_tokens")
            self.data.current_turn.stt_output_tokens = metrics.get("output_tokens")
            self.data.current_turn.stt_total_tokens = metrics.get("total_tokens")

    def on_stt_start(self):
        """Called when STT processing starts"""
        # Don't overwrite STT timestamps if turn is locked (LLM has started)
        if self.data.turn_timestamps_locked:
            return
        self.data.stt_start_time = time.perf_counter()
        if self.data.current_turn:
            self.data.current_turn.stt_start_time = self.data.stt_start_time
    
    def on_stt_complete(self, user_text: str):
        """Called when STT processing completes"""
        # Don't overwrite STT timestamps if turn is locked (LLM has started)
        if self.data.turn_timestamps_locked:
            return
        if self.data.current_turn and self.data.current_turn.stt_preemptive_generation_enabled and self.data.current_turn.stt_preemptive_generation_occurred:
            logger.info("STT preemptive generation occurred, skipping stt complete")
            return
        if self.data.stt_start_time:
            stt_end_time = time.perf_counter()
            stt_latency = stt_end_time - self.data.stt_start_time
            if self.data.current_turn:
                self.data.current_turn.stt_transcript = user_text
                self.data.current_turn.stt_end_time = stt_end_time
                self.data.current_turn.stt_latency = self._round_latency(stt_latency)
                logger.info(f"stt latency: {self.data.current_turn.stt_latency}ms")

                if self.playground:
                    self.playground_manager.send_cascading_metrics(metrics={"stt_latency": self.data.current_turn.stt_latency})
                
            self.data.stt_start_time = None
    
    def on_knowledge_base_start(self):
        """Called when knowledge base processing starts"""
        self.data.knowledge_base_start_time = time.perf_counter()
        if self.data.current_turn:
            self.data.current_turn.kb_start_time = self.data.knowledge_base_start_time
    
    def on_knowledge_base_complete(self, documents: List[str], scores: List[float], record_id: List[str] = None):
        """Called when knowledge base processing completes"""
        if self.data.knowledge_base_start_time:
            knowledge_base_end_time = time.perf_counter()
            kb_retrieval_latency = knowledge_base_end_time - self.data.knowledge_base_start_time
            if self.data.current_turn:
                self.data.current_turn.kb_documents = documents
                self.data.current_turn.kb_scores = scores
                self.data.current_turn.kb_record_ids = record_id
                self.data.current_turn.kb_end_time = knowledge_base_end_time
                self.data.current_turn.kb_retrieval_latency = self._round_latency(kb_retrieval_latency)
                logger.info(f"knowledge base retrieval latency: {self.data.current_turn.kb_retrieval_latency}ms")

                if self.playground:
                    self.playground_manager.send_cascading_metrics(metrics={"kb_retrieval_latency": self.data.current_turn.kb_retrieval_latency})
                
            self.data.knowledge_base_start_time = None

    def on_llm_start(self):
        """Called when LLM processing starts"""
        self.data.llm_start_time = time.perf_counter()
        # Lock timestamps to prevent STT/EOU overwrites from subsequent events
        self.data.turn_timestamps_locked = True
        
        if self.data.current_turn:
            self.data.current_turn.llm_start_time = self.data.llm_start_time
    
    def on_llm_input(self, text: str):
        """Record the actual text sent to LLM"""
        if self.data.current_turn:
            self.data.current_turn.llm_input = text
    
    def on_llm_complete(self):
        """Called when LLM processing completes"""
        if self.data.llm_start_time:
            llm_end_time = time.perf_counter()
            llm_duration = llm_end_time - self.data.llm_start_time
            if self.data.current_turn:
                self.data.current_turn.llm_end_time = llm_end_time
                self.data.current_turn.llm_duration = self._round_latency(llm_duration)
                logger.info(f"llm duration: {self.data.current_turn.llm_duration}ms")

                if self.playground:
                    self.playground_manager.send_cascading_metrics(metrics={"llm_duration": self.data.current_turn.llm_duration})
                
            self.data.llm_start_time = None
    
    def set_llm_input(self, text: str):
        """Record the actual text sent to LLM"""
        if self.data.current_turn:
            self.data.current_turn.llm_input = text

    def on_tts_start(self):
        """Called when TTS processing starts"""
        self.data.tts_start_time = time.perf_counter()
        self.data.tts_first_byte_time = None
        if self.data.current_turn:
            self.data.current_turn.tts_start_time = self.data.tts_start_time

    
    def on_tts_first_byte(self):
        """Called when TTS produces first audio byte - this is our TTS latency"""
        if self.data.tts_start_time:
            now = time.perf_counter()
            # ttfb = now - self.data.tts_start_time // no need to take the difference as we are using the start time of the tts span
            if self.data.current_turn:
                self.data.current_turn.tts_end_time = now
                self.data.current_turn.ttfb = self._round_latency((self.data.current_turn.tts_end_time - self.data.tts_start_time))
                logger.info(f"tts ttfb: {self.data.current_turn.ttfb}ms")

                if self.playground:
                    self.playground_manager.send_cascading_metrics(metrics={"ttfb": self.data.current_turn.ttfb})
            
            self.data.tts_first_byte_time = now
    
    def on_eou_start(self):
        """Called when EOU (End of Utterance) processing starts"""
        # Don't overwrite EOU timestamps if turn is locked (LLM has started)
        if self.data.turn_timestamps_locked:
            return
        self.data.eou_start_time = time.perf_counter()
        if self.data.current_turn:
            self.data.current_turn.eou_start_time = self.data.eou_start_time
            
    
    def on_eou_complete(self):
        """Called when EOU processing completes"""
        # Don't overwrite EOU timestamps if turn is locked (LLM has started)
        if self.data.turn_timestamps_locked:
            return
        if self.data.eou_start_time:
            eou_end_time = time.perf_counter()
            eou_latency = eou_end_time - self.data.eou_start_time
            if self.data.current_turn:
                self.data.current_turn.eou_end_time = eou_end_time
                self.data.current_turn.eou_latency = self._round_latency(eou_latency)
                # self._end_timeline_event("eou_processing", eou_end_time)
                logger.info(f"eou latency: {self.data.current_turn.eou_latency}ms")

                if self.playground:
                    self.playground_manager.send_cascading_metrics(metrics={"eou_latency": self.data.current_turn.eou_latency})
            
            self.data.eou_start_time = None

    def on_wait_for_additional_speech(self, duration: float, eou_probability: float):
        """Called when waiting for additional speech"""
        if self.data.current_turn:
            self.data.current_turn.wait_for_additional_speech_duration = self._round_latency(duration)
            self.data.current_turn.waited_for_additional_speech = True
            self.data.current_turn.eou_probability = eou_probability
            logger.info(f"wait for additional speech duration: {self.data.current_turn.wait_for_additional_speech_duration}ms")

            if self.playground:
                self.playground_manager.send_cascading_metrics(metrics={"wait_for_additional_speech_duration": self.data.current_turn.wait_for_additional_speech_duration})

    def set_user_transcript(self, transcript: str):
        """Set the user transcript for the current turn and update timeline"""
        if self.data.current_turn:
            self.data.current_turn.user_speech = transcript
            logger.info(f"user input speech: {transcript}")
            user_speech_events = [event for event in self.data.current_turn.timeline 
                                if event.event_type == "user_speech"]
            if self.playground:
                self.playground_manager.send_cascading_metrics(metrics={"user_speech": self.data.current_turn.user_speech})

            if user_speech_events:
                most_recent_event = user_speech_events[-1]
                most_recent_event.text = transcript
            else:
                current_time = time.perf_counter()
                self._start_timeline_event("user_speech", current_time)
                if self.data.current_turn.timeline:
                    self.data.current_turn.timeline[-1].text = transcript
    
    def set_agent_response(self, response: str):
        """Set the agent response for the current turn and update timeline"""
        if self.data.current_turn:
            self.data.current_turn.agent_speech = response
            logger.info(f"agent output speech: {response}")
            if not any(event.event_type == "agent_speech" for event in self.data.current_turn.timeline):
                current_time = time.perf_counter()
                self._start_timeline_event("agent_speech", current_time)
            
            self._update_timeline_event_text("agent_speech", response)

            if self.playground:
                self.playground_manager.send_cascading_metrics(metrics={"agent_speech": self.data.current_turn.agent_speech})
    
    def add_function_tool_call(self, tool_name: str):
        """Track when a function tool is called in the current turn"""
        if self.data.current_turn:
            self.data.current_turn.function_tools_called.append(tool_name)
            tool_timestamp = {
                "tool_name": tool_name,
                "timestamp": time.perf_counter(),
                "readable_time": time.strftime("%H:%M:%S", time.localtime())
            }
            self.data.current_turn.function_tool_timestamps.append(tool_timestamp)

            if self.playground:
                self.playground_manager.send_cascading_metrics(metrics={"function_tool_timestamps": self.data.current_turn.function_tool_timestamps})

    def add_error(self, source: str, message: str):
        """Add an error to the current turn"""
        if self.data.current_turn:
            self.data.current_turn.errors.append({
                "source": source,
                "message": message,
                "timestamp": time.time()
            })

            if self.playground:
                self.playground_manager.send_cascading_metrics(metrics={"errors": self.data.current_turn.errors})

    def set_a2a_handoff(self):
        """Set the A2A enabled and handoff occurred flags for the current turn in A2A scenarios."""
        if self.data.current_turn:
            self.data.current_turn.is_a2a_enabled = True
            self.data.current_turn.handoff_occurred = True

            if self.playground:
                self.playground_manager.send_cascading_metrics(metrics={"handoff_occurred": self.data.current_turn.handoff_occurred})
    
    def set_llm_usage(self, usage: Dict[str, int]):
        """Set token usage and calculate TPS"""
        if not self.data.current_turn or not usage:
            return

        if self.data.current_turn:
            self.data.current_turn.prompt_tokens = usage.get("prompt_tokens")
            self.data.current_turn.completion_tokens = usage.get("completion_tokens")
            self.data.current_turn.total_tokens = usage.get("total_tokens")
            self.data.current_turn.prompt_cached_tokens = usage.get("prompt_cached_tokens")

        if self.data.current_turn and self.data.current_turn.llm_duration and self.data.current_turn.llm_duration > 0 and self.data.current_turn.completion_tokens > 0:
            latency_seconds = self.data.current_turn.llm_duration / 1000
            self.data.current_turn.tokens_per_second = round(self.data.current_turn.completion_tokens / latency_seconds, 2)

            if self.playground:
                self.playground_manager.send_cascading_metrics(metrics={"prompt_tokens": self.data.current_turn.prompt_tokens})
                self.playground_manager.send_cascading_metrics(metrics={"completion_tokens": self.data.current_turn.completion_tokens})
                self.playground_manager.send_cascading_metrics(metrics={"total_tokens": self.data.current_turn.total_tokens})
                self.playground_manager.send_cascading_metrics(metrics={"prompt_cached_tokens": self.data.current_turn.prompt_cached_tokens})
                self.playground_manager.send_cascading_metrics(metrics={"tokens_per_second": self.data.current_turn.tokens_per_second})
    
    def add_tts_characters(self, count: int):
        """Add to the total character count for the current turn"""
        if self.data.current_turn:
            if self.data.current_turn.tts_characters:
                self.data.current_turn.tts_characters += count
            else:
                self.data.current_turn.tts_characters = count

            if self.playground:
                self.playground_manager.send_cascading_metrics(metrics={"tts_characters": self.data.current_turn.tts_characters})
    
    def on_stt_preflight_end(self):
        """Called when STT preflight event received"""
        if self.data.current_turn and self.data.current_turn.stt_start_time:
            self.data.current_turn.stt_preflight_end_time = time.perf_counter()
            self.data.current_turn.stt_preflight_latency = self._round_latency(self.data.current_turn.stt_preflight_end_time - self.data.current_turn.stt_start_time)
            logger.info(f"stt preflight latency: {self.data.current_turn.stt_preflight_latency}ms")

            if self.playground:
                self.playground_manager.send_cascading_metrics(metrics={"stt_preflight_latency": self.data.current_turn.stt_preflight_latency})

    def on_stt_interim_end(self):
        """Called when STT interim event received"""
        if self.data.current_turn and self.data.current_turn.stt_start_time:
            self.data.current_turn.stt_interim_end_time = time.perf_counter()
            self.data.current_turn.stt_interim_latency = self._round_latency(self.data.current_turn.stt_interim_end_time - self.data.current_turn.stt_start_time)
            logger.info(f"stt interim latency: {self.data.current_turn.stt_interim_latency}ms")
            
            if self.playground:
                self.playground_manager.send_cascading_metrics(metrics={"stt_interim_latency": self.data.current_turn.stt_interim_latency})
    
    def on_llm_first_token(self):
        """Called when LLM first token received"""
        if self.data.current_turn:
            self.data.current_turn.llm_first_token_time = time.perf_counter()
            self.data.current_turn.llm_ttft = self._round_latency(self.data.current_turn.llm_first_token_time - self.data.current_turn.llm_start_time)
            logger.info(f"llm ttft: {self.data.current_turn.llm_ttft}ms")

            if self.playground:
                self.playground_manager.send_cascading_metrics(metrics={"llm_ttft": self.data.current_turn.llm_ttft})
    
    def set_preemptive_generation_enabled(self, enabled: bool):
        if self.data:
            self.data.stt_preemptive_generation_enabled = enabled
    
    def on_stt_preemptive_generation(self, text: str, match: bool):
        if self.data.current_turn:
            self.data.current_turn.stt_preflight_transcript = text
            self.data.current_turn.stt_preemptive_generation_occurred = match
    
    def config_vad(self, min_silence_duration: float = None, min_speech_duration: float = None, threshold: float = None):
        """Configure VAD parameters for metrics tracking"""
        if self.data:
            if min_silence_duration is not None:
                self.data.vad_min_silence_duration = min_silence_duration
            if min_speech_duration is not None:
                self.data.vad_min_speech_duration = min_speech_duration
            if threshold is not None:
                self.data.vad_threshold = threshold

    def on_vad_end_of_speech(self):
        """Called when VAD detects end of speech"""
        if self.data.current_turn:
            self.data.current_turn.vad_end_of_speech_time = time.perf_counter()
            logger.info(f"VAD end of speech detected at {self.data.current_turn.vad_end_of_speech_time}")
    
    
    def set_recording_started(self, started: bool):
        self.data.recording_started = started
    
    def set_recording_stopped(self, stopped: bool):
        self.data.recording_stopped = stopped

    def set_metrics(self, min_speech_wait_timeout: float, max_speech_wait_timeout: float):
        if self.data:
            self.data.min_speech_wait_timeout = min_speech_wait_timeout
            self.data.max_speech_wait_timeout = max_speech_wait_timeout

    def set_playground_manager(self, manager: Optional["PlaygroundManager"]):
        self.playground = True
        self.playground_manager = manager

    # Background Audio tracking methods
    def on_background_audio_start(self, file_path: str = None, looping: bool = False):
        """Called when background audio starts playing"""
        now = time.perf_counter()
        self.data.background_audio_start_time = now
        # Extract just the filename, not the full path
        file_name = os.path.basename(file_path) if file_path else None
        if self.data.current_turn:
            self.data.current_turn.background_audio_file_path = file_name
            self.data.current_turn.background_audio_looping = looping
            self._start_timeline_event("background_audio", now)
        # Create session-level span for background audio start
        if self.traces_flow_manager:
            self.traces_flow_manager.create_background_audio_start_span(file_path=file_name, looping=looping, start_time=now)
        logger.info(f"Background audio started: file={file_name}, looping={looping}")

    def on_background_audio_stop(self):
        """Called when background audio stops"""
        now = time.perf_counter()
        self.data.background_audio_end_time = now
        if self.data.current_turn:
            self._end_timeline_event("background_audio", now)
        # Create session-level span for background audio stop
        if self.traces_flow_manager:
            self.traces_flow_manager.create_background_audio_stop_span(end_time=now)
        logger.info(f"Background audio stopped")

    # Thinking Audio tracking methods
    def on_thinking_audio_start(self, file_path: str = None, looping: bool = True, override_thinking: bool = True):
        """Called when thinking audio starts playing"""
        now = time.perf_counter()
        self.data.thinking_audio_start_time = now
        # Extract just the filename, not the full path
        file_name = os.path.basename(file_path) if file_path else None
        if self.data.current_turn:
            self.data.current_turn.thinking_audio_file_path = file_name
            self.data.current_turn.thinking_audio_looping = looping
            self.data.current_turn.thinking_audio_override_thinking = override_thinking
            self._start_timeline_event("thinking_audio", now)
        logger.info(f"Thinking audio started: file={file_name}, looping={looping}, override_thinking={override_thinking}")

    def on_thinking_audio_stop(self):
        """Called when thinking audio stops"""
        now = time.perf_counter()
        self.data.thinking_audio_end_time = now
        if self.data.current_turn:
            self._end_timeline_event("thinking_audio", now)
        logger.info(f"Thinking audio stopped")