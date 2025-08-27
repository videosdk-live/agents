import time
import hashlib
from typing import Dict, Optional, Any
from dataclasses import asdict
from opentelemetry.trace import Span
from .models import TimelineEvent, CascadingTurnData, CascadingMetricsData
from .analytics import AnalyticsClient
from .traces_flow import TracesFlowManager
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
            'user_speech_start_time': 'userSpeechStartTime',
            'user_speech_end_time': 'userSpeechEndTime',
            'stt_latency': 'sttLatency',
            'llm_latency': 'llmLatency',
            'tts_latency': 'ttsLatency',
            'eou_latency': 'eouLatency',
            'e2e_latency': 'e2eLatency',
            'function_tools_called': 'functionToolsCalled',
            'system_instructions': 'systemInstructions',
            'errors': 'errors',
            'function_tool_timestamps': 'functionToolTimestamps',
            'stt_start_time': 'sttStartTime',
            'stt_end_time': 'sttEndTime',
            'tts_start_time': 'ttsStartTime',
            'tts_end_time': 'ttsEndTime',
            'llm_start_time': 'llmStartTime',
            'llm_end_time': 'llmEndTime',
            'eou_start_time': 'eouStartTime',
            'eou_end_time': 'eouEndTime',
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
            'handoff_occurred': 'handOffOccurred'
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
        if turn.llm_latency:
            e2e_components.append(turn.llm_latency)
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
        tts_present = turn.tts_latency is not None  
        llm_present = turn.llm_latency is not None
        eou_present = turn.eou_latency is not None
        
        if not any([stt_present, tts_present, llm_present, eou_present]):
            return False
        
        present_latencies = []
        if stt_present:
            present_latencies.append("STT")
        if tts_present:
            present_latencies.append("TTS") 
        if llm_present:
            present_latencies.append("LLM")
        if eou_present:
            present_latencies.append("EOU")
        
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
            eou_provider_class=self.data.eou_provider_class,
            eou_model_name=self.data.eou_model_name
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

            if not self._validate_interaction_has_required_latencies(self.data.current_turn):
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
            interaction_data = asdict(self.data.current_turn)
            interaction_data['timeline'] = [asdict(event) for event in self.data.current_turn.timeline]
            transformed_data = self._transform_to_camel_case(interaction_data)
            # transformed_data = self._intify_latencies_and_timestamps(transformed_data)

            always_remove_fields = [
                'errors',
                'functionToolTimestamps',
                'sttStartTime', 'sttEndTime',
                'ttsStartTime', 'ttsEndTime',
                'llmStartTime', 'llmEndTime',
                'eouStartTime', 'eouEndTime',
                'is_a2a_enabled',
                "interactionId",
                "timestamp"
            ]

            if not self.data.current_turn.is_a2a_enabled: 
                always_remove_fields.append("handOffOccurred")

            for field in always_remove_fields:
                if field in transformed_data:
                    del transformed_data[field]

            if len(self.data.turns) > 1: 
                provider_fields = [
                    'systemInstructions',
                    'llmProviderClass', 'llmModelName',
                    'sttProviderClass', 'sttModelName',
                    'ttsProviderClass', 'ttsModelName'
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
            self.pending_user_start_time = None
    
    def on_interrupted(self):
        """Called when the user interrupts the agent"""
        if self.data.is_agent_speaking:
            self.data.total_interruptions += 1
            if self.data.current_turn:
                self.data.current_turn.interrupted = True
                logger.info(f"User interrupted the agent. Total interruptions: {self.data.total_interruptions}")
    
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
        
        if self.data.current_turn:
            self.data.current_turn.user_speech_end_time = self.data.user_speech_end_time
            self._end_timeline_event("user_speech", self.data.user_speech_end_time)
    
    def on_agent_speech_start(self):
        """Called when agent starts speaking (actual audio output)"""
        self.data.is_agent_speaking = True
        self.data.agent_speech_start_time = time.perf_counter()
        
        if self.data.current_turn:
            if not any(event.event_type == "agent_speech" and event.end_time is None for event in self.data.current_turn.timeline):
                self._start_timeline_event("agent_speech", self.data.agent_speech_start_time)
    
    def on_agent_speech_end(self):
        """Called when agent stops speaking"""
        self.data.is_agent_speaking = False
        agent_speech_end_time = time.perf_counter()
        
        if self.data.current_turn:
            self._end_timeline_event("agent_speech", agent_speech_end_time)
                
        if self.data.tts_start_time and self.data.tts_first_byte_time:
            total_tts_latency = self.data.tts_first_byte_time - self.data.tts_start_time
            if self.data.current_turn:
                self.data.current_turn.tts_end_time = agent_speech_end_time
                self.data.current_turn.tts_latency = self._round_latency(total_tts_latency)
            self.data.tts_start_time = None
            self.data.tts_first_byte_time = None
        elif self.data.tts_start_time:
            # If we have start time but no first byte time, just reset
            self.data.tts_start_time = None
            self.data.tts_first_byte_time = None
    
    def on_stt_start(self):
        """Called when STT processing starts"""
        self.data.stt_start_time = time.perf_counter()
        if self.data.current_turn:
            self.data.current_turn.stt_start_time = self.data.stt_start_time
    
    def on_stt_complete(self):
        """Called when STT processing completes"""
        
        if self.data.stt_start_time:
            stt_end_time = time.perf_counter()
            stt_latency = stt_end_time - self.data.stt_start_time
            if self.data.current_turn:
                self.data.current_turn.stt_end_time = stt_end_time
                self.data.current_turn.stt_latency = self._round_latency(stt_latency)
                logger.info(f"stt latency: {self.data.current_turn.stt_latency}ms")
            self.data.stt_start_time = None
    
    def on_llm_start(self):
        """Called when LLM processing starts"""
        self.data.llm_start_time = time.perf_counter()
        
        if self.data.current_turn:
            self.data.current_turn.llm_start_time = self.data.llm_start_time
    
    def on_llm_complete(self):
        """Called when LLM processing completes"""
        if self.data.llm_start_time:
            llm_end_time = time.perf_counter()
            llm_latency = llm_end_time - self.data.llm_start_time
            if self.data.current_turn:
                self.data.current_turn.llm_end_time = llm_end_time
                self.data.current_turn.llm_latency = self._round_latency(llm_latency)
                logger.info(f"llm latency: {self.data.current_turn.llm_latency}ms")
            self.data.llm_start_time = None
    
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
                self.data.current_turn.ttfb = now
                logger.info(f"tts ttfb: {(self.data.current_turn.ttfb - self.data.tts_start_time) * 1000}ms")
            self.data.tts_first_byte_time = now
    
    def on_eou_start(self):
        """Called when EOU (End of Utterance) processing starts"""
        self.data.eou_start_time = time.perf_counter()
        if self.data.current_turn:
            self.data.current_turn.eou_start_time = self.data.eou_start_time
            
    
    def on_eou_complete(self):
        """Called when EOU processing completes"""
        if self.data.eou_start_time:
            eou_end_time = time.perf_counter()
            eou_latency = eou_end_time - self.data.eou_start_time
            if self.data.current_turn:
                self.data.current_turn.eou_end_time = eou_end_time
                self.data.current_turn.eou_latency = self._round_latency(eou_latency)
                # self._end_timeline_event("eou_processing", eou_end_time)
                logger.info(f"eou latency: {self.data.current_turn.eou_latency}ms")
            self.data.eou_start_time = None
    
    def set_user_transcript(self, transcript: str):
        """Set the user transcript for the current turn and update timeline"""
        if self.data.current_turn:
            logger.info(f"user input speech: {transcript}")
            user_speech_events = [event for event in self.data.current_turn.timeline 
                                if event.event_type == "user_speech"]
            
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
            logger.info(f"agent output speech: {response}")
            if not any(event.event_type == "agent_speech" for event in self.data.current_turn.timeline):
                current_time = time.perf_counter()
                self._start_timeline_event("agent_speech", current_time)
            
            self._update_timeline_event_text("agent_speech", response)
    
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

    def add_error(self, source: str, message: str):
        """Add an error to the current turn"""
        if self.data.current_turn:
            self.data.current_turn.errors.append({
                "source": source,
                "message": message,
                "timestamp": time.time()
            })

    def set_a2a_handoff(self):
        """Set the A2A enabled and handoff occurred flags for the current turn in A2A scenarios."""
        if self.data.current_turn:
            self.data.current_turn.is_a2a_enabled = True
            self.data.current_turn.handoff_occurred = True
