import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import asdict, field, fields
from opentelemetry.trace import Span
from .models import TimelineEvent, InteractionMetrics, MetricsData
from .analytics import AnalyticsClient
from .traces_flow import TracesFlowManager


class MetricsCollector:
    """Collects and tracks performance metrics for AI agent interactions"""
    
    def __init__(self):
        self.data = MetricsData()
        self.analytics_client = AnalyticsClient()
        self.traces_flow_manager: Optional[TracesFlowManager] = None
        self.active_spans: Dict[str, Span] = {}
        
    def set_traces_flow_manager(self, manager: TracesFlowManager):
        """Set the TracesFlowManager instance"""
        self.traces_flow_manager = manager

    def _generate_interaction_id(self) -> str:
        """Generate a hash-based interaction ID"""
        timestamp = str(time.time())
        session_id = self.data.session_id or "default"
        interaction_count = str(self.data.total_interactions)
        
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
            'llm_provider_class': 'llmProviderClass',
            'llm_model_name': 'llmModelName',
            'stt_provider_class': 'sttProviderClass',
            'stt_model_name': 'sttModelName',
            'tts_provider_class': 'ttsProviderClass',
            'tts_model_name': 'ttsModelName',
            'hand_off_count': 'handOffCount'
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
    
    def _start_timeline_event(self, event_type: str, start_time: float) -> None:
        """Start a timeline event"""
        if self.data.current_interaction:
            event = TimelineEvent(
                event_type=event_type,
                start_time=start_time
            )
            self.data.current_interaction.timeline.append(event)
    
    def _end_timeline_event(self, event_type: str, end_time: float) -> None:
        """End a timeline event and calculate duration"""
        if self.data.current_interaction:
            for event in reversed(self.data.current_interaction.timeline):
                if event.event_type == event_type and event.end_time is None:
                    event.end_time = end_time
                    event.duration_ms = self._round_latency(end_time - event.start_time)
                    break
    
    def _update_timeline_event_text(self, event_type: str, text: str) -> None:
        """Update timeline event with text content"""
        if self.data.current_interaction:
            for event in reversed(self.data.current_interaction.timeline):
                if event.event_type == event_type and not event.text:
                    event.text = text
                    break
    
    def _calculate_e2e_metrics(self, interaction: InteractionMetrics) -> None:
        """Calculate E2E and E2ET latencies based on individual component latencies"""
        e2e_components = []
        if interaction.stt_latency:
            e2e_components.append(interaction.stt_latency)
        if interaction.llm_latency:
            e2e_components.append(interaction.llm_latency)
        if interaction.tts_latency: 
            e2e_components.append(interaction.tts_latency)
        
        if e2e_components:
            interaction.e2e_latency = round(sum(e2e_components), 4)
        
    def set_session_id(self, session_id: str):
        """Set the session ID for metrics tracking"""
        self.data.session_id = session_id
        self.analytics_client.set_session_id(session_id)
    
    def set_system_instructions(self, instructions: str):
        """Set the system instructions for this session"""
        self.data.system_instructions = instructions
    
    def set_provider_info(self, llm_provider: str = "", llm_model: str = "", 
                         stt_provider: str = "", stt_model: str = "",
                         tts_provider: str = "", tts_model: str = ""):
        """Set the provider class and model information for this session"""
        self.data.llm_provider_class = llm_provider
        self.data.llm_model_name = llm_model
        self.data.stt_provider_class = stt_provider
        self.data.stt_model_name = stt_model
        self.data.tts_provider_class = tts_provider
        self.data.tts_model_name = tts_model
    
    def start_new_interaction(self, user_transcript: str = "") -> None:
        """Start tracking a new user-agent interaction"""
        if self.data.current_interaction:
            self.complete_current_interaction()
        
        self.data.total_interactions += 1
        interaction_id = self._generate_interaction_id()
        
        self.data.current_interaction = InteractionMetrics(
            interaction_id=interaction_id,
            system_instructions=self.data.system_instructions if self.data.total_interactions == 1 else "",
            llm_provider_class=self.data.llm_provider_class if self.data.total_interactions == 1 else "",
            llm_model_name=self.data.llm_model_name if self.data.total_interactions == 1 else "",
            stt_provider_class=self.data.stt_provider_class if self.data.total_interactions == 1 else "",
            stt_model_name=self.data.stt_model_name if self.data.total_interactions == 1 else "",
            tts_provider_class=self.data.tts_provider_class if self.data.total_interactions == 1 else "",
            tts_model_name=self.data.tts_model_name if self.data.total_interactions == 1 else ""
        )
        
        if self.data.is_user_speaking and self.data.user_input_start_time:
            self.data.current_interaction.user_speech_start_time = self.data.user_input_start_time
        elif user_transcript and not self.data.is_user_speaking:
            current_time = time.perf_counter()
            estimated_speech_duration = max(len(user_transcript) / 10, 1.0)
            estimated_start_time = current_time - estimated_speech_duration
            
            self.data.current_interaction.user_speech_start_time = estimated_start_time
            self.data.current_interaction.user_speech_end_time = current_time
            
            self._start_timeline_event("user_speech", estimated_start_time)
            self._end_timeline_event("user_speech", current_time)
        
        if user_transcript:
            self.set_user_transcript(user_transcript)
    
    def complete_current_interaction(self) -> None:
        """Complete and store the current interaction"""
        if self.data.current_interaction:
            self._calculate_e2e_metrics(self.data.current_interaction)

            if self.traces_flow_manager:
                self.traces_flow_manager.create_interaction_trace(self.data.current_interaction)

            self.data.interactions.append(self.data.current_interaction)
            interaction_data = asdict(self.data.current_interaction)
            interaction_data['timeline'] = [asdict(event) for event in self.data.current_interaction.timeline]
            transformed_data = self._transform_to_camel_case(interaction_data)

            always_remove_fields = [
                'errors',
                'functionToolTimestamps',
                'sttStartTime', 'sttEndTime',
                'ttsStartTime', 'ttsEndTime',
                'llmStartTime', 'llmEndTime',
                'is_a2a_enabled'
            ]

            if not self.data.current_interaction.is_a2a_enabled: 
                always_remove_fields.append("handoff_occurred")

            for field in always_remove_fields:
                if field in transformed_data:
                    del transformed_data[field]

            if len(self.data.interactions) > 1: 
                provider_fields = [
                    'systemInstructions',
                    'llmProviderClass', 'llmModelName',
                    'sttProviderClass', 'sttModelName',
                    'ttsProviderClass', 'ttsModelName'
                ]
                for field in provider_fields:
                    if field in transformed_data:
                        del transformed_data[field]

            interaction_payload = {
                "sessionId": self.data.session_id,           
                "data": [transformed_data]               
            }
            
            self.analytics_client.send_interaction_analytics_safe(interaction_payload)
            self.data.current_interaction = None
    
    def on_user_speech_start(self):
        """Called when user starts speaking"""
        if self.data.is_agent_speaking:
            self.data.total_interruptions += 1
            if self.data.current_interaction:
                self.data.current_interaction.interrupted = True
        
        self.data.is_user_speaking = True
        self.data.user_input_start_time = time.perf_counter()
        
        if self.data.current_interaction:
            self.data.current_interaction.user_speech_start_time = self.data.user_input_start_time
            self._start_timeline_event("user_speech", self.data.user_input_start_time)
    
    def on_user_speech_end(self):
        """Called when user stops speaking"""
        self.data.is_user_speaking = False
        self.data.user_speech_end_time = time.perf_counter()
        
        if self.data.current_interaction:
            self.data.current_interaction.user_speech_end_time = self.data.user_speech_end_time
            self._end_timeline_event("user_speech", self.data.user_speech_end_time)
    
    def on_agent_speech_start(self):
        """Called when agent starts speaking (actual audio output)"""
        self.data.is_agent_speaking = True
        self.data.agent_speech_start_time = time.perf_counter()
        
        if self.data.current_interaction:
            self._start_timeline_event("agent_speech", self.data.agent_speech_start_time)
    
    def on_agent_speech_end(self):
        """Called when agent stops speaking"""
        self.data.is_agent_speaking = False
        agent_speech_end_time = time.perf_counter()
        
        if self.data.current_interaction:
            self._end_timeline_event("agent_speech", agent_speech_end_time)
        
        if self.data.tts_start_time:
            total_tts_latency = agent_speech_end_time - self.data.tts_start_time
            if self.data.current_interaction:
                self.data.current_interaction.tts_end_time = agent_speech_end_time
                self.data.current_interaction.tts_latency = self._round_latency(total_tts_latency)
            self.data.tts_start_time = None
            self.data.tts_first_byte_time = None
    
    def on_stt_start(self):
        """Called when STT processing starts"""
        self.data.stt_start_time = time.perf_counter()
        if self.data.current_interaction:
            self.data.current_interaction.stt_start_time = self.data.stt_start_time
    
    def on_stt_complete(self):
        """Called when STT processing completes"""
        if self.data.stt_start_time:
            stt_end_time = time.perf_counter()
            stt_latency = stt_end_time - self.data.stt_start_time
            if self.data.current_interaction:
                self.data.current_interaction.stt_end_time = stt_end_time
                self.data.current_interaction.stt_latency = self._round_latency(stt_latency)
            self.data.stt_start_time = None
    
    def on_llm_start(self):
        """Called when LLM processing starts"""
        self.data.llm_start_time = time.perf_counter()
        
        if self.data.current_interaction:
            self.data.current_interaction.llm_start_time = self.data.llm_start_time
    
    def on_llm_complete(self):
        """Called when LLM processing completes"""
        if self.data.llm_start_time:
            llm_end_time = time.perf_counter()
            llm_latency = llm_end_time - self.data.llm_start_time
            if self.data.current_interaction:
                self.data.current_interaction.llm_end_time = llm_end_time
                self.data.current_interaction.llm_latency = self._round_latency(llm_latency)
            self.data.llm_start_time = None
    
    def on_tts_start(self):
        """Called when TTS processing starts"""
        self.data.tts_start_time = time.perf_counter()
        self.data.tts_first_byte_time = None
        if self.data.current_interaction:
            self.data.current_interaction.tts_start_time = self.data.tts_start_time
    
    def on_tts_first_byte(self):
        """Called when TTS produces first audio byte - this is our TTS latency"""
        if self.data.tts_start_time:
            now = time.perf_counter()
            ttfb = now - self.data.tts_start_time
            if self.data.current_interaction:
                self.data.current_interaction.ttfb = self._round_latency(ttfb)
            self.data.tts_first_byte_time = now
    
    def set_user_transcript(self, transcript: str):
        """Set the user transcript for the current interaction and update timeline"""
        if self.data.current_interaction:
            self._update_timeline_event_text("user_speech", transcript)
    
    def set_agent_response(self, response: str):
        """Set the agent response for the current interaction and update timeline"""
        if self.data.current_interaction:
            self._update_timeline_event_text("agent_speech", response)
    
    def add_function_tool_call(self, tool_name: str):
        """Track when a function tool is called in the current interaction"""
        if self.data.current_interaction:
            self.data.current_interaction.function_tools_called.append(tool_name)
            tool_timestamp = {
                "tool_name": tool_name,
                "timestamp": time.perf_counter(),
                "readable_time": time.strftime("%H:%M:%S", time.localtime())
            }
            self.data.current_interaction.function_tool_timestamps.append(tool_timestamp)

    def add_error(self, source: str, message: str):
        """Add an error to the current interaction"""
        if self.data.current_interaction:
            self.data.current_interaction.errors.append({
                "source": source,
                "message": message,
                "timestamp": time.time()
            })

    def set_a2a_handoff(self):
        """Set the A2A enabled and handoff occurred flags for the current interaction in A2A scenarios."""
        if self.data.current_interaction:
            self.data.current_interaction.is_a2a_enabled = True
            self.data.current_interaction.handoff_occurred = True
