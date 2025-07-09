import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from .models import TimelineEvent, InteractionMetrics, MetricsData
from .analytics import AnalyticsClient


class MetricsCollector:
    """Collects and tracks performance metrics for AI agent interactions"""
    
    def __init__(self):
        self.data = MetricsData()
        self.analytics_client = AnalyticsClient()
        
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
    
    def start_new_interaction(self, user_transcript: str = "") -> None:
        """Start tracking a new user-agent interaction"""
        if self.data.current_interaction:
            self.complete_current_interaction()
        
        self.data.total_interactions += 1
        interaction_id = self._generate_interaction_id()
        
        self.data.current_interaction = InteractionMetrics(
            interaction_id=interaction_id,
            system_instructions=self.data.system_instructions if self.data.total_interactions == 1 else ""
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
            self.data.interactions.append(self.data.current_interaction)
            
            interaction_data = asdict(self.data.current_interaction)
            interaction_data['timeline'] = [asdict(event) for event in self.data.current_interaction.timeline]
            
            self.analytics_client.send_interaction_analytics_safe(interaction_data)
            
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
                self.data.current_interaction.tts_latency = self._round_latency(total_tts_latency)
            self.data.tts_start_time = None
            self.data.tts_first_byte_time = None
    
    def on_stt_start(self):
        """Called when STT processing starts"""
        self.data.stt_start_time = time.perf_counter()
    
    def on_stt_complete(self):
        """Called when STT processing completes"""
        if self.data.stt_start_time:
            stt_latency = time.perf_counter() - self.data.stt_start_time
            if self.data.current_interaction:
                self.data.current_interaction.stt_latency = self._round_latency(stt_latency)
            self.data.stt_start_time = None
    
    def on_llm_start(self):
        """Called when LLM processing starts"""
        self.data.llm_start_time = time.perf_counter()
    
    def on_llm_complete(self):
        """Called when LLM processing completes"""
        if self.data.llm_start_time:
            llm_latency = time.perf_counter() - self.data.llm_start_time
            if self.data.current_interaction:
                self.data.current_interaction.llm_latency = self._round_latency(llm_latency)
            self.data.llm_start_time = None
    
    def on_tts_start(self):
        """Called when TTS processing starts"""
        self.data.tts_start_time = time.perf_counter()
        self.data.tts_first_byte_time = None
    
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
