import time
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field, asdict


@dataclass
class TimelineEvent:
    """Data structure for a single timeline event"""
    event_type: str 
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    text: str = "" 


@dataclass
class FallbackEvent:
    """Data structure for a fallback event when a provider fails and switches to backup"""
    component_type: str  # "STT", "LLM", or "TTS"
    temporary_disable_sec: float
    permanent_disable_after_attempts: int
    recovery_attempt: int
    message: str
    # Timing for the overall fallback event
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    # Original provider connection attempt
    original_provider_label: Optional[str] = None
    original_connection_start: Optional[float] = None
    original_connection_end: Optional[float] = None
    original_connection_duration_ms: Optional[float] = None
    # New provider connection attempt (if switched)
    new_provider_label: Optional[str] = None
    new_connection_start: Optional[float] = None
    new_connection_end: Optional[float] = None
    new_connection_duration_ms: Optional[float] = None


@dataclass
class CascadingTurnData:
    """Data structure for a single user-agent turn"""
    user_speech_start_time: Optional[float] = None
    user_speech_end_time: Optional[float] = None
    user_speech_duration: Optional[float] = None
    user_speech: Optional[str] = None
    llm_input: Optional[str] = None

    agent_speech_start_time: Optional[float] = None
    agent_speech_end_time: Optional[float] = None
    agent_speech_duration: Optional[float] = None
    agent_speech: Optional[str] = None
    
    kb_documents: Optional[List[str]] = None
    kb_scores: Optional[List[float]] = None
    kb_retrieval_latency: Optional[float] = None
    kb_start_time: Optional[float] = None
    kb_end_time: Optional[float] = None

    stt_confidence: Optional[float] = None
    
    stt_input_tokens: Optional[int] = 0
    stt_output_tokens: Optional[int] = 0
    stt_total_tokens: Optional[int] = 0

    stt_latency: Optional[float] = None
    stt_start_time: Optional[float] = None
    stt_end_time: Optional[float] = None
    stt_duration: Optional[float] = None
    stt_preflight_end_time: Optional[float] = None
    stt_preflight_latency: Optional[float] = None
    stt_interim_end_time: Optional[float] = None
    stt_interim_latency: Optional[float] = None
    stt_ttfw: Optional[float] = None  # Time to first word - locked after first interim
    stt_preemptive_generation_occurred: bool = False
    stt_transcript: Optional[str] = None
    stt_preflight_transcript: Optional[str] = None
    stt_preemptive_generation_enabled: bool = False
    
    llm_latency: Optional[float] = None
    llm_start_time: Optional[float] = None
    llm_end_time: Optional[float] = None
    llm_duration: Optional[float] = None
    llm_first_token_time: Optional[float] = None
    llm_ttft: Optional[float] = None
    prompt_tokens: Optional[int] = 0
    completion_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0
    prompt_cached_tokens: Optional[int] = 0
    tokens_per_second: Optional[float] = 0

    
    tts_latency: Optional[float] = None 
    tts_start_time: Optional[float] = None
    tts_end_time: Optional[float] = None
    tts_duration: Optional[float] = None
    tts_characters: Optional[int] = 0
    ttfb: Optional[float] = None
    
    eou_latency: Optional[float] = None
    eou_start_time: Optional[float] = None
    eou_end_time: Optional[float] = None
    eou_probability: Optional[float] = None
    min_speech_wait_timeout: Optional[float] = None
    max_speech_wait_timeout: Optional[float] = None
    eou_avg_delay: Optional[float] = None
    waited_for_additional_speech: bool = False
    wait_for_additional_speech_duration: Optional[float] = None
    
    interrupted: bool = False
    interrupt_words: Optional[int] = None
    interrupt_duration: Optional[float] = None
    interrupt_reason: List[str] = field(default_factory=list)
    interrupt_start_time: Optional[float] = None
    interrupt_end_time: Optional[float] = None

    is_false_interrupt: bool = False
    false_interrupt_duration: Optional[float] = None
    false_interrupt_words: Optional[int] = None
    false_interrupt_start_time: Optional[float] = None
    false_interrupt_end_time: Optional[float] = None
    resumed_after_false_interrupt: bool = False

    interrupt_mode: Literal["VAD_ONLY", "STT_ONLY", "HYBRID"] = "HYBRID"
    interrupt_min_duration: Optional[float] = None
    interrupt_min_words: Optional[int] = None
    false_interrupt_pause_duration: Optional[float] = None
    resume_on_false_interrupt: Optional[bool] = None

    function_tool_timestamps: List[Dict[str, Any]] = field(default_factory=list)
    
    e2e_latency: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    function_tools_called: List[str] = field(default_factory=list)
    system_instructions: str = ""
    
    llm_provider_class: str = ""
    llm_model_name: str = ""
    stt_provider_class: str = ""
    stt_model_name: str = ""
    tts_provider_class: str = ""
    tts_model_name: str = ""
    vad_provider_class: str = ""
    vad_model_name: str = ""
    vad_min_silence_duration: Optional[float] = None
    vad_min_speech_duration: Optional[float] = None
    vad_threshold: Optional[float] = None
    vad_end_of_speech_time: Optional[float] = None
    eou_provider_class: str = ""
    eou_model_name: str = ""
    
    timeline: List[TimelineEvent] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    fallback_events: List['FallbackEvent'] = field(default_factory=list)
    is_a2a_enabled: bool = False
    handoff_occurred: bool = False  
    
    # Background Audio attributes (no start/end times - can be played outside turn)
    background_audio_file_path: Optional[str] = None
    background_audio_looping: Optional[bool] = None
    
    # Thinking Audio attributes (no start/end times - can be played outside turn)
    thinking_audio_file_path: Optional[str] = None
    thinking_audio_looping: Optional[bool] = None
    thinking_audio_override_thinking: Optional[bool] = None

@dataclass
class CascadingMetricsData:
    """Data structure to hold all metrics for a session"""
    session_id: Optional[str] = None
    session_start_time: float = field(default_factory=time.time)
    system_instructions: str = ""
    total_interruptions: int = 0
    total_turns: int = 0
    turns: List[CascadingTurnData] = field(default_factory=list)
    current_turn: Optional[CascadingTurnData] = None
    user_speech_end_time: Optional[float] = None
    agent_speech_start_time: Optional[float] = None
    agent_speech_end_time: Optional[float] = None
    stt_start_time: Optional[float] = None
    llm_start_time: Optional[float] = None
    tts_start_time: Optional[float] = None
    eou_start_time: Optional[float] = None
    eou_probability: Optional[float] = None
    wait_for_additional_speech_duration: Optional[float] = None
    waited_for_additional_speech: bool = False,
    min_speech_wait_timeout: Optional[float] = None,
    max_speech_wait_timeout: Optional[float] = None,
    user_input_start_time: Optional[float] = None
    is_agent_speaking: bool = False
    is_user_speaking: bool = False
    tts_first_byte_time: Optional[float] = None
    stt_preemptive_generation_enabled: bool = False
    # Lock flag to prevent STT/EOU timestamps from being overwritten once LLM starts
    turn_timestamps_locked: bool = False

    is_interrupted: bool = False

    interrupt_mode: Literal["VAD_ONLY", "STT_ONLY", "HYBRID"] = "HYBRID"
    interrupt_min_duration: Optional[float] = None
    interrupt_min_words: Optional[int] = None
    false_interrupt_pause_duration: Optional[float] = None
    resume_on_false_interrupt: Optional[bool] = None

    is_false_interrupt: bool = False
    false_interrupt_duration: Optional[float] = None
    false_interrupt_words: Optional[int] = None
    false_interrupt_start_time: Optional[float] = None
    false_interrupt_end_time: Optional[float] = None
    resumed_after_false_interrupt: bool = False

    vad_min_silence_duration: Optional[float] = None
    vad_min_speech_duration: Optional[float] = None
    vad_threshold: Optional[float] = None

    llm_provider_class: str = ""
    llm_model_name: str = ""
    stt_provider_class: str = ""
    stt_model_name: str = ""
    tts_provider_class: str = ""
    tts_model_name: str = ""
    vad_provider_class: str = ""
    vad_model_name: str = ""
    eou_provider_class: str = ""
    eou_model_name: str = ""
    
    # Background Audio session-level tracking
    background_audio_start_time: Optional[float] = None
    background_audio_end_time: Optional[float] = None
    
    # Thinking Audio session-level tracking
    thinking_audio_start_time: Optional[float] = None
    thinking_audio_end_time: Optional[float] = None
    

@dataclass
class RealtimeTurnData:
    """
    Captures a single turn between user and agent.
    Turns = one user utterance + one agent response.
    """
    session_id: Optional[str] = None
    realtime_provider_class: Optional[str] = None 
    realtime_model_name: Optional[str] = None 
    system_instructions: Optional[str] = None 
    function_tools: Optional[List[str]] = None
    mcp_tools: Optional[List[str]] = None
    user_speech_start_time: Optional[float] = None
    user_speech_end_time: Optional[float] = None
    agent_speech_start_time: Optional[float] = None
    agent_speech_end_time: Optional[float] = None
    ttfb: Optional[float] = None
    thinking_delay: Optional[float] = None
    e2e_latency: Optional[float] = None
    agent_speech_duration: Optional[float] = None
    interrupted: bool = False
    function_tools_called: List[str] = field(default_factory=list)
    timeline: List[TimelineEvent] = field(default_factory=list)
    realtime_model_errors: List[Dict[str, Any]] = field(default_factory=list)
    is_a2a_enabled: bool = False
    handoff_occurred: bool = False 
    
    # Token details
    realtime_input_tokens: Optional[int] = 0
    realtime_total_tokens: Optional[int] = 0
    realtime_output_tokens: Optional[int] = 0

    realtime_input_text_tokens: Optional[int] = 0
    realtime_input_audio_tokens: Optional[int] = 0
    realtime_input_image_tokens: Optional[int] = 0
    realtime_input_cached_tokens: Optional[int] = 0

    realtime_thoughts_tokens: Optional[int] = 0

    realtime_cached_text_tokens: Optional[int] = 0
    realtime_cached_audio_tokens: Optional[int] = 0
    realtime_cached_image_tokens: Optional[int] = 0


    realtime_output_text_tokens: Optional[int] = 0
    realtime_output_audio_tokens: Optional[int] = 0
    realtime_output_image_tokens: Optional[int] = 0

    
    def compute_latencies(self):
        if self.user_speech_start_time and self.agent_speech_start_time:
            self.ttfb = (self.agent_speech_start_time - self.user_speech_start_time) * 1000
        if self.user_speech_end_time and self.agent_speech_start_time:
            self.thinking_delay = (self.agent_speech_start_time - self.user_speech_end_time) * 1000
        if self.user_speech_start_time and self.agent_speech_end_time:
            self.e2e_latency = (self.agent_speech_end_time - self.user_speech_start_time) * 1000
        if self.agent_speech_start_time and self.agent_speech_end_time:
            self.agent_speech_duration = (self.agent_speech_end_time - self.agent_speech_start_time) * 1000

    def to_dict(self) -> Dict:
        self.compute_latencies()
        return asdict(self)
