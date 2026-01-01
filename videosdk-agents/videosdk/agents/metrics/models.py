import time
from typing import Dict, List, Optional, Any
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
    
    stt_latency: Optional[float] = None
    stt_start_time: Optional[float] = None
    stt_end_time: Optional[float] = None
    stt_duration: Optional[float] = None
    stt_preflight_end_time: Optional[float] = None
    stt_preflight_latency: Optional[float] = None
    stt_interim_end_time: Optional[float] = None
    stt_interim_latency: Optional[float] = None
    stt_preemptive_generation_occurred: bool = False
    stt_preemptive_generation_enabled: bool = False
    
    llm_latency: Optional[float] = None
    llm_start_time: Optional[float] = None
    llm_end_time: Optional[float] = None
    llm_duration: Optional[float] = None
    llm_first_token_time: Optional[float] = None
    llm_ttft: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    prompt_cached_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None

    
    tts_latency: Optional[float] = None 
    tts_start_time: Optional[float] = None
    tts_end_time: Optional[float] = None
    tts_duration: Optional[float] = None
    tts_characters: Optional[int] = None
    ttfb: Optional[float] = None
    
    eou_latency: Optional[float] = None
    eou_start_time: Optional[float] = None
    eou_end_time: Optional[float] = None
    
    function_tool_timestamps: List[Dict[str, Any]] = field(default_factory=list)
    
    e2e_latency: Optional[float] = None
    interrupted: bool = False
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
    eou_provider_class: str = ""
    eou_model_name: str = ""
    
    timeline: List[TimelineEvent] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    is_a2a_enabled: bool = False
    handoff_occurred: bool = False  

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
    user_input_start_time: Optional[float] = None
    is_agent_speaking: bool = False
    is_user_speaking: bool = False
    tts_first_byte_time: Optional[float] = None
    stt_preemptive_generation_enabled: bool = False
    
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
    

@dataclass
class RealtimeTurnData:
    """
    Captures a single turn between user and agent.
    Turns = one user utterance + one agent response.
    """
    session_id: Optional[str] = None
    provider_class_name: Optional[str] = None 
    provider_model_name: Optional[str] = None 
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
