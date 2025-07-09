import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class TimelineEvent:
    """Data structure for a single timeline event"""
    event_type: str 
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    text: str = "" 


@dataclass
class InteractionMetrics:
    """Data structure for a single user-agent interaction"""
    interaction_id: str  
    user_speech_start_time: Optional[float] = None
    user_speech_end_time: Optional[float] = None
    stt_latency: Optional[float] = None
    llm_latency: Optional[float] = None
    tts_latency: Optional[float] = None 
    ttfb: Optional[float] = None
    e2e_latency: Optional[float] = None
    interrupted: bool = False
    timestamp: float = field(default_factory=time.time)
    function_tools_called: List[str] = field(default_factory=list)
    system_instructions: str = ""
    timeline: List[TimelineEvent] = field(default_factory=list)


@dataclass
class MetricsData:
    """Data structure to hold all metrics for a session"""
    session_id: Optional[str] = None
    session_start_time: float = field(default_factory=time.time)
    system_instructions: str = ""
    total_interruptions: int = 0
    total_interactions: int = 0
    interactions: List[InteractionMetrics] = field(default_factory=list)
    current_interaction: Optional[InteractionMetrics] = None
    user_speech_end_time: Optional[float] = None
    agent_speech_start_time: Optional[float] = None
    stt_start_time: Optional[float] = None
    llm_start_time: Optional[float] = None
    tts_start_time: Optional[float] = None
    user_input_start_time: Optional[float] = None
    is_agent_speaking: bool = False
    is_user_speaking: bool = False
    tts_first_byte_time: Optional[float] = None 