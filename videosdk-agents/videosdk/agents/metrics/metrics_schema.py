import time
from typing import Dict, List, Optional, Any, Literal, Union
from dataclasses import dataclass, field, asdict


@dataclass
class BaseComponentMetrics:
    """Base for component-specific metrics with provider and model identity."""
    id: str = ""
    provider_class: str = ""
    model_name: str = ""


@dataclass
class TimelineEvent:
    """Data structure for a single timeline event."""
    event_type: str = ""
    start_time: float = 0.0
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    text: str = ""


@dataclass
class FallbackEvent:
    """Fallback event when a provider fails and switches to backup."""
    component_type: str = ""
    temporary_disable_sec: float = 0.0
    permanent_disable_after_attempts: int = 0
    recovery_attempt: int = 0
    message: str = ""
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    original_provider_label: Optional[str] = None
    original_connection_start: Optional[float] = None
    original_connection_end: Optional[float] = None
    original_connection_duration_ms: Optional[float] = None
    new_provider_label: Optional[str] = None
    new_connection_start: Optional[float] = None
    new_connection_end: Optional[float] = None
    new_connection_duration_ms: Optional[float] = None
    is_recovery: bool = False


@dataclass
class VadMetrics(BaseComponentMetrics):
    """VAD (Voice Activity Detection) metrics."""
    vad_config: Optional[Dict[str, Any]] = None
    user_speech_start_time: Optional[float] = None
    user_speech_end_time: Optional[float] = None
    vad_min_silence_duration: Optional[float] = None
    vad_min_speech_duration: Optional[float] = None
    vad_threshold: Optional[float] = None
    vad_end_of_speech_time: Optional[float] = None


@dataclass
class SttMetrics(BaseComponentMetrics):
    """STT (Speech-to-Text) metrics."""
    stt_config: Optional[Dict[str, Any]] = None
    stt_confidence: Optional[float] = None
    stt_input_tokens: Optional[int] = None
    stt_output_tokens: Optional[int] = None
    stt_total_tokens: Optional[int] = None
    stt_latency: Optional[float] = None
    stt_start_time: Optional[float] = None
    stt_end_time: Optional[float] = None
    stt_duration: Optional[float] = None
    stt_preflight_end_time: Optional[float] = None
    stt_preflight_latency: Optional[float] = None
    stt_interim_end_time: Optional[float] = None
    stt_interim_latency: Optional[float] = None
    stt_ttfw: Optional[float] = None
    stt_preemptive_generation_occurred: bool = False
    stt_preemptive_generation_enabled: bool = False
    stt_transcript: Optional[str] = None
    stt_preflight_transcript: Optional[str] = None


@dataclass
class EouMetrics(BaseComponentMetrics):
    """EOU (End of Utterance) metrics."""
    eou_latency: Optional[float] = None
    eou_start_time: Optional[float] = None
    eou_end_time: Optional[float] = None
    eou_probability: Optional[float] = None
    min_speech_wait_timeout: Optional[float] = None
    max_speech_wait_timeout: Optional[float] = None
    eou_avg_delay: Optional[float] = None
    waited_for_additional_speech: bool = False
    wait_for_additional_speech_duration: Optional[float] = None


@dataclass
class KbMetrics(BaseComponentMetrics):
    """Knowledge base retrieval metrics."""
    kb_id: Optional[str] = None
    kb_documents: Optional[List[str]] = None
    kb_scores: Optional[List[float]] = None
    kb_retrieval_latency: Optional[float] = None
    kb_start_time: Optional[float] = None
    kb_end_time: Optional[float] = None


@dataclass
class LlmMetrics(BaseComponentMetrics):
    """LLM metrics."""
    llm_config: Optional[Dict[str, Any]] = None
    llm_input: Optional[str] = None
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


@dataclass
class TtsMetrics(BaseComponentMetrics):
    """TTS metrics."""
    tts_config: Optional[Dict[str, Any]] = None
    tts_latency: Optional[float] = None
    tts_start_time: Optional[float] = None
    tts_end_time: Optional[float] = None
    tts_duration: Optional[float] = None
    tts_characters: Optional[int] = 0
    tts_first_byte_time: Optional[float] = None
    ttfb: Optional[float] = None


@dataclass
class RealtimeMetrics(BaseComponentMetrics):
    """Realtime model (full s2s) metrics."""
    realtime_config: Optional[Dict[str, Any]] = None
    realtime_input_tokens: Optional[int] = None
    realtime_total_tokens: Optional[int] = None
    realtime_output_tokens: Optional[int] = None
    realtime_input_text_tokens: Optional[int] = None
    realtime_input_audio_tokens: Optional[int] = None
    realtime_input_image_tokens: Optional[int] = None
    realtime_input_cached_tokens: Optional[int] = None
    realtime_thoughts_tokens: Optional[int] = None
    realtime_cached_text_tokens: Optional[int] = None
    realtime_cached_audio_tokens: Optional[int] = None
    realtime_cached_image_tokens: Optional[int] = None
    realtime_output_text_tokens: Optional[int] = None
    realtime_output_audio_tokens: Optional[int] = None
    realtime_output_image_tokens: Optional[int] = None


@dataclass
class InterruptionMetrics:
    """Interruption and false-interrupt metrics for a turn."""
    false_interrupt_pause_duration: Optional[float] = None
    resume_on_false_interrupt: Optional[bool] = None
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
    interrupt_mode: Optional[Literal["VAD_ONLY", "STT_ONLY", "HYBRID"]] = None
    interrupt_min_duration: Optional[float] = None
    interrupt_min_words: Optional[int] = None


@dataclass
class FunctionToolMetrics:
    """Metrics for a single function (agent) tool call in a turn."""
    tool_name: str = ""
    tool_params: Union[Dict[str, Any], List[Any]] = field(default_factory=dict)
    tool_response: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    latency: Optional[float] = None


@dataclass
class McpToolMetrics:
    """Metrics for a single MCP tool call in a turn."""
    type: Literal["local", "http"] = "local"
    tool_url: Optional[str] = None  # endpoint URL for http, or endpoint identifier
    tool_params: Union[Dict[str, Any], List[Any]] = field(default_factory=dict)
    tool_response: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    latency: Optional[float] = None


@dataclass
class ParticipantMetrics:
    """Participant/connection metrics at session level."""
    participant_id: Optional[str] = None
    kind: Optional[Literal["agent", "user"]] = None
    sip_user: Optional[bool] = None
    join_time: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class SessionMetrics:
    """Session-level metrics."""
    session_id: Optional[str] = None
    room_id: Optional[str] = None
    system_instruction: str = ""
    components: List[str] = field(default_factory=list)
    pipeline_type: Optional[str] = None
    pipeline_mode: Optional[str] = None
    realtime_mode: Optional[str] = None
    session_start_time: float = field(default_factory=time.time)
    session_end_time: Optional[float] = None
    participant_metrics: List[ParticipantMetrics] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    provider_per_component: Dict[str, Dict[str, str]] = field(default_factory=dict)
    eou_config: Optional[Dict[str, Any]] = None
    interrupt_config: Optional[Dict[str, Any]] = None


@dataclass
class TurnMetrics:
    """Single turn with lists of per-component metrics."""
    turn_id: str = ""
    is_interrupted: bool = False
    user_speech_start_time: Optional[float] = None
    user_speech_end_time: Optional[float] = None
    user_speech_duration: Optional[float] = None
    user_speech: Optional[str] = None
    agent_speech_start_time: Optional[float] = None
    agent_speech_end_time: Optional[float] = None
    agent_speech_duration: Optional[float] = None
    agent_speech: Optional[str] = None
    e2e_latency: Optional[float] = None
    function_tools_called: List[str] = field(default_factory=list)
    function_tool_timestamps: List[Dict[str, Any]] = field(default_factory=list)
    is_a2a_enabled: bool = False
    handoff_occurred: bool = False
    errors: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    vad_metrics: List[VadMetrics] = field(default_factory=list)
    stt_metrics: List[SttMetrics] = field(default_factory=list)
    eou_metrics: List[EouMetrics] = field(default_factory=list)
    kb_metrics: List[KbMetrics] = field(default_factory=list)
    llm_metrics: List[LlmMetrics] = field(default_factory=list)
    tts_metrics: List[TtsMetrics] = field(default_factory=list)
    interruption_metrics: Optional[InterruptionMetrics] = None
    realtime_metrics: List[RealtimeMetrics] = field(default_factory=list)
    timeline_event_metrics: List[TimelineEvent] = field(default_factory=list)
    fallback_events: List[FallbackEvent] = field(default_factory=list)
    function_tool_metrics: List[FunctionToolMetrics] = field(default_factory=list)
    mcp_tool_metrics: List[McpToolMetrics] = field(default_factory=list)

    def compute_e2e_latency(self) -> None:
        """Calculate E2E latency by summing component latencies (STT + EOU + LLM TTFT + TTS)."""
        e2e_components = []

        if self.stt_metrics:
            stt = self.stt_metrics[-1]
            if stt.stt_latency is not None:
                e2e_components.append(stt.stt_latency)

        if self.eou_metrics:
            eou = self.eou_metrics[-1]
            if eou.eou_latency is not None:
                e2e_components.append(eou.eou_latency)

        if self.llm_metrics:
            llm = self.llm_metrics[-1]
            if llm.llm_ttft is not None:
                e2e_components.append(llm.llm_ttft)

        if self.tts_metrics:
            tts = self.tts_metrics[-1]
            if tts.tts_latency is not None:
                e2e_components.append(tts.tts_latency)

        if self.realtime_metrics:
            rt = self.realtime_metrics[-1]
            if hasattr(rt, "realtime_ttfb") and getattr(rt, "realtime_ttfb", None) is not None:
                e2e_components.append(rt.realtime_ttfb)

        if e2e_components:
            self.e2e_latency = round(sum(e2e_components), 4)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize turn metrics to a dictionary."""
        self.compute_e2e_latency()
        return asdict(self)