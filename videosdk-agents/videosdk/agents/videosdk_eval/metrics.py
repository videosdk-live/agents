from enum import Enum

class Metric(Enum):
    STT_LATENCY = "stt_latency"
    LLM_LATENCY = "llm_latency"
    TTS_LATENCY = "tts_latency"
    END_TO_END_LATENCY = "end_to_end_latency"


class JudgeMetric(Enum):
    REASONING = "reasoning"
    CONCLUSION = "conclusion"