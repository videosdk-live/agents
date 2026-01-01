from enum import Enum

class EvalMetric(Enum):
    STT_LATENCY = "stt_latency"
    LLM_LATENCY = "llm_ttft"
    TTS_LATENCY = "ttfb"
    END_TO_END_LATENCY = "e2e_latency"

class LLMAsJudgeMetric(Enum):
    REASONING = "reasoning: why did the agent respond in this way?"
    RELEVANCE = "relevance: was the agent's response relevant to the user's request?"
    CLARITY = "clarity: was the agent's response clear and easy to understand?"
    SCORE = "score: how would you rate the agent's response out of 10?"