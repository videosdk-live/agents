"""
Metrics Transformer - Converts internal Python metrics to API server schema format.

Transforms TurnMetrics (with component metric lists) into the flat
agentInteractionSchema format expected by the API server.

Rules:
- Latencies & performance metrics: use the LAST VALID metric from the list.
- Cost & usage metrics (tokens, characters, duration): AGGREGATE (sum) across all items.
- Keys are converted from snake_case to camelCase.
- All latencies/durations are converted from seconds to milliseconds, rounded to 2 decimals.
- Negative numeric values are clamped to 0.
"""

from typing import Dict, Any, List, Optional
from dataclasses import asdict
import logging

from .metrics_schema import (
    TurnMetrics, SessionMetrics,
    SttMetrics, LlmMetrics, TtsMetrics, EouMetrics,
    VadMetrics, RealtimeMetrics, KbMetrics,
    TimelineEvent
)

logger = logging.getLogger(__name__)


def _to_ms(value: Any) -> float:
    """Convert a value from seconds to milliseconds, round to 2 decimals, clamp negatives to 0."""
    if value is None:
        return 0
    ms = round(float(value) * 1000, 2)
    return max(ms, 0)


def _clamp(value: Any) -> Any:
    """Clamp a numeric value: if negative, return 0."""
    if isinstance(value, (int, float)) and value < 0:
        return 0
    return value


def _safe_val(value: Any, default: Any = 0) -> Any:
    """Return value if not None, else default."""
    return value if value is not None else default


def _sum_attr(items: list, attr: str) -> Any:
    """Sum a numeric attribute across a list of dataclass instances, skipping None."""
    total = sum(getattr(item, attr) or 0 for item in items)
    return total if total else 0


def _last_valid(items: list, attr: str) -> Any:
    """Get the last item in the list that has a non-None value for the given attribute."""
    for item in reversed(items):
        val = getattr(item, attr, None)
        if val is not None:
            return val
    return None


def _last_valid_item(items: list, attr: str) -> Any:
    """Get the last item object that has a non-None value for the given attribute."""
    for item in reversed(items):
        val = getattr(item, attr, None)
        if val is not None:
            return item
    return None


def transform_turn(turn: TurnMetrics) -> Dict[str, Any]:
    """
    Transform a TurnMetrics object into the flat agentInteractionSchema format.

    Latencies come from the last valid component metric.
    Costs/tokens are aggregated (summed) across all component metrics.
    All latencies/durations are converted from seconds → milliseconds.
    Negative values are clamped to 0.
    """
    result: Dict[str, Any] = {}

    # ── Turn-level fields (direct mapping) ──
    result["userSpeechStartTime"] = _safe_val(turn.user_speech_start_time)
    result["userSpeechEndTime"] = _safe_val(turn.user_speech_end_time)
    result["agentSpeechStartTime"] = _safe_val(turn.agent_speech_start_time)
    result["agentSpeechEndTime"] = _safe_val(turn.agent_speech_end_time)
    result["agentSpeech"] = turn.agent_speech or ""
    result["userSpeechDuration"] = _to_ms(turn.user_speech_duration)
    result["agentSpeechDuration"] = _to_ms(turn.agent_speech_duration)
    result["e2eLatency"] = round(max(turn.e2e_latency or 0, 0), 2)  # Already in ms from tracker
    result["interrupted"] = turn.is_interrupted
    result["handOffOccurred"] = turn.handoff_occurred
    result["handedOffTo"] = ""  # Not tracked in current schema
    result["functionToolsCalled"] = turn.function_tools_called or []
    result["mcpToolsCalled"] = [m.tool_url or "" for m in turn.mcp_tool_metrics] if turn.mcp_tool_metrics else []
    result["sipCallTransferOccurred"] = False  # Not tracked in current schema
    result["sipCallTransferredTo"] = ""

    # ── STT metrics ──
    stt_list = turn.stt_metrics or []
    if stt_list:
        # Provider info from last item
        result["sttProviderClass"] = stt_list[-1].provider_class or ""
        result["sttModelName"] = stt_list[-1].model_name or ""

        # Latencies: last valid → convert to ms
        result["sttLatency"] = _to_ms(_last_valid(stt_list, "stt_latency"))
        result["sttPreflightLatency"] = _to_ms(_last_valid(stt_list, "stt_preflight_latency"))
        result["sttInterimLatency"] = _to_ms(_last_valid(stt_list, "stt_interim_latency"))
        result["sttConfidence"] = _safe_val(_last_valid(stt_list, "stt_confidence"))
        result["sttPreemptiveGenerationEnabled"] = any(s.stt_preemptive_generation_enabled for s in stt_list)
        result["sttPreemptiveGenerationOccurred"] = any(s.stt_preemptive_generation_occurred for s in stt_list)

        # Aggregated: duration (→ ms), tokens
        result["sttDuration"] = round(max(_sum_attr(stt_list, "stt_duration"), 0), 2)  # Keep in seconds
        result["sttInputTokens"] = _sum_attr(stt_list, "stt_input_tokens")
        result["sttOutputTokens"] = _sum_attr(stt_list, "stt_output_tokens")
        result["sttTotalTokens"] = _sum_attr(stt_list, "stt_total_tokens")
    else:
        result["sttProviderClass"] = ""
        result["sttModelName"] = ""
        result["sttLatency"] = 0
        result["sttPreflightLatency"] = 0
        result["sttInterimLatency"] = 0
        result["sttConfidence"] = 0
        result["sttDuration"] = 0
        result["sttPreemptiveGenerationEnabled"] = False
        result["sttPreemptiveGenerationOccurred"] = False
        result["sttInputTokens"] = 0
        result["sttOutputTokens"] = 0
        result["sttTotalTokens"] = 0

    # ── EOU metrics ──
    eou_list = turn.eou_metrics or []
    if eou_list:
        result["eouProviderClass"] = eou_list[-1].provider_class or ""
        result["eouModelName"] = eou_list[-1].model_name or ""
        result["eouLatency"] = _to_ms(_last_valid(eou_list, "eou_latency"))
    else:
        result["eouProviderClass"] = ""
        result["eouModelName"] = ""
        result["eouLatency"] = 0

    # ── VAD metrics ──
    vad_list = turn.vad_metrics or []
    if vad_list:
        result["vadProviderClass"] = vad_list[-1].provider_class or ""
        result["vadModelName"] = vad_list[-1].model_name or ""
    else:
        result["vadProviderClass"] = ""
        result["vadModelName"] = ""

    # ── KB metrics ──
    kb_list = turn.kb_metrics or []
    if kb_list:
        # Take last KB result
        last_kb = kb_list[-1]
        result["kbDocuments"] = last_kb.kb_documents or []
        result["kbScores"] = last_kb.kb_scores or []
        result["kbRetrievalLatency"] = _to_ms(last_kb.kb_retrieval_latency)
    else:
        result["kbDocuments"] = []
        result["kbScores"] = []
        result["kbRetrievalLatency"] = 0

    # ── LLM metrics ──
    llm_list = turn.llm_metrics or []
    if llm_list:
        # Provider info from last item
        result["llmProviderClass"] = llm_list[-1].provider_class or ""
        result["llmModelName"] = llm_list[-1].model_name or ""
        result["llmInput"] = _last_valid(llm_list, "llm_input") or ""

        # Latencies: last valid → convert to ms
        result["llmLatency"] = _to_ms(_last_valid(llm_list, "llm_latency"))
        result["llmDuration"] = _to_ms(_last_valid(llm_list, "llm_duration"))
        result["ttft"] = _to_ms(_last_valid(llm_list, "llm_ttft"))
        result["tokensPerSecond"] = _clamp(_safe_val(_last_valid(llm_list, "tokens_per_second")))

        # Aggregated: tokens
        result["promptTokens"] = _sum_attr(llm_list, "prompt_tokens")
        result["completionTokens"] = _sum_attr(llm_list, "completion_tokens")
        result["totalTokens"] = _sum_attr(llm_list, "total_tokens")
        result["promptCachedTokens"] = _sum_attr(llm_list, "prompt_cached_tokens")
    else:
        result["llmProviderClass"] = ""
        result["llmModelName"] = ""
        result["llmInput"] = ""
        result["llmLatency"] = 0
        result["llmDuration"] = 0
        result["ttft"] = 0
        result["tokensPerSecond"] = 0
        result["promptTokens"] = 0
        result["completionTokens"] = 0
        result["totalTokens"] = 0
        result["promptCachedTokens"] = 0

    # Costs (not tracked in Python yet, set to 0)
    result["llmInputCost"] = 0
    result["llmOutputCost"] = 0
    result["llmCachedCost"] = 0

    # ── TTS metrics ──
    tts_list = turn.tts_metrics or []
    if tts_list:
        result["ttsProviderClass"] = tts_list[-1].provider_class or ""
        result["ttsModelName"] = tts_list[-1].model_name or ""

        # Latencies: last valid → convert to ms
        result["ttsLatency"] = _to_ms(_last_valid(tts_list, "tts_latency"))
        result["ttfb"] = _to_ms(_last_valid(tts_list, "ttfb"))
        result["ttsDuration"] = _to_ms(_last_valid(tts_list, "tts_duration"))

        # Aggregated: characters
        result["ttsCharacters"] = _sum_attr(tts_list, "tts_characters")
    else:
        result["ttsProviderClass"] = ""
        result["ttsModelName"] = ""
        result["ttsLatency"] = 0
        result["ttfb"] = 0
        result["ttsDuration"] = 0
        result["ttsCharacters"] = 0

    # ── Realtime metrics ──
    rt_list = turn.realtime_metrics or []
    if rt_list:
        result["realtimeProviderClass"] = rt_list[-1].provider_class or ""
        result["realtimeModelName"] = rt_list[-1].model_name or ""

        # Aggregated: tokens
        result["realtimeInputTokens"] = _sum_attr(rt_list, "realtime_input_tokens")
        result["realtimeOutputTokens"] = _sum_attr(rt_list, "realtime_output_tokens")
        result["realtimeTotalTokens"] = _sum_attr(rt_list, "realtime_total_tokens")
        result["realtimeInputTextTokens"] = _sum_attr(rt_list, "realtime_input_text_tokens")
        result["realtimeInputAudioTokens"] = _sum_attr(rt_list, "realtime_input_audio_tokens")
        result["realtimeInputImageTokens"] = _sum_attr(rt_list, "realtime_input_image_tokens")
        result["realtimeInputCachedTokens"] = _sum_attr(rt_list, "realtime_input_cached_tokens")
        result["realtimeThoughtsTokens"] = _sum_attr(rt_list, "realtime_thoughts_tokens")
        result["realtimeCachedTextTokens"] = _sum_attr(rt_list, "realtime_cached_text_tokens")
        result["realtimeCachedAudioTokens"] = _sum_attr(rt_list, "realtime_cached_audio_tokens")
        result["realtimeCachedImageTokens"] = _sum_attr(rt_list, "realtime_cached_image_tokens")
        result["realtimeOutputTextTokens"] = _sum_attr(rt_list, "realtime_output_text_tokens")
        result["realtimeOutputAudioTokens"] = _sum_attr(rt_list, "realtime_output_audio_tokens")
        result["realtimeOutputImageTokens"] = _sum_attr(rt_list, "realtime_output_image_tokens")
    else:
        result["realtimeProviderClass"] = ""
        result["realtimeModelName"] = ""
        result["realtimeInputTokens"] = 0
        result["realtimeOutputTokens"] = 0
        result["realtimeTotalTokens"] = 0
        result["realtimeInputTextTokens"] = 0
        result["realtimeInputAudioTokens"] = 0
        result["realtimeInputImageTokens"] = 0
        result["realtimeInputCachedTokens"] = 0
        result["realtimeThoughtsTokens"] = 0
        result["realtimeCachedTextTokens"] = 0
        result["realtimeCachedAudioTokens"] = 0
        result["realtimeCachedImageTokens"] = 0
        result["realtimeOutputTextTokens"] = 0
        result["realtimeOutputAudioTokens"] = 0
        result["realtimeOutputImageTokens"] = 0

    # Realtime costs (not tracked in Python yet)
    result["realtimeInputCost"] = 0
    result["realtimeOutputCost"] = 0
    result["realtimeCachedCost"] = 0

    # Per-turn costs (not tracked in Python yet)
    result["sttCost"] = 0
    result["llmCost"] = 0
    result["ttsCost"] = 0
    result["realtimeCost"] = 0
    result["totalCost"] = 0

    # ── Timeline ──
    timeline_events = turn.timeline_event_metrics or []
    result["timeline"] = [
        {
            "eventType": _map_timeline_event_type(evt.event_type),
            "startTime": _safe_val(evt.start_time),
            "endTime": evt.end_time,
            "durationInMs": evt.duration_ms,
            "text": evt.text or "",
        }
        for evt in timeline_events
    ]

    # Enrich user_speech timeline events with endTime/durationInMs from turn data
    for tl_evt in result["timeline"]:
        if tl_evt["eventType"] == "user_speech" and tl_evt["endTime"] is None:
            if turn.user_speech_end_time:
                tl_evt["endTime"] = _safe_val(turn.user_speech_end_time)
                if tl_evt["startTime"]:
                    tl_evt["durationInMs"] = round(
                        (turn.user_speech_end_time - (tl_evt["startTime"])) * 1000, 4
                    )

    # Final pass: clamp any remaining negative numeric values to 0
    for key, val in result.items():
        if isinstance(val, (int, float)) and val < 0:
            result[key] = 0

    return result


def _map_timeline_event_type(event_type: str) -> str:
    """Map internal timeline event types to API schema event types."""
    mapping = {
        "transcript_ready": "user_speech",
        "content_generated": "llm_response",
        # "agent_speech" is added directly from agent_turn_end hook
    }
    return mapping.get(event_type, event_type)


def transform_session(
    session: SessionMetrics,
    completed_turns: List[TurnMetrics],
    meeting_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transform SessionMetrics + completed turns into the agentSessionAnalyticsSchema format.

    Note: sessionId, userId, and meetingId are server-side concepts.
    The Python SDK sets meetingId from room_id, sessionId and userId
    are managed by the API server.
    """
    interactions = [transform_turn(turn) for turn in completed_turns]

    result: Dict[str, Any] = {
        "meetingId": meeting_id or session.room_id or "",
        "interactions": interactions,
    }

    return result
