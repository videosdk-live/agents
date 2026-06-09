import atexit
from typing import Any, Dict, Optional, List, NamedTuple
from opentelemetry.trace import Span, StatusCode
from opentelemetry import trace
from .integration import create_span, complete_span
from .metrics_schema import (
    TurnMetrics,
    SttMetrics, LlmMetrics, TtsMetrics,
    EouMetrics, VadMetrics,
    InterruptionMetrics, FallbackEvent, KbMetrics,
    RealtimeMetrics,
    SessionMetrics, ParticipantMetrics,
)
import asyncio
from dataclasses import asdict
import time
import logging

logger = logging.getLogger(__name__)


class TurnBounds(NamedTuple):
    """Computed time bounds for the three new turn-level spans."""
    user_start: Optional[float]
    user_end: Optional[float]
    agent_start: Optional[float]
    agent_end: Optional[float]
    parent_start: Optional[float]
    parent_end: Optional[float]


def _eou_end_with_wait(eou: Any) -> Optional[float]:
    """EOU end time extended by any post-EOU wait period."""
    if eou is None or eou.eou_end_time is None:
        return None
    wait_seconds = (eou.eou_wait_ms or 0) / 1000.0
    return eou.eou_end_time + wait_seconds


def _interrupt_end_time(turn: Any) -> Optional[float]:
    """Resolve when the agent's audio actually stopped on a true interrupt."""
    interrupt = getattr(turn, "interruption_metrics", None)
    if interrupt is None:
        return None
    if interrupt.interrupt_end_time is not None:
        return interrupt.interrupt_end_time
    if interrupt.interrupt_start_time is not None and interrupt.interrupt_duration is not None:
        return interrupt.interrupt_start_time + interrupt.interrupt_duration
    if interrupt.interrupt_start_time is not None and interrupt.interrupt_min_duration is not None:
        return interrupt.interrupt_start_time + interrupt.interrupt_min_duration
    return None


def _realtime_agent_start(turn: Any) -> Optional[float]:
    """In realtime mode, agent-side processing begins when user speech ends."""
    if not getattr(turn, "realtime_metrics", None):
        return None
    return turn.user_speech_end_time


def _compute_turn_time_bounds(turn: Any) -> TurnBounds:
    """Compute User Turn / Agent Turn / parent Turn time bounds from a TurnMetrics instance.

    See spec §2 for the rule set. None values are excluded from min/max; if a
    side has no usable timestamps, it falls back to perf_counter() so the span
    still renders as a zero-width diagnostic marker.
    """
    def collect(values):
        return [v for v in values if v is not None]

    # --- User side ---
    user_starts = collect([
        turn.user_speech_start_time,
        *(v.user_speech_start_time for v in turn.vad_metrics),
        *(s.stt_start_time for s in turn.stt_metrics),
        *(e.eou_start_time for e in turn.eou_metrics),
        *(k.kb_start_time for k in turn.kb_metrics),
    ])
    user_ends = collect([
        turn.user_speech_end_time,
        *(v.user_speech_end_time for v in turn.vad_metrics),
        *(s.stt_end_time for s in turn.stt_metrics),
        *(_eou_end_with_wait(e) for e in turn.eou_metrics),
        *(k.kb_end_time for k in turn.kb_metrics),
    ])
    for ev in turn.timeline_event_metrics:
        if ev.event_type == "user_speech":
            if ev.start_time:
                user_starts.append(ev.start_time)
                # Also use start_time as a floor for user_ends so the User Turn
                # bounds always cover the timeline event. Without this, when a
                # timeline event arrives with start_time > EOU end and no
                # end_time set (interrupt-buffer artifacts), the User Input
                # Speech span renders OUTSIDE its User Turn parent.
                user_ends.append(ev.start_time)
            if ev.end_time:
                user_ends.append(ev.end_time)

    # --- Agent side ---
    agent_starts = collect([
        *(l.llm_start_time for l in turn.llm_metrics),
        *(t.tts_start_time for t in turn.tts_metrics),
        turn.agent_speech_start_time,
        _realtime_agent_start(turn),
    ])
    agent_ends = collect([
        *(t.tts_end_time for t in turn.tts_metrics),
        *(l.llm_end_time for l in turn.llm_metrics),
        turn.agent_speech_end_time,
        _interrupt_end_time(turn),
    ])
    for ev in turn.timeline_event_metrics:
        if ev.event_type == "agent_speech":
            if ev.start_time:
                agent_starts.append(ev.start_time)
                # See user-side comment above — symmetric guard.
                agent_ends.append(ev.start_time)
            if ev.end_time:
                agent_ends.append(ev.end_time)

    user_start = min(user_starts) if user_starts else None
    user_end = max(user_ends) if user_ends else None
    agent_start = min(agent_starts) if agent_starts else None
    agent_end = max(agent_ends) if agent_ends else None

    now = time.perf_counter()
    if user_start is None and user_end is None:
        user_start = user_end = now
    elif user_start is None:
        user_start = user_end
    elif user_end is None:
        user_end = user_start

    if agent_start is None and agent_end is None:
        agent_start = agent_end = now
    elif agent_start is None:
        agent_start = agent_end
    elif agent_end is None:
        agent_end = agent_start

    parent_start = min(user_start, agent_start)
    parent_end = max(user_end, agent_end)

    return TurnBounds(
        user_start=user_start,
        user_end=user_end,
        agent_start=agent_start,
        agent_end=agent_end,
        parent_start=parent_start,
        parent_end=parent_end,
    )


_PIPELINE_MODE_CANONICAL = {
    "realtime": "realtime",
    "hybrid": "hybrid",
    "full_cascading": "cascade",
    "partial_cascading": "cascade",
    "llm_tts_only": "cascade",
    "stt_llm_only": "cascade",
    "stt_tts_only": "cascade",
    "llm_only": "cascade",
    "stt_only": "cascade",
    "tts_only": "cascade",
}


def _canonical_pipeline_mode(raw: Optional[str]) -> str:
    """Reduce PipelineMode enum values to the 3-mode contract used on the wire.

    Spec §3 commits to {"cascade", "realtime", "hybrid"} as the value of
    `pipeline_mode` mirrored on User Turn / Agent Turn spans. The internal
    PipelineMode enum has finer-grained variants (full_cascading,
    partial_cascading, etc.) that we collapse to one of those three here.
    """
    if not raw:
        return "cascade"
    return _PIPELINE_MODE_CANONICAL.get(raw, "cascade")


def _turn_level_attrs(turn_index: int, turn: Any, turn_side: str) -> Dict[str, Any]:
    """Attributes mirrored onto BOTH User Turn and Agent Turn for a given turn.

    With the parent Turn #N span removed, the renderer pairs the two halves
    via the `turn_index` attribute. `is_interrupted` and `pipeline_mode` mirror
    onto each half so consumers can filter/badge without joining sides.
    """
    pm_raw = getattr(turn.session_metrics, "pipeline_mode", None) if turn.session_metrics else None
    return {
        "turn_side": turn_side,
        "turn_index": turn_index,
        "turn_id": turn.turn_id or "",
        "is_interrupted": turn.is_interrupted,
        "pipeline_mode": _canonical_pipeline_mode(pm_raw),
    }


def _provider_attrs(turn: Any, component: str) -> Dict[str, str]:
    """Extract provider_class / model_name from session_metrics for a component."""
    sm = getattr(turn, "session_metrics", None)
    if sm is None or not sm.provider_per_component:
        return {}
    entry = sm.provider_per_component.get(component) or {}
    out: Dict[str, str] = {}
    if entry.get("provider_class"):
        out["provider_class"] = entry["provider_class"]
    if entry.get("model_name"):
        out["model_name"] = entry["model_name"]
    return out


def _create_vad_span(turn: Any, vad: Any, parent: Span) -> None:
    pdata = _provider_attrs(turn, "vad")
    provider_class = pdata.get("provider_class")
    if not provider_class:
        return  # no provider info → skip; matches current behavior
    attrs = dict(pdata)
    if vad.vad_min_silence_duration:
        attrs["min_silence_duration"] = vad.vad_min_silence_duration
    if vad.vad_min_speech_duration:
        attrs["min_speech_duration"] = vad.vad_min_speech_duration
    if vad.vad_threshold:
        attrs["threshold"] = vad.vad_threshold

    if vad.user_speech_start_time is None and vad.user_speech_end_time is not None:
        vad.user_speech_start_time = vad.user_speech_end_time - (vad.vad_min_silence_duration or 0)
    elif vad.user_speech_start_time is not None and vad.user_speech_end_time is None:
        vad.user_speech_end_time = vad.user_speech_start_time + (vad.vad_min_silence_duration or 0)

    span = create_span(
        f"{provider_class}: VAD Processing",
        attrs,
        parent_span=parent,
        start_time=vad.user_speech_start_time,
    )
    complete_span(span, StatusCode.OK, end_time=vad.user_speech_end_time)


def _create_stt_span(turn: Any, stt: Any, parent: Span) -> None:
    pdata = _provider_attrs(turn, "stt")
    provider_class = pdata.get("provider_class", "STT")
    attrs = dict(pdata)
    attrs["input"] = "N/A"
    if stt.stt_latency is not None:
        attrs["duration_ms"] = stt.stt_latency
    if stt.stt_start_time:
        attrs["start_timestamp"] = stt.stt_start_time
    if stt.stt_end_time:
        attrs["end_timestamp"] = stt.stt_end_time
    if stt.stt_transcript:
        attrs["output"] = stt.stt_transcript
    if provider_class == "DeepgramSTTV2" and turn.preemtive_generation_enabled:
        attrs["stt_preemptive_generation_enabled"] = turn.preemtive_generation_enabled

    span = create_span(
        f"{provider_class}: Speech to Text Processing",
        attrs,
        parent_span=parent,
        start_time=stt.stt_start_time,
    )

    if span and stt.stt_preemptive_generation_enabled:
        preflight_attrs = {
            "preemptive_generation_occurred": stt.stt_preemptive_generation_occurred,
            "partial_text": stt.stt_preflight_transcript,
            "final_text": stt.stt_transcript,
        }
        if stt.stt_preemptive_generation_occurred:
            preflight_attrs["preemptive_generation_latency"] = stt.stt_preflight_latency
        preflight_span = create_span(
            "Preemptive Generation",
            preflight_attrs,
            parent_span=span,
            start_time=stt.stt_start_time,
        )
        complete_span(preflight_span, StatusCode.OK, end_time=stt.stt_preflight_end_time or stt.stt_end_time)

    complete_span(span, StatusCode.OK, end_time=stt.stt_end_time)


def _create_eou_span(turn: Any, eou: Any, parent: Span) -> None:
    pdata = _provider_attrs(turn, "eou")
    provider_class = pdata.get("provider_class", "EOU")
    attrs = dict(pdata)
    if turn.user_speech:
        attrs["input"] = turn.user_speech
    if eou.eou_latency is not None:
        attrs["duration_ms"] = eou.eou_latency
    if eou.eou_start_time:
        attrs["start_timestamp"] = eou.eou_start_time
    if eou.eou_end_time:
        attrs["end_timestamp"] = eou.eou_end_time
    if eou.waited_for_additional_speech:
        attrs["waited_for_additional_speech"] = eou.waited_for_additional_speech
    if eou.eou_probability:
        attrs["eou_probability"] = round(eou.eou_probability, 4)
    eou_cfg = (turn.session_metrics.eou_config or {}) if turn.session_metrics else {}
    if eou_cfg.get("min_speech_wait_timeout"):
        attrs["min_speech_wait_timeout"] = eou_cfg["min_speech_wait_timeout"]
    if eou_cfg.get("max_speech_wait_timeout"):
        attrs["max_speech_wait_timeout"] = eou_cfg["max_speech_wait_timeout"]

    span = create_span(
        f"{provider_class}: End-Of-Utterance Detection",
        attrs,
        parent_span=parent,
        start_time=eou.eou_start_time,
    )

    if span and eou.waited_for_additional_speech and eou.eou_wait_ms is not None and eou.eou_end_time is not None:
        wait_ms = round(eou.eou_wait_ms, 4)
        wait_span = create_span(
            "EOU Wait",
            {"eou_wait_ms": wait_ms, "eou_probability": round(eou.eou_probability or 0, 4)},
            parent_span=span,
            start_time=eou.eou_end_time,
        )
        complete_span(wait_span, StatusCode.OK, end_time=eou.eou_end_time + wait_ms / 1000.0)

    complete_span(span, StatusCode.OK, end_time=eou.eou_end_time)


def _create_kb_span(turn: Any, kb: Any, parent: Span) -> None:
    attrs: Dict[str, Any] = {}
    if turn.user_speech:
        attrs["input"] = turn.user_speech
    if kb.kb_retrieval_latency:
        attrs["retrieval_latency_ms"] = kb.kb_retrieval_latency
    if kb.kb_start_time:
        attrs["start_timestamp"] = kb.kb_start_time
    if kb.kb_end_time:
        attrs["end_timestamp"] = kb.kb_end_time
    if kb.kb_documents:
        if len(kb.kb_documents) <= 5:
            attrs["documents"] = ", ".join(kb.kb_documents)
        else:
            attrs["documents"] = f"{len(kb.kb_documents)} documents"
        attrs["document_count"] = len(kb.kb_documents)
    if kb.kb_scores:
        attrs["scores"] = ", ".join(str(round(s, 4)) for s in kb.kb_scores[:5])

    span = create_span(
        "Knowledge Base: Retrieval",
        attrs,
        parent_span=parent,
        start_time=kb.kb_start_time,
    )
    complete_span(span, StatusCode.OK, end_time=kb.kb_end_time)


def _create_user_input_speech_span(ev: Any, parent: Span, fallback_end: Optional[float]) -> None:
    attrs = {"Transcript": ev.text, "duration_ms": ev.duration_ms}
    span = create_span(
        "User Input Speech",
        attrs,
        parent_span=parent,
        start_time=ev.start_time,
    )
    end = ev.end_time if ev.end_time else fallback_end
    # Defensive: perf_counter skew between the user_speech timeline event and
    # EOU end can put fallback_end ~1 ms *before* ev.start_time, which would
    # produce a negative-duration span. Clamp to start so the span is at least
    # zero-width rather than malformed.
    if ev.start_time is not None and end is not None and end < ev.start_time:
        end = ev.start_time
    complete_span(span, StatusCode.OK, end_time=end)


def _create_llm_span(
    turn: Any,
    llm: Any,
    parent: Span,
    round_index: int = 1,
    total_rounds: int = 1,
) -> Optional[Span]:
    pdata = _provider_attrs(turn, "llm")
    provider_class = pdata.get("provider_class", "LLM")
    attrs = dict(pdata)
    if llm.llm_input:
        attrs["input"] = llm.llm_input
    if llm.llm_duration:
        attrs["duration_ms"] = llm.llm_duration
    if llm.llm_start_time:
        attrs["start_timestamp"] = llm.llm_start_time
    if llm.llm_end_time:
        attrs["end_timestamp"] = llm.llm_end_time
    # Per-round output: tool-call summary for tool-producing rounds, the spoken
    # answer for the final round. Avoids the old single-entry concatenation.
    if getattr(llm, "produced_tool_calls", None):
        attrs["output"] = "→ tool call: " + ", ".join(llm.produced_tool_calls)
    elif turn.agent_speech:
        attrs["output"] = turn.agent_speech
    if llm.prompt_tokens:
        attrs["input_tokens"] = llm.prompt_tokens
    if llm.completion_tokens:
        attrs["output_tokens"] = llm.completion_tokens
    if llm.prompt_cached_tokens:
        attrs["cached_input_tokens"] = llm.prompt_cached_tokens
    if llm.total_tokens:
        attrs["total_tokens"] = llm.total_tokens
    if total_rounds > 1:
        attrs["llm_round"] = round_index

    name = f"{provider_class}: LLM Processing"
    if total_rounds > 1:
        name = f"{name} ({round_index})"

    span = create_span(name, attrs, parent_span=parent, start_time=llm.llm_start_time)

    if span and llm.llm_ttft is not None and llm.llm_start_time is not None:
        ttft_span = create_span(
            "Time to First Token",
            {"llm_ttft": llm.llm_ttft},
            parent_span=span,
            start_time=llm.llm_start_time,
        )
        complete_span(ttft_span, StatusCode.OK, end_time=llm.llm_start_time + (llm.llm_ttft / 1000.0))

    complete_span(span, StatusCode.OK, end_time=llm.llm_end_time)
    return span


def _is_llm_round_husk(llm: Any) -> bool:
    """A follow-up LLM round that was started but produced nothing and never
    completed — e.g. the post-tool round after a terminal tool (end_call) tore the
    session down before any token arrived. Rendering it adds a content-less span
    whose duration is a meaningless render-time fallback. Drop it.

    A round that did ANY work is never a husk: it completed (has end_time), streamed
    a token (has ttft), reported usage (has tokens), or requested a tool.
    """
    return (
        llm.llm_end_time is None
        and llm.llm_ttft is None
        and not getattr(llm, "produced_tool_calls", None)
        and not (llm.prompt_tokens or llm.completion_tokens or llm.total_tokens)
    )


def _create_invoked_tool_span(tool: Any, parent: Span) -> None:
    """A function-tool execution as a sibling under Agent Turn, with real
    duration plus the call's args (tool_params) and result (tool_response)."""
    attrs: Dict[str, Any] = {}
    if tool.tool_params:
        attrs["args"] = tool.tool_params
    if tool.tool_response:
        attrs["output"] = tool.tool_response
    if tool.latency is not None:
        attrs["duration_ms"] = tool.latency
    start = tool.start_time
    span = create_span(
        f"Invoked Tool: {tool.tool_name or 'unknown'}",
        attrs,
        parent_span=parent,
        start_time=start,
    )
    end = tool.end_time if tool.end_time is not None else start
    if start is not None and end is not None and end < start:
        end = start  # defensive clamp — never emit negative duration
    complete_span(span, StatusCode.OK, end_time=end)


def _create_tts_span(turn: Any, tts: Any, parent: Span) -> Optional[Span]:
    pdata = _provider_attrs(turn, "tts")
    provider_class = pdata.get("provider_class", "TTS")
    attrs = dict(pdata)
    if turn.agent_speech:
        attrs["input"] = turn.agent_speech
    if tts.tts_duration:
        attrs["duration_ms"] = tts.tts_duration
    if tts.tts_start_time:
        attrs["start_timestamp"] = tts.tts_start_time
    if tts.tts_end_time:
        attrs["end_timestamp"] = tts.tts_end_time
    if tts.tts_characters:
        attrs["characters"] = tts.tts_characters
    if turn.agent_speech_duration:
        attrs["audio_duration_ms"] = turn.agent_speech_duration
    attrs["output"] = "N/A"

    span = create_span(
        f"{provider_class}: Text to Speech Processing",
        attrs,
        parent_span=parent,
        start_time=tts.tts_start_time,
    )

    if span and tts.tts_first_byte_time is not None:
        ttfb_span = create_span(
            "Time to First Byte",
            parent_span=span,
            start_time=tts.tts_start_time,
        )
        complete_span(ttfb_span, StatusCode.OK, end_time=tts.tts_first_byte_time)

    complete_span(span, StatusCode.OK, end_time=tts.tts_end_time)
    return span


def _create_rt_span(turn: Any, rt: Any, parent: Span) -> Optional[Span]:
    pdata = _provider_attrs(turn, "realtime")
    provider_class = pdata.get("provider_class", "Realtime")
    attrs = dict(pdata)

    rt_start = turn.user_speech_end_time if turn.user_speech_end_time else turn.agent_speech_start_time
    rt_end = turn.agent_speech_start_time

    span = create_span(
        f"{provider_class}: Realtime Processing",
        attrs,
        parent_span=parent,
        start_time=rt_start,
    )

    if span:
        if turn.function_tool_metrics:
            for tool in turn.function_tool_metrics:
                _create_invoked_tool_span(tool, span)
        else:
            # Fallback for older data with names but no FunctionToolMetrics.
            for tool_name in turn.function_tools_called or []:
                now = time.perf_counter()
                tool_span = create_span(f"Invoked Tool: {tool_name}", parent_span=span, start_time=now)
                complete_span(tool_span, StatusCode.OK, end_time=now)

        if turn.e2e_latency is not None and rt_start is not None and rt_end is not None:
            ttfw_span = create_span(
                "Time to First Word",
                {"duration_ms": turn.e2e_latency},
                parent_span=span,
                start_time=rt_start,
            )
            complete_span(ttfw_span, StatusCode.OK, end_time=rt_end)

    complete_span(span, StatusCode.OK, end_time=rt_end)
    return span


def _create_agent_output_speech_span(ev: Any, parent: Span, fallback_end: Optional[float]) -> Optional[Span]:
    attrs = {"Transcript": ev.text, "duration_ms": ev.duration_ms}
    span = create_span(
        "Agent Output Speech",
        attrs,
        parent_span=parent,
        start_time=ev.start_time,
    )
    # NOTE: span is intentionally NOT completed here. Task 7's _create_user_interjection_span
    # writes its children into this span before we end it. The caller of this helper is
    # responsible for ending it via end_span() at the agent_turn populator boundary.
    return span


def _create_thinking_audio_span(turn: Any, ev: Any, parent: Span, fallback_end: Optional[float]) -> None:
    attrs: Dict[str, Any] = {}
    if turn.thinking_audio_file_path:
        attrs["file_path"] = turn.thinking_audio_file_path
    if turn.thinking_audio_looping is not None:
        attrs["looping"] = turn.thinking_audio_looping
    if turn.thinking_audio_override_thinking is not None:
        attrs["override_thinking"] = turn.thinking_audio_override_thinking
    if ev.duration_ms is not None:
        attrs["duration_ms"] = ev.duration_ms

    span = create_span(
        "Thinking Audio",
        attrs,
        parent_span=parent,
        start_time=ev.start_time,
    )
    end = ev.end_time if ev.end_time else fallback_end
    if ev.start_time is not None and end is not None and end < ev.start_time:
        end = ev.start_time
    complete_span(span, StatusCode.OK, end_time=end)


def _create_user_interjection_span(interruption: Any, agent_speech_span: Optional[Span], agent_turn_span: Span) -> None:
    """User Interjection (Resumed | Escalated) — replaces the old False Interruption span.

    Parents to the Agent Output Speech span when present (so the overlap is visible).
    Falls back to Agent Turn directly if agent speech timeline event was lost.
    """
    if interruption.false_interrupt_start_time is None:
        return

    parent = agent_speech_span if agent_speech_span is not None else agent_turn_span

    attrs: Dict[str, Any] = {}
    if interruption.interrupt_mode:
        attrs["interrupt_mode"] = interruption.interrupt_mode
    if interruption.false_interrupt_pause_duration:
        attrs["pause_duration_config"] = interruption.false_interrupt_pause_duration
    if interruption.false_interrupt_duration:
        attrs["false_interrupt_duration"] = interruption.false_interrupt_duration
    if interruption.resumed_after_false_interrupt:
        attrs["resumed_after_false_interrupt"] = True
        attrs["actual_duration"] = interruption.false_interrupt_duration

    end = interruption.false_interrupt_end_time
    if end is None:
        end = interruption.interrupt_start_time  # escalated path

    name = (
        "User Interjection (Resumed)"
        if interruption.resumed_after_false_interrupt
        else "User Interjection (Escalated)"
    )
    span = create_span(name, attrs, parent_span=parent, start_time=interruption.false_interrupt_start_time)
    complete_span(span, StatusCode.OK, message="User interjection detected", end_time=end)


_USER_SIDE_COMPONENTS = {"stt", "eou", "vad", "kb", "turn_detector"}
_AGENT_SIDE_COMPONENTS = {"llm", "tts", "realtime"}


def _is_user_side_component(component_type: Optional[str]) -> bool:
    if not component_type:
        return False
    return component_type.lower() in _USER_SIDE_COMPONENTS


def _is_agent_side_component(component_type: Optional[str]) -> bool:
    if not component_type:
        return False
    return component_type.lower() in _AGENT_SIDE_COMPONENTS


def _create_fallback_span(fallback: Any, parent: Span) -> None:
    if fallback.is_recovery:
        name = f"Recovery: {fallback.component_type}"
        attrs = {
            "temporary_disable_sec": fallback.temporary_disable_sec,
            "permanent_disable_after_attempts": fallback.permanent_disable_after_attempts,
            "recovery_attempt": fallback.recovery_attempt,
            "message": fallback.message,
            "restored_provider": fallback.new_provider_label,
            "previous_provider": fallback.original_provider_label,
        }
        span = create_span(name, attrs, parent_span=parent, start_time=fallback.start_time)
        complete_span(span, StatusCode.OK, message="Recovery completed", end_time=fallback.end_time)
        return

    name = f"Fallback: {fallback.component_type}"
    attrs = {
        "temporary_disable_sec": fallback.temporary_disable_sec,
        "permanent_disable_after_attempts": fallback.permanent_disable_after_attempts,
        "recovery_attempt": fallback.recovery_attempt,
        "message": fallback.message,
    }
    span = create_span(name, attrs, parent_span=parent, start_time=fallback.start_time)
    if not span:
        return

    if fallback.original_provider_label:
        orig_span = create_span(
            f"Connection: {fallback.original_provider_label}",
            {"provider": fallback.original_provider_label, "status": "failed"},
            parent_span=span,
            start_time=fallback.start_time,
        )
        complete_span(orig_span, StatusCode.ERROR, end_time=fallback.start_time)

    if fallback.new_provider_label:
        new_span = create_span(
            f"Connection: {fallback.new_provider_label}",
            {"provider": fallback.new_provider_label, "status": "success"},
            parent_span=span,
            start_time=fallback.start_time,
        )
        complete_span(new_span, StatusCode.OK, end_time=fallback.start_time)

    status = StatusCode.OK if fallback.new_provider_label else StatusCode.ERROR
    complete_span(span, status, end_time=fallback.start_time)


_USER_SIDE_ERROR_SOURCES = {"STT", "TURN-D", "VAD", "KB"}
_AGENT_SIDE_ERROR_SOURCES = {"LLM", "TTS", "REALTIME", "REALTIME_MODEL"}


def _route_error_to_side(error: Dict[str, Any]) -> str:
    """Return 'user', 'agent', or 'agent' (default) for unknown sources."""
    src = (error.get("source") or "").upper()
    if src in _USER_SIDE_ERROR_SOURCES:
        return "user"
    if src in _AGENT_SIDE_ERROR_SOURCES:
        return "agent"
    logger.debug(f"Unrecognized error source {src!r}; routing to agent side by default")
    return "agent"


def _create_error_catchall_span(error: Dict[str, Any], parent: Span) -> None:
    src = error.get("source", "Unknown")
    attrs = {}
    if error.get("message"):
        attrs["error message"] = error["message"]
    start = error.get("timestamp_perf")
    span = create_span(f"{src} Error span", attrs, parent_span=parent, start_time=start)
    complete_span(span, StatusCode.ERROR, end_time=(start + 0.001) if start is not None else None)


def _create_turn_interrupted_span(interruption: Any, agent_turn_span: Span) -> None:
    if interruption.interrupt_start_time is None:
        return

    attrs: Dict[str, Any] = {}
    if interruption.interrupt_mode:
        attrs["interrupt_mode"] = interruption.interrupt_mode
    if interruption.interrupt_min_duration is not None:
        attrs["interrupt_min_duration"] = interruption.interrupt_min_duration
    if interruption.interrupt_min_words is not None:
        attrs["interrupt_min_words"] = interruption.interrupt_min_words
    if interruption.false_interrupt_pause_duration is not None:
        attrs["false_interrupt_pause_duration"] = interruption.false_interrupt_pause_duration
    if interruption.resume_on_false_interrupt is not None:
        attrs["resume_on_false_interrupt"] = interruption.resume_on_false_interrupt
    if interruption.interrupt_reason:
        attrs["interrupt_reason"] = interruption.interrupt_reason
    if interruption.interrupt_words is not None:
        attrs["interrupt_words"] = interruption.interrupt_words
    if interruption.interrupt_duration is not None:
        attrs["interrupt_duration"] = interruption.interrupt_duration
    if interruption.false_interrupt_start_time is not None:
        attrs["preceded_by_false_interrupt"] = True

    end = interruption.interrupt_end_time
    if end is None:
        if interruption.interrupt_duration is not None:
            end = interruption.interrupt_start_time + interruption.interrupt_duration
        elif interruption.interrupt_min_duration is not None:
            end = interruption.interrupt_start_time + interruption.interrupt_min_duration
        else:
            end = interruption.interrupt_start_time

    span = create_span("Turn Interrupted", attrs, parent_span=agent_turn_span, start_time=interruption.interrupt_start_time)
    complete_span(span, StatusCode.OK, message="Agent was interrupted", end_time=end)


class TracesFlowManager:
    """
    Manages the flow of OpenTelemetry traces for agent Turns,
    ensuring correct parent-child relationships between spans.
    """

    def __init__(self, room_id: str, session_id: Optional[str] = None):
        self.room_id = room_id
        self.session_id = session_id
        self.root_span: Optional[Span] = None
        self.agent_session_span: Optional[Span] = None
        self.main_turn_span: Optional[Span] = None
        self.agent_session_config_span: Optional[Span] = None
        self.agent_session_closed_span: Optional[Span] = None
        self._turn_count = 0
        self.root_span_ready = asyncio.Event()
        self.a2a_span: Optional[Span] = None
        self._a2a_turn_count = 0
        self.session_metrics: Optional[SessionMetrics] = None
        self.participant_metrics: Optional[List[ParticipantMetrics]] = []
        self._atexit_done = False
        # If the agent process dies before Room.disconnect runs (KeyboardInterrupt,
        # crash, SIGTERM), the long-lived root + main_turn spans would never .end()
        # — the OTLP BatchSpanProcessor would drop them, producing orphan traces
        # at the collector. This hook ensures they're ended before the telemetry
        # atexit force-flushes the BSP.
        atexit.register(self._atexit_end_meeting)

    def set_session_metrics(self, session_metrics: SessionMetrics):
        """Set the session metrics for the trace manager."""
        self.session_metrics = session_metrics

    def set_session_id(self, session_id: str):
        """Set the session ID for the trace manager."""
        self.session_id = session_id

    def start_agent_joined_meeting(self, attributes: Dict[str, Any]):
        """Starts the root span for the agent joining a meeting."""
        if self.root_span:
            return
        
        agent_name = attributes.get('agent_name', 'UnknownAgent')
        agent_id = attributes.get('peerId', 'UnknownID')

        span_name = f"Agent Session: agentName_{agent_name}_agentId_{agent_id}"

        start_time = attributes.get('start_time', time.perf_counter())
        self.root_span = create_span(span_name, attributes, start_time=start_time)

        # Always set root_span_ready so downstream awaits don't hang
        # when telemetry is not initialized (create_span returns None)
        self.root_span_ready.set()

    async def start_agent_session_config(self, attributes: Dict[str, Any]):
        """Starts the span for the agent's session configuration, child of the root span."""
        await self.root_span_ready.wait()
        if not self.root_span:
            return

        if self.agent_session_config_span:
            return

        start_time = attributes.get('start_time', time.perf_counter())
        self.agent_session_config_span = create_span("Session Configuration", attributes, parent_span=self.root_span, start_time=start_time)
        if self.agent_session_config_span:
            with trace.use_span(self.agent_session_config_span):
                self.end_agent_session_config()

    def end_agent_session_config(self):
        """Completes the agent session config span."""
        end_time = time.perf_counter()
        self.end_span(self.agent_session_config_span, "Agent session config ended", end_time=end_time)
        self.agent_session_config_span = None

    async def start_agent_session_closed(self, attributes: Dict[str, Any]):
        """Starts the span for agent session closed."""
        await self.root_span_ready.wait()
        if not self.root_span:
            return

        if self.agent_session_closed_span:
            return

        start_time = attributes.get('start_time', time.perf_counter())
        self.agent_session_closed_span = create_span("Agent Session Closed", attributes, parent_span=self.root_span, start_time=start_time)

    def end_agent_session_closed(self):
        """Completes the agent session closed span."""
        end_time = time.perf_counter()
        self.end_span(self.agent_session_closed_span, "Agent session closed", end_time=end_time)
        self.agent_session_closed_span = None

    async def start_agent_session(self, attributes: Dict[str, Any]):
        """Starts the span for the agent's session, child of the root span."""
        await self.root_span_ready.wait()
        if not self.root_span:
            return

        if self.agent_session_span:
            return

        start_time = attributes.get('start_time', time.perf_counter())
        p_m =[]
        a_p_m = []
        for p in self.participant_metrics:
            if p.kind == "user":
                p_m.append(asdict(p))
            else:
                a_p_m.append(asdict(p))
        attributes["participant_metrics"] = p_m
        attributes["agent_participant_metrics"] = a_p_m
        self.agent_session_span = create_span("Session Started", attributes, parent_span=self.root_span, start_time=start_time)
        
        self.start_main_turn()

    def start_main_turn(self):
        """Starts a parent span for all user-agent turn."""
        if not self.agent_session_span:
            return

        if self.main_turn_span:
            return
            
        start_time = time.perf_counter()
        self.main_turn_span = create_span("User & Agent Turns", parent_span=self.agent_session_span, start_time=start_time)
    
    def create_unified_turn_trace(self, turn: TurnMetrics, session: Any = None) -> None:
        """Creates the per-turn span tree with User Turn / Agent Turn split.

        See docs/superpowers/specs/2026-05-28-traces-user-agent-split-design.md
        for the full structure. The legacy flat shape is gone, and the
        intermediate Turn #N grouping span has also been dropped — User Turn
        and Agent Turn now hang directly off `User & Agent Turns`. The renderer
        pairs the two halves via the `turn_index` attribute.
        """
        if not self.main_turn_span:
            return

        self._turn_count += 1
        bounds = _compute_turn_time_bounds(turn)

        user_turn_span = self._populate_user_turn(turn, self.main_turn_span, bounds)
        agent_turn_span = self._populate_agent_turn(turn, self.main_turn_span, bounds)

        self.end_span(user_turn_span, end_time=bounds.user_end)
        self.end_span(agent_turn_span, end_time=bounds.agent_end)

    def _populate_user_turn(self, turn: TurnMetrics, parent: Span, bounds: TurnBounds) -> Optional[Span]:
        attrs = _turn_level_attrs(self._turn_count, turn, "user")
        user_turn_span = create_span(
            f"User Turn #{self._turn_count}",
            attrs,
            parent_span=parent,
            start_time=bounds.user_start,
        )
        if not user_turn_span:
            return None

        for vad in turn.vad_metrics or []:
            try:
                _create_vad_span(turn, vad, user_turn_span)
            except Exception as e:
                logger.error(f"Error creating VAD span: {e}")

        for stt in turn.stt_metrics or []:
            try:
                _create_stt_span(turn, stt, user_turn_span)
            except Exception as e:
                logger.error(f"Error creating STT span: {e}")

        for eou in turn.eou_metrics or []:
            try:
                _create_eou_span(turn, eou, user_turn_span)
            except Exception as e:
                logger.error(f"Error creating EOU span: {e}")

        for kb in turn.kb_metrics or []:
            try:
                _create_kb_span(turn, kb, user_turn_span)
            except Exception as e:
                logger.error(f"Error creating KB span: {e}")

        for ev in turn.timeline_event_metrics or []:
            if ev.event_type == "user_speech":
                _create_user_input_speech_span(ev, user_turn_span, bounds.user_end)

        for event in turn.fallback_events or []:
            if not _is_user_side_component(event.component_type):
                continue
            try:
                _create_fallback_span(event, user_turn_span)
            except Exception as e:
                logger.error(f"Error creating Fallback span on user side: {e}")

        for error in turn.errors or []:
            if _route_error_to_side(error) != "user":
                continue
            try:
                _create_error_catchall_span(error, user_turn_span)
            except Exception as e:
                logger.error(f"Error creating user-side error span: {e}")

        return user_turn_span

    def _populate_agent_turn(self, turn: TurnMetrics, parent: Span, bounds: TurnBounds) -> Optional[Span]:
        attrs = _turn_level_attrs(self._turn_count, turn, "agent")
        agent_turn_span = create_span(
            f"Agent Turn #{self._turn_count}",
            attrs,
            parent_span=parent,
            start_time=bounds.agent_start,
        )
        if not agent_turn_span:
            return None

        # LLM rounds + tool executions render as time-ordered siblings under
        # Agent Turn, so the request → tool → follow-up sequence (and its
        # latency) is visible. See spec §2/§3.
        # In realtime mode, tools are owned by _create_rt_span (nested under
        # Realtime Processing). Skip them here to avoid double-emission.
        is_realtime = bool(turn.realtime_metrics)
        # Drop content-less husk rounds (started but cut off by a terminal tool like
        # end_call: never completed, no token, no usage, no tool call). Their bar
        # length is a meaningless render-time artifact. A round that did ANY work is kept.
        llm_rounds = [llm for llm in (turn.llm_metrics or []) if not _is_llm_round_husk(llm)]
        total_rounds = len(llm_rounds)
        agent_primary: list = []
        for idx, llm in enumerate(llm_rounds, start=1):
            agent_primary.append((llm.llm_start_time, idx, "llm", llm))
        if not is_realtime:
            for tool in turn.function_tool_metrics or []:
                agent_primary.append((tool.start_time, 0, "tool", tool))
        agent_primary.sort(key=lambda x: (x[0] if x[0] is not None else 0))
        for _start, idx, kind, payload in agent_primary:
            try:
                if kind == "llm":
                    _create_llm_span(turn, payload, agent_turn_span, round_index=idx, total_rounds=total_rounds)
                else:
                    _create_invoked_tool_span(payload, agent_turn_span)
            except Exception as e:
                logger.error(f"Error creating {kind} span: {e}")

        for tts in turn.tts_metrics or []:
            try:
                _create_tts_span(turn, tts, agent_turn_span)
            except Exception as e:
                logger.error(f"Error creating TTS span: {e}")

        for rt in turn.realtime_metrics or []:
            try:
                _create_rt_span(turn, rt, agent_turn_span)
            except Exception as e:
                logger.error(f"Error creating RT span: {e}")

        agent_speech_span: Optional[Span] = None
        for ev in turn.timeline_event_metrics or []:
            if ev.event_type == "agent_speech":
                agent_speech_span = _create_agent_output_speech_span(ev, agent_turn_span, bounds.agent_end)
            elif ev.event_type == "thinking_audio":
                _create_thinking_audio_span(turn, ev, agent_turn_span, bounds.agent_end)

        if turn.interruption_metrics and turn.interruption_metrics.false_interrupt_start_time is not None:
            try:
                _create_user_interjection_span(turn.interruption_metrics, agent_speech_span, agent_turn_span)
            except Exception as e:
                logger.error(f"Error creating User Interjection span: {e}")

        if turn.is_interrupted and turn.interruption_metrics is not None:
            try:
                _create_turn_interrupted_span(turn.interruption_metrics, agent_turn_span)
            except Exception as e:
                logger.error(f"Error creating Turn Interrupted span: {e}")

        if agent_speech_span:
            # Close the agent speech span at its own end_time (or fall back to agent_end).
            agent_ev = next(
                (e for e in turn.timeline_event_metrics or [] if e.event_type == "agent_speech"),
                None,
            )
            agent_speech_end = (agent_ev.end_time if agent_ev and agent_ev.end_time else bounds.agent_end)
            # Defensive: clamp end >= start so we never emit a negative-duration span
            # (same perf_counter skew issue as User Input Speech).
            if agent_ev and agent_ev.start_time is not None and agent_speech_end is not None and agent_speech_end < agent_ev.start_time:
                agent_speech_end = agent_ev.start_time
            self.end_span(agent_speech_span, end_time=agent_speech_end)

        for event in turn.fallback_events or []:
            if not _is_agent_side_component(event.component_type):
                continue
            try:
                _create_fallback_span(event, agent_turn_span)
            except Exception as e:
                logger.error(f"Error creating Fallback span on agent side: {e}")

        for error in turn.errors or []:
            if _route_error_to_side(error) != "agent":
                continue
            try:
                _create_error_catchall_span(error, agent_turn_span)
            except Exception as e:
                logger.error(f"Error creating agent-side error span: {e}")

        return agent_turn_span

    def end_main_turn(self):
        """Completes the main turn span."""
        self.end_span(self.main_turn_span, "All turns processed", end_time=time.perf_counter())
        self.main_turn_span = None

    def agent_say_called(self, message: str, turn_id: Optional[str] = None):
        """Creates a span for the agent's say method."""
        if not self.agent_session_span:
            return

        # Parent directly to the session span. trace.get_current_span() can't be used
        # here: this SDK parents spans manually and never activates them in the OTel
        # context, so get_current_span() always returns INVALID_SPAN (truthy,
        # trace_id=0). Passing that as the parent makes the SDK mint a fresh trace_id,
        # orphaning Agent Say into its own "session".
        attrs: Dict[str, Any] = {"Agent Say Message": message}
        if turn_id:
            attrs["turn_id"] = turn_id
        agent_say_span = create_span(
            "Agent Say",
            attrs,
            parent_span=self.agent_session_span,
            start_time=time.perf_counter()
        )

        self.end_span(agent_say_span, "Agent say span created", end_time=time.perf_counter())

    def agent_reply_called(self, instructions: str, turn_id: Optional[str] = None):
        """Creates a span for an agent reply invocation."""
        if not self.agent_session_span:
            return

        # See agent_say_called: get_current_span() returns INVALID_SPAN here and would
        # orphan the span into a new trace. Parent directly to the session span.
        attrs: Dict[str, Any] = {"Agent Reply Instructions": instructions}
        if turn_id:
            attrs["turn_id"] = turn_id
        agent_reply_span = create_span(
            "Agent Reply",
            attrs,
            parent_span=self.agent_session_span,
            start_time=time.perf_counter()
        )

        self.end_span(agent_reply_span, "Agent reply span created", end_time=time.perf_counter())

    def create_components_change_trace(self, components_change_status: Dict[str, Any], components_change_data: Dict[str, Any], time_data: Dict[str, Any], turn_id: Optional[str] = None) -> None:
        """
        Creates a span for the agent's components change.
        Args:
            components_change_status: Status of the components change.
            components_change_data: Data of the components change.
            time_data: Time data of the components change.
            turn_id: Id of the turn active when the change happened, or None.
        """
        if not self.main_turn_span:
            return

        attr = {}

        if components_change_data.get("new_stt") is not None:
            attr["new_stt"] = components_change_data["new_stt"]
        if components_change_data.get("new_tts") is not None:
            attr["new_tts"] = components_change_data["new_tts"]
        if components_change_data.get("new_llm") is not None:
            attr["new_llm"] = components_change_data["new_llm"]
        if components_change_data.get("new_vad") is not None:
            attr["new_vad"] = components_change_data["new_vad"]
        if components_change_data.get("new_turn_detector") is not None:
            attr["new_turn_detector"] = components_change_data["new_turn_detector"]
        if components_change_data.get("new_denoise") is not None:
            attr["new_denoise"] = components_change_data["new_denoise"]
        if components_change_status:
            attr["components_change_status"] = components_change_status
        if turn_id:
            attr["turn_id"] = turn_id

        self.components_change_span = create_span(
            "Components Change",
            attr,
            parent_span=self.main_turn_span,
            start_time=time_data.get("start_time", time.perf_counter())
        )

        self.end_span(self.components_change_span, "Components change span created", end_time=time_data.get("end_time", time.perf_counter()))
        self.components_change_span = None

    def create_pipeline_change_trace(self, time_data: Dict[str, Any], original_pipeline_config: Dict[str, Any], new_pipeline_config: Dict[str, Any], turn_id: Optional[str] = None) -> None:
        """
        Creates a span for the agent's pipeline change.
        Args:
            time_data: Time data of the pipeline change.
            original_pipeline_config: Original pipeline configuration.
            new_pipeline_config: New pipeline configuration.
            turn_id: Id of the turn active when the change happened, or None.
        """
        if not self.main_turn_span:
            return

        attr: Dict[str, Any] = {
            "original_pipeline_config": original_pipeline_config,
            "new_pipeline_config": new_pipeline_config
        }
        if turn_id:
            attr["turn_id"] = turn_id
        pipeline_change_span = create_span(
            "Pipeline Change",
            attr,
            parent_span=self.main_turn_span,
            start_time=time_data.get("start_time", time.perf_counter())
        )

        self.end_span(pipeline_change_span, "Pipeline change span created", end_time=time_data.get("end_time", time.perf_counter()))

    def create_a2a_trace(self, name: str, attributes: Dict[str, Any], turn_id: Optional[str] = None) -> Optional[Span]:
        """Creates an A2A trace under the main turn span. `turn_id` (turn active when A2A began, or None) tags the parent for renderer placement."""
        if not self.main_turn_span:
            return None

        if not self.a2a_span:
            a2a_attrs: Dict[str, Any] = {"total_a2a_turns": self._a2a_turn_count}
            if turn_id:
                a2a_attrs["turn_id"] = turn_id
            self.a2a_span = create_span(
                "Agent-to-Agent Communications",
                a2a_attrs,
                parent_span=self.main_turn_span
            )

        if not self.a2a_span:
            return None

        self._a2a_turn_count += 1
        span_name = f"A2A {self._a2a_turn_count}: {name}"
        
        a2a_span = create_span(
            span_name, 
            {
                **attributes,
                "a2a_turn_number": self._a2a_turn_count,
                "parent_span": "Agent-to-Agent Communications"
            }, 
            parent_span=self.a2a_span,
            start_time=time.perf_counter()
        )
        
        return a2a_span

    def end_a2a_trace(self, span: Optional[Span], message: str = ""):
        """Ends an A2A trace span."""
        complete_span(span, StatusCode.OK, end_time=time.perf_counter())

    def end_a2a_communication(self):
        """Ends the A2A communication parent span."""
        complete_span(self.a2a_span, StatusCode.OK, end_time=time.perf_counter())
        self.a2a_span = None
        self._a2a_turn_count = 0  

    def create_thinking_audio_start_span(self, file_path: str = None, looping: bool = False, start_time: float = None):
        """Creates a 'Playing Thinking Audio' point-in-time span at session level."""
        if not self.main_turn_span:
            return None

        attrs = {"event": "start", "looping": looping}
        if file_path:
            attrs["file_path"] = file_path

        t = start_time or time.perf_counter()
        span = create_span("Playing Thinking Audio", attrs, parent_span=self.main_turn_span, start_time=t)
        self.end_span(span, message="Thinking audio started", end_time=t)
        return span

    def create_thinking_audio_stop_span(self, end_time: float = None):
        """Creates a 'Stopped Thinking Audio' point-in-time span at session level."""
        if not self.main_turn_span:
            return None

        t = end_time or time.perf_counter()
        span = create_span("Stopped Thinking Audio", {"event": "stop"}, parent_span=self.main_turn_span, start_time=t)
        self.end_span(span, message="Thinking audio stopped", end_time=t)
        return span

    def create_background_audio_start_span(self, file_path: str = None, looping: bool = False, start_time: float = None, turn_id: Optional[str] = None):
        """Creates a 'Playing Background Audio' span. `turn_id` (turn active at start, or None) lets the renderer place it below that turn."""
        if not self.main_turn_span:
            return None

        bg_audio_attrs = {}
        if file_path:
            bg_audio_attrs["file_path"] = file_path
        bg_audio_attrs["looping"] = looping
        bg_audio_attrs["event"] = "start"
        if turn_id:
            bg_audio_attrs["turn_id"] = turn_id

        start_span = create_span("Playing Background Audio", bg_audio_attrs, parent_span=self.main_turn_span, start_time=start_time or time.perf_counter())
        # End immediately as a point-in-time event
        self.end_span(start_span, message="Background audio started", end_time=start_time or time.perf_counter())
        return start_span

    def create_background_audio_stop_span(self, file_path: str = None, looping: bool = False, end_time: float = None, turn_id: Optional[str] = None):
        """Creates a 'Stopped Background Audio' span. `turn_id` (turn active at stop, or None) lets the renderer place it below that turn."""
        if not self.main_turn_span:
            return None

        bg_audio_attrs = {}
        if file_path:
            bg_audio_attrs["file_path"] = file_path
        bg_audio_attrs["looping"] = looping
        bg_audio_attrs["event"] = "stop"
        if turn_id:
            bg_audio_attrs["turn_id"] = turn_id

        stop_span = create_span("Stopped Background Audio", bg_audio_attrs, parent_span=self.main_turn_span, start_time=end_time or time.perf_counter())
        # End immediately as a point-in-time event
        self.end_span(stop_span, message="Background audio stopped", end_time=end_time or time.perf_counter())
        return stop_span

    def end_agent_session(self):
        """Completes the agent session span."""
        if self.main_turn_span:
            self.end_main_turn()
        self.end_span(self.agent_session_span, "Agent session ended", end_time=time.perf_counter())
        self.agent_session_span = None

    def agent_meeting_end(self):
        """Completes the root span."""
        if self.agent_session_span:
            self.end_agent_session()
        if self.agent_session_config_span:
            self.end_agent_session_config()
        if self.agent_session_closed_span:
            self.end_agent_session_closed()
        self.end_span(self.root_span, "Agent left meeting", end_time=time.perf_counter())
        self.root_span = None
        # Normal shutdown path: cleanup is done, let atexit drop the strong
        # reference so the manager can be GC'd in long-running workers.
        self._atexit_done = True
        try:
            atexit.unregister(self._atexit_end_meeting)
        except Exception:
            pass

    def _atexit_end_meeting(self) -> None:
        """Last-resort: end any still-open long-lived spans on process exit.

        Runs only when `agent_meeting_end()` did NOT run on the normal shutdown
        path (e.g., KeyboardInterrupt before Room.disconnect, uncaught
        exception, SIGTERM). Idempotent — `agent_meeting_end()` sets
        `_atexit_done` and unregisters this hook on success.
        """
        if self._atexit_done:
            return
        self._atexit_done = True
        try:
            if self.root_span is not None or self.agent_session_span is not None or self.main_turn_span is not None:
                self.agent_meeting_end()
        except Exception as e:
            try:
                print(f"[TRACES] atexit end-meeting failed: {e}")
            except Exception:
                pass

    def end_span(self, span: Optional[Span], message: str = "", status_code: StatusCode = StatusCode.OK, end_time: Optional[float] = None):
        """Completes a given span with a status."""
        if span:
            if end_time is None:
                end_time = time.perf_counter()
            desc = message if status_code == StatusCode.ERROR else ""
            complete_span(span, status_code, desc, end_time)