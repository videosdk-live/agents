from __future__ import annotations

import asyncio
import hashlib
import time
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Union
from dataclasses import asdict

from .metrics_schema import (
    TurnMetrics,
    SessionMetrics,
    ParticipantMetrics,
    TimelineEvent,
    VadMetrics,
    SttMetrics,
    EouMetrics,
    LlmMetrics,
    TtsMetrics,
    RealtimeMetrics,
    InterruptionMetrics,
    FunctionToolMetrics,
    McpToolMetrics,
    KbMetrics,
    FallbackEvent,
)
from .analytics import AnalyticsClient

if TYPE_CHECKING:
    from .traces_flow import TracesFlowManager
    from ..playground_manager import PlaygroundManager

logger = logging.getLogger(__name__)

REALTIME_PROVIDER_CLASS_NAMES = frozenset({
    "GeminiRealtime",
    "OpenAIRealtime",    
})


class MetricsCollector:
    """Single metrics collector for all pipeline modes.

    Replaces both CascadingMetricsCollector and RealtimeMetricsCollector
    with one collector that uses the component-wise metrics schema.
    """

    def __init__(self) -> None:
        self.session = SessionMetrics()
        self.current_turn: Optional[TurnMetrics] = None
        self.turns: List[TurnMetrics] = []
        self.analytics_client = AnalyticsClient()
        self.traces_flow_manager: Optional[TracesFlowManager] = None
        self.playground_manager: Optional[PlaygroundManager] = None
        self.playground: bool = False

        # Pipeline configuration
        self.pipeline_mode: Optional[str] = None
        self.realtime_mode: Optional[str] = None
        self.active_components: Optional[frozenset] = None

        # Transient timing state (not part of schema)
        self._stt_start_time: Optional[float] = None
        self._llm_start_time: Optional[float] = None
        self._tts_start_time: Optional[float] = None
        self._eou_start_time: Optional[float] = None
        self._tts_first_byte_time: Optional[float] = None

        # Speech state tracking
        self._is_agent_speaking: bool = False
        self._is_user_speaking: bool = False
        self._user_input_start_time: Optional[float] = None
        self._user_speech_end_time: Optional[float] = None
        self._agent_speech_start_time: Optional[float] = None
        self._pending_user_start_time: Optional[float] = None

        # Turn counting
        self._total_turns: int = 0
        self._total_interruptions: int = 0

        # Realtime agent speech end timer
        self._agent_speech_end_timer: Optional[asyncio.TimerHandle] = None

    # ──────────────────────────────────────────────
    # Session lifecycle
    # ──────────────────────────────────────────────

    def configure_pipeline(
        self,
        pipeline_mode: Any,
        realtime_mode: Any = None,
        active_components: Any = None,
    ) -> None:
        """Configure the collector with pipeline information.

        Args:
            pipeline_mode: PipelineMode enum value
            realtime_mode: RealtimeMode enum value (optional)
            active_components: frozenset of PipelineComponent (optional)
        """
        self.pipeline_mode = pipeline_mode.value if hasattr(pipeline_mode, "value") else str(pipeline_mode)
        self.realtime_mode = realtime_mode.value if realtime_mode and hasattr(realtime_mode, "value") else (str(realtime_mode) if realtime_mode else None)
        self.active_components = active_components

        self.session.pipeline_mode = self.pipeline_mode
        self.session.realtime_mode = self.realtime_mode

        if active_components:
            self.session.components = sorted(
                c.value if hasattr(c, "value") else str(c) for c in active_components
            )

        logger.info(f"[metrics] Pipeline configured: mode={self.pipeline_mode}, realtime={self.realtime_mode}")

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for metrics tracking."""
        self.session.session_id = session_id
        self.analytics_client.set_session_id(session_id)

    def set_system_instructions(self, instructions: str) -> None:
        """Set the system instructions for this session."""
        self.session.system_instruction = instructions

    def set_provider_info(self, component_type: str, provider_class: str, model_name: str) -> None:
        """Set provider info for a specific component.

        Args:
            component_type: e.g. "stt", "llm", "tts", "vad", "eou", "realtime"
            provider_class: class name of the provider
            model_name: model identifier
        """
        self.session.provider_per_component[component_type] = {
            "provider_class": provider_class,
            "model_name": model_name,
        }

    def update_provider_class(self, component_type: str, provider_class: str) -> None:
        """Update the provider class for a specific component when fallback occurs.
        
        Args:
            component_type: "STT", "LLM", "TTS", etc.
            provider_class: The new provider class name (e.g., "GoogleLLM")
        """
        if component_type in self.session.provider_per_component:
            self.session.provider_per_component[component_type]["provider_class"] = provider_class
            logger.info(f"Updated {component_type} provider class to: {provider_class}")

    @staticmethod
    def _eou_config_to_dict(eou_config: Any) -> Dict[str, Any]:
        """Convert EOUConfig to a serializable dict for session storage."""
        if eou_config is None:
            return {}
        return {
            "mode": getattr(eou_config, "mode", None),
            "min_max_speech_wait_timeout": getattr(eou_config, "min_max_speech_wait_timeout", None),
        }

    @staticmethod
    def _interrupt_config_to_dict(interrupt_config: Any) -> Dict[str, Any]:
        """Convert InterruptConfig to a serializable dict for session storage."""
        if interrupt_config is None:
            return {}
        return {
            "mode": getattr(interrupt_config, "mode", None),
            "interrupt_min_duration": getattr(interrupt_config, "interrupt_min_duration", None),
            "interrupt_min_words": getattr(interrupt_config, "interrupt_min_words", None),
            "false_interrupt_pause_duration": getattr(interrupt_config, "false_interrupt_pause_duration", None),
            "resume_on_false_interrupt": getattr(interrupt_config, "resume_on_false_interrupt", None),
        }

    def set_eou_config(self, eou_config: Any) -> None:
        """Store EOU config on session for later use (internal tracking, not sent)."""
        self.session.eou_config = self._eou_config_to_dict(eou_config)

    def set_interrupt_config(self, interrupt_config: Any) -> None:
        """Store Interrupt config on session for later use (internal tracking, not sent)."""
        self.session.interrupt_config = self._interrupt_config_to_dict(interrupt_config)

    def add_participant_metrics(
        self,
        participant_id: str,
        kind: Optional[str] = None,
        sip_user: Optional[bool] = None,
        join_time: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a participant entry (agent or user) into session.participant_metrics."""
        self.session.participant_metrics.append(
            ParticipantMetrics(
                participant_id=participant_id,
                kind=kind,
                sip_user=sip_user,
                join_time=join_time,
                meta=meta,
            )
        )

    def set_traces_flow_manager(self, manager: TracesFlowManager) -> None:
        """Set the TracesFlowManager instance."""
        self.traces_flow_manager = manager

    def set_playground_manager(self, manager: Optional[PlaygroundManager]) -> None:
        """Set the PlaygroundManager instance."""
        self.playground = True
        self.playground_manager = manager

    def finalize_session(self) -> None:
        """Finalize the session, completing any in-progress turn."""
        if self.current_turn:
            self.complete_turn()
        self.session.session_end_time = time.time()

    # ──────────────────────────────────────────────
    # Turn lifecycle
    # ──────────────────────────────────────────────

    def _generate_turn_id(self) -> str:
        """Generate a hash-based turn ID."""
        timestamp = str(time.time())
        session_id = self.session.session_id or "default"
        turn_count = str(self._total_turns)
        hash_input = f"{timestamp}_{session_id}_{turn_count}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    @staticmethod
    def _round_latency(latency: float) -> float:
        """Convert latency from seconds to milliseconds and round."""
        return round(latency * 1000, 4)

    def start_turn(self) -> None:
        """Start a new turn. Completes any existing turn first."""
        logger.info(f"[metrics] start_turn() called | current_turn={'exists' if self.current_turn else 'None'} | total_turns={self._total_turns}")
        if self.current_turn:
            logger.info(f"[metrics] start_turn() completing existing turn before creating new one")
            self.complete_turn()

        self._total_turns += 1
        self.current_turn = TurnMetrics(turn_id=self._generate_turn_id())

        # Carry over pending user start time from a previously discarded turn
        if self._pending_user_start_time is not None:
            self.current_turn.user_speech_start_time = self._pending_user_start_time
            self._start_timeline_event("user_speech", self._pending_user_start_time)

        # If user is currently speaking, capture the start time
        if self._is_user_speaking and self._user_input_start_time:
            if self.current_turn.user_speech_start_time is None:
                self.current_turn.user_speech_start_time = self._user_input_start_time
                if not any(ev.event_type == "user_speech" for ev in self.current_turn.timeline_event_metrics):
                    self._start_timeline_event("user_speech", self._user_input_start_time)

    def complete_turn(self) -> None:
        """Complete the current turn, calculate E2E, validate, serialize and send."""
        logger.info(f"[metrics] complete_turn() called | current_turn={'exists' if self.current_turn else 'None'} | total_turns={self._total_turns}")
        if not self.current_turn:
            logger.info(f"[metrics] complete_turn() early return — no current_turn")
            return

        # Calculate E2E latency
        self.current_turn.compute_e2e_latency()

        # Validate turn has meaningful data
        has_stt = bool(self.current_turn.stt_metrics and any(s.stt_latency is not None for s in self.current_turn.stt_metrics))
        has_llm = bool(self.current_turn.llm_metrics and any(l.llm_ttft is not None for l in self.current_turn.llm_metrics))
        has_tts = bool(self.current_turn.tts_metrics and any(t.tts_latency is not None or t.ttfb is not None for t in self.current_turn.tts_metrics))
        logger.info(f"[metrics] complete_turn() validation | stt={has_stt} llm={has_llm} tts={has_tts} | user_speech_start={self.current_turn.user_speech_start_time is not None} | e2e={self.current_turn.e2e_latency}")
        if not self._validate_turn(self.current_turn) and self._total_turns > 1:
            logger.warning(f"[metrics] complete_turn() DISCARDING turn — validation failed | total_turns={self._total_turns}")
            # Cache user start time for next turn
            if self.current_turn.user_speech_start_time is not None:
                if (self._pending_user_start_time is None or
                        self.current_turn.user_speech_start_time < self._pending_user_start_time):
                    self._pending_user_start_time = self.current_turn.user_speech_start_time
                    logger.info(f"[metrics] Caching earliest user start: {self._pending_user_start_time}")
            self.current_turn = None
            return

        # Send trace
        if self.traces_flow_manager:
            self.traces_flow_manager.create_unified_turn_trace(self.current_turn, self.session)

        self.turns.append(self.current_turn)

        # Send to playground
        if self.playground and self.playground_manager:
            self.playground_manager.send_cascading_metrics(metrics=self.current_turn, full_turn_data=True)

        # Serialize and send analytics
        interaction_data = self._serialize_turn(self.current_turn)
        transformed_data = self._transform_to_camel_case(interaction_data)
        transformed_data = self._remove_internal_fields(transformed_data)

        if len(self.turns) > 1:
            self._remove_provider_fields(transformed_data)

        transformed_data = self._remove_negatives(transformed_data)

        self.analytics_client.send_interaction_analytics_safe(transformed_data)

        self.current_turn = None
        self._pending_user_start_time = None

        # Reset transient timing state so stale values don't leak into the next turn
        self._stt_start_time = None
        self._is_user_speaking = False

    def schedule_turn_complete(self, timeout: float = 1.0) -> None:
        """Schedule turn completion after a timeout (for realtime modes)."""
        if self._agent_speech_end_timer:
            self._agent_speech_end_timer.cancel()

        try:
            loop = asyncio.get_event_loop()
            self._agent_speech_end_timer = loop.call_later(timeout, self._finalize_realtime_turn)
        except RuntimeError:
            # No event loop available, complete immediately
            self.complete_turn()

    def _finalize_realtime_turn(self) -> None:
        """Finalize a realtime turn after the timeout."""
        if not self.current_turn:
            return

        # Ensure agent speech end time is set
        if self.current_turn.agent_speech_start_time and not self.current_turn.agent_speech_end_time:
            self.current_turn.agent_speech_end_time = time.perf_counter()

        # Ensure user speech end time is set
        if self.current_turn.user_speech_start_time and not self.current_turn.user_speech_end_time:
            self.current_turn.user_speech_end_time = time.perf_counter()

        # Close any open timeline events
        current_time = time.perf_counter()
        for event in self.current_turn.timeline_event_metrics:
            if event.end_time is None:
                event.end_time = current_time
                event.duration_ms = round((current_time - event.start_time) * 1000, 4)

        # Compute realtime latencies
        self._compute_realtime_latencies()

        self._agent_speech_end_timer = None
        self.complete_turn()

    def _compute_realtime_latencies(self) -> None:
        """Compute TTFB, thinking_delay, agent_speech_duration for realtime turns."""
        turn = self.current_turn
        if not turn:
            return

        if turn.user_speech_start_time and turn.agent_speech_start_time:
            ttfb = (turn.agent_speech_start_time - turn.user_speech_start_time) * 1000
            turn.e2e_latency = round(ttfb, 4)  # For realtime, e2e is user_start → agent_start

        if turn.user_speech_end_time and turn.agent_speech_start_time:
            thinking_delay = (turn.agent_speech_start_time - turn.user_speech_end_time) * 1000
            # Store in realtime metrics if present
            if turn.realtime_metrics:
                pass  # Will be enhanced if RealtimeMetrics gets thinking_delay field

        if turn.agent_speech_start_time and turn.agent_speech_end_time:
            turn.agent_speech_duration = round(
                (turn.agent_speech_end_time - turn.agent_speech_start_time) * 1000, 4
            )

    # ──────────────────────────────────────────────
    # VAD metrics
    # ──────────────────────────────────────────────

    def on_user_speech_start(self) -> None:
        """Called when user starts speaking (VAD start)."""
        if not self.current_turn:
            self.start_turn()

        logger.info(f"[metrics] on_user_speech_start() called | _is_user_speaking={self._is_user_speaking} | current_turn={'exists' if self.current_turn else 'None'}")
        if self._is_user_speaking:
            logger.info(f"[metrics] on_user_speech_start() early return — already speaking")
            return

        self._is_user_speaking = True
        self._user_input_start_time = time.perf_counter()

        if self.current_turn:
            if self.current_turn.user_speech_start_time is None:
                self.current_turn.user_speech_start_time = self._user_input_start_time

            if not any(ev.event_type == "user_speech" for ev in self.current_turn.timeline_event_metrics):
                self._start_timeline_event("user_speech", self._user_input_start_time)

    def on_user_speech_end(self) -> None:
        """Called when user stops speaking (VAD end)."""
        logger.info(f"[metrics] on_user_speech_end() called | current_turn={'exists' if self.current_turn else 'None'}")
        self._is_user_speaking = False
        self._user_speech_end_time = time.perf_counter()

        if self.current_turn and self.current_turn.user_speech_start_time:
            self.current_turn.user_speech_end_time = self._user_speech_end_time
            self.current_turn.user_speech_duration = self._round_latency(
                self.current_turn.user_speech_end_time - self.current_turn.user_speech_start_time
            )
            self._end_timeline_event("user_speech", self._user_speech_end_time)
            logger.info(f"user speech duration: {self.current_turn.user_speech_duration}ms")

            if self.playground and self.playground_manager:
                self.playground_manager.send_cascading_metrics(
                    metrics={"user_speech_duration": self.current_turn.user_speech_duration}
                )

    def set_vad_config(self, vad_config: Dict[str, Any]) -> None:
        """Set VAD configuration."""
        if not self.current_turn.vad_metrics:
            self.current_turn.vad_metrics.append(VadMetrics())

        vad = self.current_turn.vad_metrics[-1]
        vad.vad_threshold = vad_config.get("threshold")
        vad.vad_min_speech_duration = vad_config.get("min_speech_duration")
        vad.vad_min_silence_duration = vad_config.get("min_silence_duration")

    # ──────────────────────────────────────────────
    # STT metrics
    # ──────────────────────────────────────────────

    def on_stt_start(self) -> None:
        """Called when STT processing starts."""
        self._stt_start_time = time.perf_counter()
        if self.current_turn:
            logger.info(f"[metrics] on_stt_start() called | current_turn={'exists' if self.current_turn else 'None'}")
            # Create STT metrics entry if not present
            if not self.current_turn.stt_metrics:
                self.current_turn.stt_metrics.append(SttMetrics())
            stt = self.current_turn.stt_metrics[-1]
            stt.stt_start_time = self._stt_start_time

    def on_stt_complete(self, transcript: str = "", duration: float = 0.0, confidence: float = 0.0) -> None:
        """Called when STT processing completes."""
        if self.current_turn and self.current_turn.stt_metrics:
            stt = self.current_turn.stt_metrics[-1]
            if stt.stt_preemptive_generation_enabled and stt.stt_preemptive_generation_occurred:
                logger.info("STT preemptive generation occurred, skipping stt complete")
                return

        if not self.current_turn:
            return

        if not self.current_turn.stt_metrics:
            self.current_turn.stt_metrics.append(SttMetrics())
        stt = self.current_turn.stt_metrics[-1]

        stt_end_time = time.perf_counter()
        stt.stt_end_time = stt_end_time

        if self._stt_start_time:
            stt_latency = self._round_latency(stt_end_time - self._stt_start_time)
            stt.stt_latency = stt_latency
            stt.stt_confidence = confidence
            stt.stt_duration = duration
            logger.info(f"stt latency: {stt_latency}ms | stt confidence: {confidence} | stt duration: {duration}ms")

            if self.playground and self.playground_manager:
                self.playground_manager.send_cascading_metrics(metrics={"stt_latency": stt_latency})

        if transcript:
            stt.stt_transcript = transcript

        self._stt_start_time = None

    def on_stt_preflight_end(self) -> None:
        """Called when STT preflight event is received."""
        if self.current_turn and self.current_turn.stt_metrics:
            stt = self.current_turn.stt_metrics[-1]
            if stt.stt_start_time:
                stt.stt_preflight_end_time = time.perf_counter()
                stt.stt_preflight_latency = self._round_latency(
                    stt.stt_preflight_end_time - stt.stt_start_time
                )
                logger.info(f"stt preflight latency: {stt.stt_preflight_latency}ms")

                if self.playground and self.playground_manager:
                    self.playground_manager.send_cascading_metrics(
                        metrics={"stt_preflight_latency": stt.stt_preflight_latency}
                    )

    def on_stt_interim_end(self) -> None:
        """Called when STT interim event is received."""
        if self.current_turn and self.current_turn.stt_metrics:
            stt = self.current_turn.stt_metrics[-1]
            if stt.stt_start_time and not stt.stt_interim_latency:
                stt.stt_interim_end_time = time.perf_counter()
                stt.stt_interim_latency = self._round_latency(
                    stt.stt_interim_end_time - stt.stt_start_time
                )
                logger.info(f"stt interim latency: {stt.stt_interim_latency}ms")

                if self.playground and self.playground_manager:
                    self.playground_manager.send_cascading_metrics(
                        metrics={"stt_interim_latency": stt.stt_interim_latency}
                    )

    def set_stt_usage(self, input_tokens: int = 0, output_tokens: int = 0, total_tokens: int = 0) -> None:
        """Set STT token usage."""
        if self.current_turn:
            if not self.current_turn.stt_metrics:
                self.current_turn.stt_metrics.append(SttMetrics())
            stt = self.current_turn.stt_metrics[-1]
            stt.stt_input_tokens = input_tokens
            stt.stt_output_tokens = output_tokens
            stt.stt_total_tokens = total_tokens

    def set_preemptive_generation_enabled(self) -> None:
        """Mark preemptive generation as enabled for current STT."""
        if self.current_turn:
            if not self.current_turn.stt_metrics:
                self.current_turn.stt_metrics.append(SttMetrics())
            self.current_turn.stt_metrics[-1].stt_preemptive_generation_enabled = True

    # ──────────────────────────────────────────────
    # EOU metrics
    # ──────────────────────────────────────────────

    def on_eou_start(self) -> None:
        """Called when EOU processing starts."""
        if not self.current_turn:
            self.start_turn()
        self._eou_start_time = time.perf_counter()
        if self.current_turn:
            if not self.current_turn.eou_metrics:
                self.current_turn.eou_metrics.append(EouMetrics())
            self.current_turn.eou_metrics[-1].eou_start_time = self._eou_start_time

    def on_eou_complete(self, probability: Optional[float] = None) -> None:
        """Called when EOU processing completes."""
        if self._eou_start_time:
            eou_end_time = time.perf_counter()
            eou_latency = self._round_latency(eou_end_time - self._eou_start_time)

            if self.current_turn:
                if not self.current_turn.eou_metrics:
                    self.current_turn.eou_metrics.append(EouMetrics())
                eou = self.current_turn.eou_metrics[-1]
                eou.eou_end_time = eou_end_time
                eou.eou_latency = eou_latency
                if probability is not None:
                    eou.eou_probability = probability
                logger.info(f"eou latency: {eou_latency}ms")

                if self.playground and self.playground_manager:
                    self.playground_manager.send_cascading_metrics(metrics={"eou_latency": eou_latency})

            self._eou_start_time = None

    # ──────────────────────────────────────────────
    # LLM metrics
    # ──────────────────────────────────────────────

    def on_llm_start(self) -> None:
        """Called when LLM processing starts."""
        self._llm_start_time = time.perf_counter()
        if self.current_turn:
            if not self.current_turn.llm_metrics:
                self.current_turn.llm_metrics.append(LlmMetrics())
            self.current_turn.llm_metrics[-1].llm_start_time = self._llm_start_time

    def on_llm_first_token(self) -> None:
        """Called when first LLM token is received."""
        if self.current_turn and self.current_turn.llm_metrics:
            llm = self.current_turn.llm_metrics[-1]
            if llm.llm_start_time:
                llm.llm_first_token_time = time.perf_counter()
                llm.llm_ttft = self._round_latency(llm.llm_first_token_time - llm.llm_start_time)
                logger.info(f"llm ttft: {llm.llm_ttft}ms")

                if self.playground and self.playground_manager:
                    self.playground_manager.send_cascading_metrics(metrics={"llm_ttft": llm.llm_ttft})

    def on_llm_complete(self) -> None:
        """Called when LLM processing completes."""
        if self._llm_start_time:
            llm_end_time = time.perf_counter()
            llm_duration = self._round_latency(llm_end_time - self._llm_start_time)

            if self.current_turn and self.current_turn.llm_metrics:
                llm = self.current_turn.llm_metrics[-1]
                llm.llm_end_time = llm_end_time
                llm.llm_duration = llm_duration
                logger.info(f"llm duration: {llm_duration}ms")

                if self.playground and self.playground_manager:
                    self.playground_manager.send_cascading_metrics(metrics={"llm_duration": llm_duration})

            self._llm_start_time = None

    def set_llm_input(self, text: str) -> None:
        """Record the actual text sent to LLM."""
        if self.current_turn:
            if not self.current_turn.llm_metrics:
                self.current_turn.llm_metrics.append(LlmMetrics())
            self.current_turn.llm_metrics[-1].llm_input = text

    def set_llm_usage(self, usage: Dict[str, Any]) -> None:
        """Set LLM token usage and calculate TPS."""
        if not self.current_turn or not usage:
            return

        if not self.current_turn.llm_metrics:
            self.current_turn.llm_metrics.append(LlmMetrics())

        llm = self.current_turn.llm_metrics[-1]
        llm.prompt_tokens = usage.get("prompt_tokens")
        llm.completion_tokens = usage.get("completion_tokens")
        llm.total_tokens = usage.get("total_tokens")
        llm.prompt_cached_tokens = usage.get("prompt_cached_tokens")

        if llm.llm_duration and llm.llm_duration > 0 and llm.completion_tokens and llm.completion_tokens > 0:
            latency_seconds = llm.llm_duration / 1000
            llm.tokens_per_second = round(llm.completion_tokens / latency_seconds, 2)

        if self.playground and self.playground_manager:
            self.playground_manager.send_cascading_metrics(metrics={
                "prompt_tokens": llm.prompt_tokens,
                "completion_tokens": llm.completion_tokens,
                "total_tokens": llm.total_tokens,
                "tokens_per_second": llm.tokens_per_second,
            })

    # ──────────────────────────────────────────────
    # TTS metrics
    # ──────────────────────────────────────────────

    def on_tts_start(self) -> None:
        """Called when TTS processing starts."""
        if not self.current_turn:
            self.start_turn()
        self._tts_start_time = time.perf_counter()
        self._tts_first_byte_time = None
        if self.current_turn:
            if not self.current_turn.tts_metrics:
                self.current_turn.tts_metrics.append(TtsMetrics())
            self.current_turn.tts_metrics[-1].tts_start_time = self._tts_start_time

    def on_tts_first_byte(self) -> None:
        """Called when TTS produces first audio byte."""
        if self._tts_start_time:
            now = time.perf_counter()
            if self.current_turn and self.current_turn.tts_metrics:
                tts = self.current_turn.tts_metrics[-1]
                tts.tts_end_time = now
                tts.ttfb = self._round_latency(now - self._tts_start_time)
                tts.tts_first_byte_time = now
                logger.info(f"tts ttfb: {tts.ttfb}ms")

                if self.playground and self.playground_manager:
                    self.playground_manager.send_cascading_metrics(metrics={"ttfb": tts.ttfb})

            self._tts_first_byte_time = now

    def on_agent_speech_start(self) -> None:
        """Called when agent starts speaking (actual audio output)."""
        if not self.current_turn:
            self.start_turn()
        self._is_agent_speaking = True
        self._agent_speech_start_time = time.perf_counter()

        if self.current_turn:
            self.current_turn.agent_speech_start_time = self._agent_speech_start_time
            if not any(
                ev.event_type == "agent_speech" and ev.end_time is None
                for ev in self.current_turn.timeline_event_metrics
            ):
                self._start_timeline_event("agent_speech", self._agent_speech_start_time)

    def on_agent_speech_end(self) -> None:
        """Called when agent stops speaking."""
        logger.info(f"[metrics] on_agent_speech_end() called | current_turn={'exists' if self.current_turn else 'None'} | _tts_start={self._tts_start_time is not None}")
        self._is_agent_speaking = False
        agent_speech_end_time = time.perf_counter()

        if self.current_turn:
            self._end_timeline_event("agent_speech", agent_speech_end_time)
            self.current_turn.agent_speech_end_time = agent_speech_end_time

        # Calculate TTS latency from first_byte - start
        if self._tts_start_time and self._tts_first_byte_time:
            total_tts_latency = self._tts_first_byte_time - self._tts_start_time
            if self.current_turn and self.current_turn.agent_speech_start_time:
                if self.current_turn.tts_metrics:
                    tts = self.current_turn.tts_metrics[-1]
                    tts.tts_latency = self._round_latency(total_tts_latency)

                self.current_turn.agent_speech_duration = self._round_latency(
                    agent_speech_end_time - self.current_turn.agent_speech_start_time
                )

                if self.playground and self.playground_manager:
                    self.playground_manager.send_cascading_metrics(
                        metrics={"tts_latency": self.current_turn.tts_metrics[-1].tts_latency if self.current_turn.tts_metrics else None}
                    )
                    self.playground_manager.send_cascading_metrics(
                        metrics={"agent_speech_duration": self.current_turn.agent_speech_duration}
                    )

            self._tts_start_time = None
            self._tts_first_byte_time = None
        elif self._tts_start_time:
            self._tts_start_time = None
            self._tts_first_byte_time = None

    def add_tts_characters(self, count: int) -> None:
        """Add to the total character count for the current turn."""
        if self.current_turn:
            if not self.current_turn.tts_metrics:
                self.current_turn.tts_metrics.append(TtsMetrics())
            tts = self.current_turn.tts_metrics[-1]
            tts.tts_characters = (tts.tts_characters or 0) + count

            if self.playground and self.playground_manager:
                self.playground_manager.send_cascading_metrics(metrics={"tts_characters": tts.tts_characters})

    # ──────────────────────────────────────────────
    # Interruption metrics
    # ──────────────────────────────────────────────

    def on_interrupted(self) -> None:
        """Called when user interrupts the agent."""
        if self._is_agent_speaking:
            self._total_interruptions += 1
            if self.current_turn:
                self.current_turn.is_interrupted = True
                logger.info(f"User interrupted the agent. Total interruptions: {self._total_interruptions}")

            if self.playground and self.playground_manager and self.current_turn:
                self.playground_manager.send_cascading_metrics(
                    metrics={"interrupted": self.current_turn.is_interrupted}
                )

    def on_false_interrupt_start(self, pause_duration: Optional[float] = None) -> None:
        """Called when a false interrupt is detected."""
        if self.current_turn:
            if not self.current_turn.interruption_metrics:
                self.current_turn.interruption_metrics = InterruptionMetrics()
            im = self.current_turn.interruption_metrics
            im.is_false_interrupt = True
            im.false_interrupt_start_time = time.perf_counter()
            if pause_duration is not None:
                im.false_interrupt_pause_duration = pause_duration

    def on_false_interrupt_resume(self) -> None:
        """Called when agent resumes after a false interrupt."""
        if self.current_turn and self.current_turn.interruption_metrics:
            im = self.current_turn.interruption_metrics
            im.resumed_after_false_interrupt = True
            im.resume_on_false_interrupt = True
            im.false_interrupt_end_time = time.perf_counter()
            if im.false_interrupt_start_time:
                im.false_interrupt_duration = self._round_latency(
                    im.false_interrupt_end_time - im.false_interrupt_start_time
                )

    def on_false_interrupt_escalated(self, word_count: Optional[int] = None) -> None:
        """Called when a false interrupt escalates to a real interrupt."""
        if self.current_turn and self.current_turn.interruption_metrics:
            im = self.current_turn.interruption_metrics
            im.is_false_interrupt = False
            if word_count is not None:
                im.false_interrupt_words = word_count

    # ──────────────────────────────────────────────
    # Tool metrics
    # ──────────────────────────────────────────────

    def add_function_tool_call(
        self,
        tool_name: str,
        tool_params: Optional[Union[Dict[str, Any], list]] = None,
        tool_response: Optional[Dict[str, Any]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> None:
        """Track a function tool call in the current turn."""
        if self.current_turn:
            self.current_turn.function_tools_called.append(tool_name)

            tool_metric = FunctionToolMetrics(
                tool_name=tool_name,
                tool_params=tool_params or {},
                tool_response=tool_response or {},
                start_time=start_time or time.perf_counter(),
                end_time=end_time,
            )
            if tool_metric.start_time and tool_metric.end_time:
                tool_metric.latency = self._round_latency(tool_metric.end_time - tool_metric.start_time)

            self.current_turn.function_tool_metrics.append(tool_metric)

            # Also track in function_tool_timestamps for backward compat
            tool_timestamp = {
                "tool_name": tool_name,
                "timestamp": start_time or time.perf_counter(),
                "readable_time": time.strftime("%H:%M:%S", time.localtime()),
            }
            self.current_turn.function_tool_timestamps.append(tool_timestamp)

            if self.playground and self.playground_manager:
                self.playground_manager.send_cascading_metrics(
                    metrics={"function_tool_timestamps": self.current_turn.function_tool_timestamps}
                )

    def add_mcp_tool_call(
        self,
        tool_type: str = "local",
        tool_url: Optional[str] = None,
        tool_params: Optional[Union[Dict[str, Any], list]] = None,
        tool_response: Optional[Dict[str, Any]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> None:
        """Track an MCP tool call in the current turn."""
        if self.current_turn:
            tool_metric = McpToolMetrics(
                type=tool_type,
                tool_url=tool_url,
                tool_params=tool_params or {},
                tool_response=tool_response or {},
                start_time=start_time or time.perf_counter(),
                end_time=end_time,
            )
            if tool_metric.start_time and tool_metric.end_time:
                tool_metric.latency = self._round_latency(tool_metric.end_time - tool_metric.start_time)

            self.current_turn.mcp_tool_metrics.append(tool_metric)

    # ──────────────────────────────────────────────
    # Transcript / response
    # ──────────────────────────────────────────────

    def set_user_transcript(self, transcript: str) -> None:
        """Set the user transcript for the current turn."""
        if self.current_turn:
            if self._is_agent_speaking and self.current_turn.user_speech:
                logger.info(
                    f"[metrics] Skipping set_user_transcript during agent speech "
                    f"— current turn already has user_speech, new transcript "
                    f"belongs to the next turn: {transcript}"
                )
                return

            self.current_turn.user_speech = transcript
            logger.info(f"user input speech: {transcript}")

            # Update timeline
            user_events = [
                ev for ev in self.current_turn.timeline_event_metrics if ev.event_type == "user_speech"
            ]
            if user_events:
                user_events[-1].text = transcript
            else:
                current_time = time.perf_counter()
                self._start_timeline_event("user_speech", current_time)
                if self.current_turn.timeline_event_metrics:
                    self.current_turn.timeline_event_metrics[-1].text = transcript

            if self.playground and self.playground_manager:
                self.playground_manager.send_cascading_metrics(
                    metrics={"user_speech": self.current_turn.user_speech}
                )

    def set_agent_response(self, response: str) -> None:
        """Set the agent response for the current turn."""

        if not self.current_turn:
            self.start_turn()
        self.current_turn.agent_speech = response
        logger.info(f"agent output speech: {response}")

        if not any(ev.event_type == "agent_speech" for ev in self.current_turn.timeline_event_metrics):
            current_time = time.perf_counter()
            self._start_timeline_event("agent_speech", current_time)

        self._update_timeline_event_text("agent_speech", response)

        if self.playground and self.playground_manager:
            self.playground_manager.send_cascading_metrics(
                metrics={"agent_speech": self.current_turn.agent_speech}
            )

    # ──────────────────────────────────────────────
    # Knowledge Base
    # ──────────────────────────────────────────────

    def on_knowledge_base_start(self, kb_id: Optional[str] = None) -> None:
        """Called when knowledge base processing starts."""
        if self.current_turn:
            kb_metric = KbMetrics(
                kb_id=kb_id,
                kb_start_time=time.perf_counter(),
            )
            self.current_turn.kb_metrics.append(kb_metric)

    def on_knowledge_base_complete(self, documents: List[str], scores: List[float]) -> None:
        """Called when knowledge base processing completes."""
        if self.current_turn and self.current_turn.kb_metrics:
            kb = self.current_turn.kb_metrics[-1]
            kb.kb_documents = documents
            kb.kb_scores = scores
            kb.kb_end_time = time.perf_counter()

            if kb.kb_start_time:
                kb.kb_retrieval_latency = self._round_latency(kb.kb_end_time - kb.kb_start_time)
                logger.info(f"kb retrieval latency: {kb.kb_retrieval_latency}ms")

            if self.playground and self.playground_manager:
                self.playground_manager.send_cascading_metrics(
                    metrics={
                        "kb_id": kb.kb_id,
                        "kb_retrieval_latency": kb.kb_retrieval_latency,
                        "kb_documents": kb.kb_documents,
                        "kb_scores": kb.kb_scores,
                    }
                )

    # ──────────────────────────────────────────────
    # A2A
    # ──────────────────────────────────────────────

    def set_a2a_handoff(self) -> None:
        """Set the A2A enabled and handoff occurred flags."""
        if self.current_turn:
            self.current_turn.is_a2a_enabled = True
            self.current_turn.handoff_occurred = True

            if self.playground and self.playground_manager:
                self.playground_manager.send_cascading_metrics(
                    metrics={"handoff_occurred": self.current_turn.handoff_occurred}
                )

    # ──────────────────────────────────────────────
    # Error tracking
    # ──────────────────────────────────────────────

    def add_error(self, source: str, message: str) -> None:
        """Add an error to the current turn."""
        if self.current_turn:
            self.current_turn.errors.append({
                "source": source,
                "message": message,
                "timestamp": time.time(),
            })

            if self.playground and self.playground_manager:
                self.playground_manager.send_cascading_metrics(
                    metrics={"errors": self.current_turn.errors}
                )

    def on_fallback_event(self, event_data: Dict[str, Any]) -> None:
        """Record a fallback event for the current turn"""
        if not self.current_turn:
            self.start_turn()
        print(f"Received fallback event: {event_data}")
        if self.current_turn:
            fallback_event = FallbackEvent(
                component_type=event_data.get("component_type", ""),
                temporary_disable_sec=event_data.get("temporary_disable_sec", 0),
                permanent_disable_after_attempts=event_data.get("permanent_disable_after_attempts", 0),
                recovery_attempt=event_data.get("recovery_attempt", 0),
                message=event_data.get("message", ""),
                start_time=event_data.get("start_time"),
                end_time=event_data.get("end_time"),
                duration_ms=event_data.get("duration_ms"),
                original_provider_label=event_data.get("original_provider_label"),
                original_connection_start=event_data.get("original_connection_start"),
                original_connection_end=event_data.get("original_connection_end"),
                original_connection_duration_ms=event_data.get("original_connection_duration_ms"),
                new_provider_label=event_data.get("new_provider_label"),
                new_connection_start=event_data.get("new_connection_start"),
                new_connection_end=event_data.get("new_connection_end"),
                new_connection_duration_ms=event_data.get("new_connection_duration_ms"),
                is_recovery=event_data.get("is_recovery", False),
            )
            self.current_turn.fallback_events.append(fallback_event)
            logger.info(f"Fallback event recorded: {event_data.get('component_type')} - {event_data.get('message')}")

    # ──────────────────────────────────────────────
    # Realtime-specific metrics
    # ──────────────────────────────────────────────

    def set_realtime_usage(self, usage: Dict[str, Any]) -> None:
        """Set realtime model token usage from a flat dict (input_tokens, total_tokens, cached_*, thoughts_tokens, etc.)."""
        if not self.current_turn or not usage:
            return

        if not self.current_turn.realtime_metrics:
            self.current_turn.realtime_metrics.append(RealtimeMetrics())

        rt = self.current_turn.realtime_metrics[-1]
        rt.realtime_input_tokens = usage.get("input_tokens")
        rt.realtime_output_tokens = usage.get("output_tokens")
        rt.realtime_total_tokens = usage.get("total_tokens")
        rt.realtime_input_text_tokens = usage.get("input_text_tokens")
        rt.realtime_input_audio_tokens = usage.get("input_audio_tokens")
        rt.realtime_input_image_tokens = usage.get("input_image_tokens")
        rt.realtime_input_cached_tokens = usage.get("input_cached_tokens")
        rt.realtime_thoughts_tokens = usage.get("thoughts_tokens")
        rt.realtime_cached_text_tokens = usage.get("cached_text_tokens")
        rt.realtime_cached_audio_tokens = usage.get("cached_audio_tokens")
        rt.realtime_cached_image_tokens = usage.get("cached_image_tokens")
        rt.realtime_output_text_tokens = usage.get("output_text_tokens")
        rt.realtime_output_audio_tokens = usage.get("output_audio_tokens")
        rt.realtime_output_image_tokens = usage.get("output_image_tokens")

    def set_realtime_model_error(self, error: Dict[str, Any]) -> None:
        """Track a realtime model error."""
        if self.current_turn:
            logger.error(f"realtime model error: {error}")
            self.current_turn.errors.append({
                "source": "REALTIME_MODEL",
                "message": str(error.get("message", "Unknown error")),
                "timestamp": time.time(),
            })

    # ──────────────────────────────────────────────
    # Timeline helpers
    # ──────────────────────────────────────────────

    def _start_timeline_event(self, event_type: str, start_time: float) -> None:
        """Start a timeline event."""
        if self.current_turn:
            event = TimelineEvent(event_type=event_type, start_time=start_time)
            self.current_turn.timeline_event_metrics.append(event)

    def _end_timeline_event(self, event_type: str, end_time: float) -> None:
        """End a timeline event and calculate duration."""
        if self.current_turn:
            for event in reversed(self.current_turn.timeline_event_metrics):
                if event.event_type == event_type and event.end_time is None:
                    event.end_time = end_time
                    event.duration_ms = self._round_latency(end_time - event.start_time)
                    break

    def _update_timeline_event_text(self, event_type: str, text: str) -> None:
        """Update the most recent timeline event of this type with text content."""
        if self.current_turn:
            for event in reversed(self.current_turn.timeline_event_metrics):
                if event.event_type == event_type:
                    event.text = text
                    break

    # ──────────────────────────────────────────────
    # Validation & Serialization
    # ──────────────────────────────────────────────

    def _validate_turn(self, turn: TurnMetrics) -> bool:
        """Check that the turn has at least one meaningful latency metric."""
        has_stt = bool(turn.stt_metrics and any(s.stt_latency is not None for s in turn.stt_metrics))
        has_llm = bool(turn.llm_metrics and any(l.llm_ttft is not None for l in turn.llm_metrics))
        has_tts = bool(turn.tts_metrics and any(t.tts_latency is not None or t.ttfb is not None for t in turn.tts_metrics))
        has_eou = bool(turn.eou_metrics and any(e.eou_latency is not None for e in turn.eou_metrics))
        has_realtime = bool(turn.realtime_metrics)
        has_e2e = turn.e2e_latency is not None

        return any([has_stt, has_llm, has_tts, has_eou, has_realtime, has_e2e])

    def _serialize_turn(self, turn: TurnMetrics) -> Dict[str, Any]:
        """Serialize a TurnMetrics to a flat dict for analytics API."""
        data: Dict[str, Any] = {}

        # Top-level turn fields
        data["timestamp"] = turn.timestamp
        data["e2e_latency"] = turn.e2e_latency
        data["interrupted"] = turn.is_interrupted
        data["user_speech_start_time"] = turn.user_speech_start_time
        data["user_speech_end_time"] = turn.user_speech_end_time
        data["user_speech_duration"] = turn.user_speech_duration
        data["user_speech"] = turn.user_speech
        data["agent_speech_start_time"] = turn.agent_speech_start_time
        data["agent_speech_end_time"] = turn.agent_speech_end_time
        data["agent_speech_duration"] = turn.agent_speech_duration
        data["agent_speech"] = turn.agent_speech
        data["function_tools_called"] = turn.function_tools_called
        data["function_tool_timestamps"] = turn.function_tool_timestamps
        data["is_a2a_enabled"] = turn.is_a2a_enabled
        data["handoff_occurred"] = turn.handoff_occurred
        data["errors"] = turn.errors

        # Flatten the last STT metrics entry
        if turn.stt_metrics:
            stt = turn.stt_metrics[-1]
            data["stt_latency"] = stt.stt_latency
            data["stt_start_time"] = stt.stt_start_time
            data["stt_end_time"] = stt.stt_end_time
            data["stt_preflight_latency"] = stt.stt_preflight_latency
            data["stt_interim_latency"] = stt.stt_interim_latency
            data["stt_confidence"] = stt.stt_confidence
            data["stt_duration"] = stt.stt_duration

        # Flatten the last EOU metrics entry
        if turn.eou_metrics:
            eou = turn.eou_metrics[-1]
            data["eou_latency"] = eou.eou_latency
            data["eou_start_time"] = eou.eou_start_time
            data["eou_end_time"] = eou.eou_end_time

        # Flatten the last LLM metrics entry
        if turn.llm_metrics:
            llm = turn.llm_metrics[-1]
            data["llm_ttft"] = llm.llm_ttft
            data["llm_duration"] = llm.llm_duration
            data["llm_start_time"] = llm.llm_start_time
            data["llm_end_time"] = llm.llm_end_time
            data["prompt_tokens"] = llm.prompt_tokens
            data["completion_tokens"] = llm.completion_tokens
            data["total_tokens"] = llm.total_tokens
            data["prompt_cached_tokens"] = llm.prompt_cached_tokens
            data["tokens_per_second"] = llm.tokens_per_second

        # Flatten the last KB metrics entry
        if turn.kb_metrics:
            kb = turn.kb_metrics[-1]
            data["kb_id"] = kb.kb_id
            data["kb_documents"] = kb.kb_documents
            data["kb_scores"] = kb.kb_scores
            data["kb_retrieval_latency"] = kb.kb_retrieval_latency
        # Flatten the last TTS metrics entry
        if turn.tts_metrics:
            tts = turn.tts_metrics[-1]
            data["tts_latency"] = tts.tts_latency
            data["ttfb"] = tts.ttfb
            data["tts_start_time"] = tts.tts_start_time
            data["tts_end_time"] = tts.tts_end_time
            data["tts_duration"] = tts.tts_duration
            data["tts_characters"] = tts.tts_characters

        # Flatten the last realtime (full s2s) metrics entry for analytics
        if turn.realtime_metrics:
            rt = turn.realtime_metrics[-1]
            data["realtime_input_tokens"] = rt.realtime_input_tokens
            data["realtime_total_tokens"] = rt.realtime_total_tokens
            data["realtime_output_tokens"] = rt.realtime_output_tokens
            data["realtime_input_text_tokens"] = rt.realtime_input_text_tokens
            data["realtime_input_audio_tokens"] = rt.realtime_input_audio_tokens
            data["realtime_input_image_tokens"] = rt.realtime_input_image_tokens
            data["realtime_input_cached_tokens"] = rt.realtime_input_cached_tokens
            data["realtime_thoughts_tokens"] = rt.realtime_thoughts_tokens
            data["realtime_cached_text_tokens"] = rt.realtime_cached_text_tokens
            data["realtime_cached_audio_tokens"] = rt.realtime_cached_audio_tokens
            data["realtime_cached_image_tokens"] = rt.realtime_cached_image_tokens
            data["realtime_output_text_tokens"] = rt.realtime_output_text_tokens
            data["realtime_output_audio_tokens"] = rt.realtime_output_audio_tokens
            data["realtime_output_image_tokens"] = rt.realtime_output_image_tokens

        # Provider info (from session). Use realtime_* keys for known S2S models.
        providers = self.session.provider_per_component
        if "realtime" in providers:
            data["realtime_provider_class"] = providers["realtime"]["provider_class"]
            data["realtime_model_name"] = providers["realtime"]["model_name"]
        elif "llm" in providers:
            pc = providers["llm"]["provider_class"]
            if pc in REALTIME_PROVIDER_CLASS_NAMES:
                data["realtime_provider_class"] = pc
                data["realtime_model_name"] = providers["llm"]["model_name"]
            else:
                data["llm_provider_class"] = pc
                data["llm_model_name"] = providers["llm"]["model_name"]
        if "stt" in providers:
            data["stt_provider_class"] = providers["stt"]["provider_class"]
            data["stt_model_name"] = providers["stt"]["model_name"]
        if "tts" in providers:
            data["tts_provider_class"] = providers["tts"]["provider_class"]
            data["tts_model_name"] = providers["tts"]["model_name"]
        if "vad" in providers:
            data["vad_provider_class"] = providers["vad"]["provider_class"]
            data["vad_model_name"] = providers["vad"]["model_name"]
        if "eou" in providers:
            data["eou_provider_class"] = providers["eou"]["provider_class"]
            data["eou_model_name"] = providers["eou"]["model_name"]

        # System instructions (first turn only)
        if self._total_turns == 1 or len(self.turns) == 0:
            data["system_instructions"] = self.session.system_instruction
        else:
            data["system_instructions"] = ""

        # Timeline
        data["timeline"] = [asdict(ev) for ev in turn.timeline_event_metrics]

        return data

    def _transform_to_camel_case(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform snake_case field names to camelCase for analytics."""
        field_mapping = {
            # User and agent metrics 
            "user_speech_start_time": "userSpeechStartTime",
            "user_speech_end_time": "userSpeechEndTime",
            "user_speech_duration": "userSpeechDuration",
            "agent_speech_start_time": "agentSpeechStartTime",
            "agent_speech_end_time": "agentSpeechEndTime",
            "agent_speech_duration": "agentSpeechDuration",

            # STT metrics
            "stt_latency": "sttLatency",
            "stt_start_time": "sttStartTime",
            "stt_end_time": "sttEndTime",
            "stt_preflight_latency": "sttPreflightLatency",
            "stt_interim_latency": "sttInterimLatency",
            "stt_confidence": "sttConfidence",
            "stt_duration": "sttDuration",

            # LLM metrics
            "llm_duration": "llmDuration",
            "llm_start_time": "llmStartTime",
            "llm_end_time": "llmEndTime",
            "llm_ttft": "ttft",
            "prompt_tokens": "promptTokens",
            "completion_tokens": "completionTokens",
            "total_tokens": "totalTokens",
            "prompt_cached_tokens": "promptCachedTokens",
            "tokens_per_second": "tokensPerSecond",

            # TTS metrics
            "tts_start_time": "ttsStartTime",
            "tts_end_time": "ttsEndTime",
            "tts_duration": "ttsDuration",
            "tts_characters": "ttsCharacters",
            "ttfb": "ttfb",
            "tts_latency": "ttsLatency",

            # EOU metrics
            "eou_latency": "eouLatency",
            "eou_start_time": "eouStartTime",
            "eou_end_time": "eouEndTime",

            # Realtime (full s2s) token metrics
            "realtime_input_tokens": "realtimeInputTokens",
            "realtime_total_tokens": "realtimeTotalTokens",
            "realtime_output_tokens": "realtimeOutputTokens",
            "realtime_input_text_tokens": "realtimeInputTextTokens",
            "realtime_input_audio_tokens": "realtimeInputAudioTokens",
            "realtime_input_image_tokens": "realtimeInputImageTokens",
            "realtime_input_cached_tokens": "realtimeInputCachedTokens",
            "realtime_thoughts_tokens": "realtimeThoughtsTokens",
            "realtime_cached_text_tokens": "realtimeCachedTextTokens",
            "realtime_cached_audio_tokens": "realtimeCachedAudioTokens",
            "realtime_cached_image_tokens": "realtimeCachedImageTokens",
            "realtime_output_text_tokens": "realtimeOutputTextTokens",
            "realtime_output_audio_tokens": "realtimeOutputAudioTokens",
            "realtime_output_image_tokens": "realtimeOutputImageTokens",

            "kb_id": "kbId",
            "kb_retrieval_latency": "kbRetrievalLatency",
            "kb_documents": "kbDocuments",
            "kb_scores": "kbScores",

            # Provider metrics
            "llm_provider_class": "llmProviderClass",
            "llm_model_name": "llmModelName",
            "realtime_provider_class": "realtimeProviderClass",
            "realtime_model_name": "realtimeModelName",
            "stt_provider_class": "sttProviderClass",
            "stt_model_name": "sttModelName",
            "tts_provider_class": "ttsProviderClass",
            "tts_model_name": "ttsModelName",

            # VAD metrics
            "vad_provider_class": "vadProviderClass",
            "vad_model_name": "vadModelName",

            # EOU metrics
            "eou_provider_class": "eouProviderClass",
            "eou_model_name": "eouModelName",

            # Other metrics
            "e2e_latency": "e2eLatency",
            "interrupted": "interrupted",
            "timestamp": "timestamp",
            "function_tools_called": "functionToolsCalled",
            "function_tool_timestamps": "functionToolTimestamps",
            "system_instructions": "systemInstructions",
            "handoff_occurred": "handOffOccurred",
            "is_a2a_enabled": "isA2aEnabled",
            "errors": "errors",
        }

        timeline_field_mapping = {
            "event_type": "eventType",
            "start_time": "startTime",
            "end_time": "endTime",
            "duration_ms": "durationInMs",
        }

        transformed: Dict[str, Any] = {}
        for key, value in data.items():
            camel_key = field_mapping.get(key, key)
            if key == "timeline" and isinstance(value, list):
                transformed_timeline = []
                for event in value:
                    transformed_event = {}
                    for ek, ev in event.items():
                        transformed_event[timeline_field_mapping.get(ek, ek)] = ev
                    transformed_timeline.append(transformed_event)
                transformed[camel_key] = transformed_timeline
            else:
                transformed[camel_key] = value

        return transformed

    def _remove_internal_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove internal-only fields from the analytics payload."""
        always_remove = [
            "errors",
            "functionToolTimestamps",
            "sttStartTime", "sttEndTime",
            "ttsStartTime", "ttsEndTime",
            "llmStartTime", "llmEndTime",
            "eouStartTime", "eouEndTime",
            "isA2aEnabled",
            "interactionId",
            "timestamp",
        ]

        if self.current_turn and not self.current_turn.is_a2a_enabled:
            always_remove.append("handOffOccurred")

        for f in always_remove:
            data.pop(f, None)

        return data

    def _remove_provider_fields(self, data: Dict[str, Any]) -> None:
        """Remove provider fields after first turn."""
        provider_fields = [
            "systemInstructions",
            # "llmProviderClass", "llmModelName",
            # "sttProviderClass", "sttModelName",
            # "ttsProviderClass", "ttsModelName",
            "vadProviderClass", "vadModelName",
            "eouProviderClass", "eouModelName",
        ]
        for f in provider_fields:
            data.pop(f, None)

    def _remove_negatives(self, obj: Any) -> Any:
        """Recursively clamp any numeric value < 0 to 0."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (int, float)) and v < 0:
                    obj[k] = 0
                elif isinstance(v, (dict, list)):
                    obj[k] = self._remove_negatives(v)
            return obj
        if isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, (int, float)) and v < 0:
                    obj[i] = 0
                elif isinstance(v, (dict, list)):
                    obj[i] = self._remove_negatives(v)
            return obj
        return obj
