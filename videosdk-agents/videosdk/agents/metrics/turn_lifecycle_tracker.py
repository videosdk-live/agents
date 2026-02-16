from __future__ import annotations
from typing import Dict, List, Any, Optional
import time
import uuid
import logging
from dataclasses import asdict

from .metrics_schema import (
    TurnMetrics,
    InterruptionMetrics,
    TimelineEvent,
    FunctionToolMetrics,
    EouMetrics,
)
from .component_manager import ComponentMetricsManager

logger = logging.getLogger(__name__)

class TurnLifecycleTracker:
    """
    Tracks turn lifecycle, boundaries, and calculates E2E metrics.

    Turn boundaries depend on active components:
    - STT + LLM + TTS: user_speech_start → synthesis_complete
    - LLM + TTS only: agent.say() → synthesis_complete
    - STT + LLM only: user_speech_start → generation_complete
    - Realtime: realtime_user_speech_start → realtime_agent_speech_end
    """

    def __init__(self, component_manager: ComponentMetricsManager, pipeline: Any = None):
        self.component_manager = component_manager
        self.current_turn: Optional[TurnMetrics] = None
        self.completed_turns: List[TurnMetrics] = []
        self._pipeline_config = pipeline  # Store pipeline for config access

        # Track timestamps for E2E calculation
        self._turn_start_time: Optional[float] = None
        self._component_timestamps: Dict[str, Dict[str, float]] = {}

        # Buffer for timestamps that arrive before turn is created
        self._buffered_stt_start_time: Optional[float] = None

    def start_turn(self, trigger: str = "user_speech") -> TurnMetrics:
        """
        Start a new turn and initialize component metrics.

        Args:
            trigger: What triggered the turn ('user_speech', 'agent_say', 'realtime')

        Returns:
            New TurnMetrics instance
        """
        turn_id = f"turn_{str(uuid.uuid4())[:8]}"
        self._turn_start_time = time.perf_counter()

        # Create turn with component metrics instances
        turn = TurnMetrics(turn_id=turn_id)

        # Initialize metrics for each active component
        active_types = self.component_manager.get_active_component_types()
        logger.info(f"[METRICS DEBUG] start_turn: Active component types: {active_types}")

        for component_type in active_types:
            self._add_component_metric(turn, component_type)

        self.current_turn = turn

        # Apply buffered STT start time if it exists
        if self._buffered_stt_start_time is not None and turn.stt_metrics:
            turn.stt_metrics[-1].stt_start_time = self._buffered_stt_start_time
            logger.info(f"[METRICS DEBUG] Applied buffered stt_start_time = {self._buffered_stt_start_time}")
            self._buffered_stt_start_time = None  # Clear the buffer

        logger.info(f"Started turn: {turn_id} (trigger={trigger})")
        logger.info(f"[METRICS DEBUG] start_turn: Created turn with STT metrics: {len(turn.stt_metrics)}, EOU: {len(turn.eou_metrics)}, TTS: {len(turn.tts_metrics)}")

        return turn

    def _add_component_metric(self, turn: TurnMetrics, component_type: str) -> None:
        """Helper to create and append a new metric instance for a component."""
        metrics = self.component_manager.create_turn_metrics(component_type)
        if metrics:
            logger.info(f"[METRICS DEBUG] Created new {component_type} metrics")
            if component_type == 'vad':
                turn.vad_metrics.append(metrics)
            elif component_type == 'stt':
                turn.stt_metrics.append(metrics)
            elif component_type == 'eou':
                turn.eou_metrics.append(metrics)
            elif component_type == 'llm':
                turn.llm_metrics.append(metrics)
            elif component_type == 'tts':
                turn.tts_metrics.append(metrics)
            elif component_type == 'realtime':
                turn.realtime_metrics.append(metrics)

    def complete_turn(self) -> Optional[TurnMetrics]:
        """
        Complete the current turn, calculate E2E, validate, and archive.

        Returns:
            Completed TurnMetrics if valid, None otherwise
        """
        if not self.current_turn:
            logger.warning("No active turn to complete")
            return None

        turn = self.current_turn

        # Calculate E2E latency
        self._calculate_e2e_latency(turn)

        # Validate turn has at least one metric
        if not self._validate_turn(turn):
            logger.warning(f"Turn {turn.turn_id} has no valid metrics, discarding")
            self.current_turn = None
            return None

        # Archive turn
        self.completed_turns.append(turn)
        self.current_turn = None

        logger.info(
            f"Completed turn: {turn.turn_id} "
            f"(e2e={turn.e2e_latency}ms, components={self._get_turn_components(turn)})"
        )

        return turn

    def _calculate_e2e_latency(self, turn: TurnMetrics) -> None:
        """
        Calculate end-to-end latency by summing component latencies.
        Only includes latencies from components that are present.

        E2E = sum([STT_latency, EOU_latency, LLM_TTFT, TTS_latency])
        """
        total_ms = 0.0

        # STT latency (use the first one that has latency, or sum? usually E2E is linear path)
        # We'll use the LAST STT event if multiple exist, assuming the last one triggered the response.
        if turn.stt_metrics:
            last_valid_stt = next((s for s in reversed(turn.stt_metrics) if s.stt_latency), None)
            if last_valid_stt:
                total_ms += self._to_milliseconds(last_valid_stt.stt_latency)

        # EOU latency
        if turn.eou_metrics:
            last_valid_eou = next((e for e in reversed(turn.eou_metrics) if e.eou_latency), None)
            if last_valid_eou:
                total_ms += self._to_milliseconds(last_valid_eou.eou_latency)

        # LLM TTFT (time to first token)
        if turn.llm_metrics:
            last_valid_llm = next((l for l in reversed(turn.llm_metrics) if l.llm_ttft), None)
            if last_valid_llm:
                total_ms += self._to_milliseconds(last_valid_llm.llm_ttft)

        # TTS latency
        if turn.tts_metrics:
            last_valid_tts = next((t for t in reversed(turn.tts_metrics) if t.ttfb), None)
            if last_valid_tts:
                total_ms += self._to_milliseconds(last_valid_tts.ttfb)

        # Round to 4 decimal places
        turn.e2e_latency = round(total_ms, 4) if total_ms > 0 else None

    def _validate_turn(self, turn: TurnMetrics) -> bool:
        """
        Validate that turn has at least one meaningful metric.
        A valid turn must have at least one of: stt_latency, llm_ttft, ttfb, or eou_latency.
        """
        # Check STT
        if turn.stt_metrics:
            for stt in turn.stt_metrics:
                if stt.stt_latency:
                    return True

        # Check EOU
        if turn.eou_metrics:
            for eou in turn.eou_metrics:
                if eou.eou_latency:
                    return True

        # Check LLM
        if turn.llm_metrics:
            for llm in turn.llm_metrics:
                if llm.llm_ttft:
                    return True

        # Check TTS
        if turn.tts_metrics:
            for tts in turn.tts_metrics:
                if tts.ttfb:
                    return True

        # Check Realtime
        if turn.realtime_metrics:
            return True

        return False

    def _to_milliseconds(self, seconds: float) -> float:
        """Convert seconds to milliseconds and round to 4 decimals."""
        return round(seconds * 1000, 4)

    def _get_turn_components(self, turn: TurnMetrics) -> List[str]:
        """Get list of components present in this turn."""
        components = []
        if turn.vad_metrics:
            components.append('vad')
        if turn.stt_metrics:
            components.append('stt')
        if turn.eou_metrics:
            components.append('eou')
        if turn.llm_metrics:
            components.append('llm')
        if turn.tts_metrics:
            components.append('tts')
        if turn.realtime_metrics:
            components.append('realtime')
        return components

    # Event handlers for updating turn metrics

    def on_user_speech_start(self, timestamp: float, transcript: str = "") -> None:
        """Called when user starts speaking."""
        if not self.current_turn:
            self.start_turn(trigger="user_speech")

        if self.current_turn:
            # Only set the turn-level start time if it's not set
            if self.current_turn.user_speech_start_time is None:
                self.current_turn.user_speech_start_time = timestamp
                logger.info(f"[METRICS DEBUG] Set turn start time: {timestamp}")

            if transcript:
                # Append to existing transcript or set if new
                if self.current_turn.user_speech:
                    self.current_turn.user_speech += " " + transcript
                else:
                    self.current_turn.user_speech = transcript

            # Handle VAD metrics list - create new if last one is closed
            if self.current_turn.vad_metrics:
                last_vad = self.current_turn.vad_metrics[-1]
                if last_vad.user_speech_end_time is not None or last_vad.vad_end_of_speech_time is not None:
                    # Last VAD segment finished, create a new one
                    logger.info("[METRICS DEBUG] Last VAD finished, creating new active VAD metric")
                    self._add_component_metric(self.current_turn, 'vad')
            
            # Update the (now guaranteed active) last VAD metric
            if self.current_turn.vad_metrics:
                self.current_turn.vad_metrics[-1].user_speech_start_time = timestamp

    def on_user_speech_end(self, timestamp: float) -> None:
        """Called when user stops speaking."""
        if self.current_turn:
            # Set end time on turn level (updates to the LATEST end time)
            self.current_turn.user_speech_end_time = timestamp
            
            if self.current_turn.user_speech_start_time:
                duration = timestamp - self.current_turn.user_speech_start_time
                self.current_turn.user_speech_duration = round(duration, 4)

            # Update VAD metric
            if self.current_turn.vad_metrics:
                self.current_turn.vad_metrics[-1].user_speech_end_time = timestamp

    def on_agent_speech_start(self, timestamp: float) -> None:
        """Called when agent starts speaking."""
        if self.current_turn:
            self.current_turn.agent_speech_start_time = timestamp

    def on_agent_speech_end(self, timestamp: float, text: str = "") -> None:
        """Called when agent finishes speaking."""
        if self.current_turn:
            self.current_turn.agent_speech_end_time = timestamp
            if self.current_turn.agent_speech_start_time:
                duration = timestamp - self.current_turn.agent_speech_start_time
                self.current_turn.agent_speech_duration = round(duration, 4)
            if text:
                self.current_turn.agent_speech = text

    def on_stt_start(self, timestamp: float) -> None:
        """Called when STT processing starts."""
        logger.info(f"[METRICS DEBUG] on_stt_start called - current_turn exists: {self.current_turn is not None}")

        if self.current_turn:
            # Check if we need a new STT metric (if last one is finished or has transcript)
            if self.current_turn.stt_metrics:
                last_stt = self.current_turn.stt_metrics[-1]
                if last_stt.stt_end_time is not None or last_stt.stt_transcript is not None:
                     logger.info("[METRICS DEBUG] Last STT finished, creating new active STT metric")
                     self._add_component_metric(self.current_turn, 'stt')

            if self.current_turn.stt_metrics:
                self.current_turn.stt_metrics[-1].stt_start_time = timestamp
                logger.info(f"[METRICS DEBUG] Set stt_start_time = {timestamp}")
        else:
            # Buffer the timestamp for when the turn is created
            self._buffered_stt_start_time = timestamp
            logger.info(f"[METRICS DEBUG] Buffered stt_start_time = {timestamp} (turn will be created soon)")

    def on_stt_preflight_end(self, timestamp: float, transcript: str = "") -> None:
        """Called when preflight transcript is received (for preemptive generation)."""
        if self.current_turn and self.current_turn.stt_metrics:
            stt = self.current_turn.stt_metrics[-1]
            stt.stt_preflight_end_time = timestamp
            stt.stt_preflight_transcript = transcript

            # Calculate preflight latency
            if stt.stt_start_time:
                stt.stt_preflight_latency = timestamp - stt.stt_start_time

    def on_stt_interim_end(self, timestamp: float) -> None:
        """Called when interim transcript is received."""
        if self.current_turn and self.current_turn.stt_metrics:
            stt = self.current_turn.stt_metrics[-1]
            stt.stt_interim_end_time = timestamp

            # Calculate interim latency (TTFW - Time To First Word)
            if stt.stt_start_time:
                stt.stt_interim_latency = timestamp - stt.stt_start_time
                stt.stt_ttfw = stt.stt_interim_latency

    def on_stt_complete(
        self,
        timestamp: float,
        transcript: str,
        confidence: float = None,
        duration: float = None,
        metadata: Dict[str, Any] = None,
        is_preemptive: bool = False
    ) -> None:
        """Called when STT completes."""
        logger.info(f"[METRICS DEBUG] on_stt_complete called - confidence: {confidence}, duration: {duration}, current_turn exists: {self.current_turn is not None}")

        # Fallback: if no turn exists, start one now so we never lose STT metrics
        if not self.current_turn:
            logger.info("[METRICS DEBUG] No active turn on stt_complete — starting fallback turn from STT event")
            self.start_turn(trigger="stt_complete")

        if self.current_turn and self.current_turn.stt_metrics:
            stt = self.current_turn.stt_metrics[-1]
            stt.stt_end_time = timestamp
            stt.stt_transcript = transcript
            logger.info(f"[METRICS DEBUG] Set stt_end_time = {timestamp}")

            # Also ensure this transcript is added to the turn-level user speech
            # This handles cases where STT is the source of truth for text
            current_speech = self.current_turn.user_speech or ""
            if transcript and transcript not in current_speech:
                 self.current_turn.user_speech = (current_speech + " " + transcript).strip()

            # Set confidence if provided
            if confidence is not None:
                stt.stt_confidence = confidence
                logger.info(f"[METRICS DEBUG] Set stt_confidence = {confidence}")
            else:
                logger.warning(f"[METRICS DEBUG] No confidence provided to on_stt_complete")

            # Set duration from STT provider if provided
            if duration is not None:
                stt.stt_duration = duration
                logger.info(f"[METRICS DEBUG] Set stt_duration from provider = {duration}")
            else:
                logger.warning(f"[METRICS DEBUG] No duration provided to on_stt_complete")

            # Back-calculate stt_start_time if missing (VAD was missed)
            if stt.stt_start_time is None:
                if duration is not None:
                    stt.stt_start_time = timestamp - duration
                    logger.info(f"[METRICS DEBUG] Back-calculated stt_start_time = {stt.stt_start_time} (end - duration)")
                    # Also update turn-level user_speech_start_time if missing
                    if self.current_turn.user_speech_start_time is None:
                        self.current_turn.user_speech_start_time = stt.stt_start_time
                        logger.info(f"[METRICS DEBUG] Set user_speech_start_time from back-calculated stt_start = {stt.stt_start_time}")
                else:
                    logger.warning("[METRICS DEBUG] stt_start_time is None and no duration to back-calculate")

            # Calculate latency from timestamps
            if stt.stt_start_time:
                stt.stt_latency = timestamp - stt.stt_start_time
                logger.info(f"[METRICS DEBUG] Calculated stt_latency = {stt.stt_latency}")
                # If no duration from provider, use calculated latency
                if stt.stt_duration is None:
                    stt.stt_duration = stt.stt_latency
                    logger.info(f"[METRICS DEBUG] Used calculated latency for duration: {stt.stt_duration}")
            else:
                logger.warning(f"[METRICS DEBUG] Cannot calculate stt_latency - start_time is None")

            # Set preemptive generation flag
            stt.stt_preemptive_generation_occurred = is_preemptive

            # Capture provider-specific metrics from metadata
            if metadata:
                # Check for additional metrics in metadata
                if "metrics" in metadata:
                    provider_metrics = metadata["metrics"]
                    # Store in stt_config for provider-specific data
                    if stt.stt_config is None:
                        stt.stt_config = {}
                    stt.stt_config["provider_metrics"] = provider_metrics

    def on_llm_start(self, timestamp: float, input_text: str = "") -> None:
        """Called when LLM processing starts."""
        # If no turn exists (e.g., process_text flow), start one
        if not self.current_turn:
            logger.info("[METRICS DEBUG] No active turn on llm_start — starting turn for text_input")
            self.start_turn(trigger="text_input")
            # Store the user text as pending say text for the user_speech timeline
            if input_text:
                self._pending_user_text = input_text

        if self.current_turn and self.current_turn.llm_metrics:
            llm = self.current_turn.llm_metrics[-1]
            llm.llm_start_time = timestamp
            if input_text:
                llm.llm_input = input_text

    def on_llm_first_token(self, timestamp: float) -> None:
        """Called when LLM generates first token."""
        if self.current_turn and self.current_turn.llm_metrics:
            llm = self.current_turn.llm_metrics[-1]
            llm.llm_first_token_time = timestamp

            # Calculate TTFT
            if llm.llm_start_time:
                llm.llm_ttft = timestamp - llm.llm_start_time

    def on_llm_complete(self, timestamp: float, tokens: Dict[str, int] = None) -> None:
        """Called when LLM completes generation."""
        if self.current_turn and self.current_turn.llm_metrics:
            llm = self.current_turn.llm_metrics[-1]
            llm.llm_end_time = timestamp

            # Calculate duration and latency
            if llm.llm_start_time:
                llm.llm_duration = timestamp - llm.llm_start_time
                llm.llm_latency = llm.llm_duration

            # Set token usage
            if tokens:
                llm.prompt_tokens = tokens.get('prompt_tokens')
                llm.completion_tokens = tokens.get('completion_tokens')
                llm.total_tokens = tokens.get('total_tokens')
                llm.prompt_cached_tokens = tokens.get('prompt_cached_tokens')

                # Calculate tokens per second
                if llm.llm_duration and llm.completion_tokens:
                    llm.tokens_per_second = round(
                        llm.completion_tokens / llm.llm_duration, 2
                    )

    def on_tts_start(self, timestamp: float) -> None:
        """Called when TTS processing starts."""
        # If no turn exists (e.g., agent.say() call), start one
        if not self.current_turn:
            logger.info("[METRICS DEBUG] No active turn on tts_start — starting turn for say()")
            self.start_turn(trigger="agent_say")
            # Set llm_input to system prompt for say() turns (first interaction)
            if self._pipeline_config and hasattr(self._pipeline_config, 'agent') and self._pipeline_config.agent:
                agent = self._pipeline_config.agent
                if hasattr(agent, 'instructions') and agent.instructions and self.current_turn.llm_metrics:
                    self.current_turn.llm_metrics[-1].llm_input = agent.instructions

        # Apply pending say text if available (works for both say() and process_text flows)
        if hasattr(self, '_pending_say_text') and self._pending_say_text and self.current_turn:
            self.current_turn.agent_speech = self._pending_say_text
            self._pending_say_text = None

        if self.current_turn and self.current_turn.tts_metrics:
            self.current_turn.tts_metrics[-1].tts_start_time = timestamp

    def on_tts_first_byte(self, timestamp: float) -> None:
        """Called when TTS generates first audio byte."""
        if self.current_turn and self.current_turn.tts_metrics:
            tts = self.current_turn.tts_metrics[-1]
            tts.tts_first_byte_time = timestamp

            # Calculate TTFB
            if tts.tts_start_time:
                tts.ttfb = timestamp - tts.tts_start_time

    def on_tts_complete(self, timestamp: float, characters: int = None) -> None:
        """Called when TTS completes."""
        if self.current_turn and self.current_turn.tts_metrics:
            tts = self.current_turn.tts_metrics[-1]
            tts.tts_end_time = timestamp

            # Calculate duration and latency
            if tts.tts_start_time:
                tts.tts_duration = timestamp - tts.tts_start_time
                tts.tts_latency = tts.tts_duration

            if characters:
                tts.tts_characters = characters

    def on_eou_start(self, timestamp: float) -> None:
        """Called when EOU detection starts."""
        logger.info(f"[METRICS DEBUG] on_eou_start called - current_turn exists: {self.current_turn is not None}")

        if self.current_turn:
            logger.info(f"[METRICS DEBUG] eou_metrics list: {len(self.current_turn.eou_metrics) if self.current_turn.eou_metrics else 0} items")
            
            # Check if we need a new EOU metric (if last one is finished or has end time)
            if self.current_turn.eou_metrics:
                 last_eou = self.current_turn.eou_metrics[-1]
                 if last_eou.eou_end_time is not None:
                     logger.info("[METRICS DEBUG] Last EOU finished, creating new active EOU metric")
                     self._add_component_metric(self.current_turn, 'eou')

        if self.current_turn and self.current_turn.eou_metrics:
            eou = self.current_turn.eou_metrics[-1]
            eou.eou_start_time = timestamp
            logger.info(f"[METRICS DEBUG] Set eou_start_time = {timestamp}")

            # Populate EOU config if not already set (do this once per turn)
            if eou.min_speech_wait_timeout is None:
                logger.info(f"[METRICS DEBUG] Calling _populate_eou_config")
                self._populate_eou_config(eou)
                logger.info(f"[METRICS DEBUG] After populate: min={eou.min_speech_wait_timeout}, max={eou.max_speech_wait_timeout}")
        else:
            logger.warning(f" [METRICS DEBUG] Cannot set eou_start_time - turn or eou_metrics missing")

    def on_eou_complete(self, timestamp: float, probability: float = None) -> None:
        """Called when EOU detection completes."""
        logger.info(f"[METRICS DEBUG] on_eou_complete called - probability: {probability}, current_turn exists: {self.current_turn is not None}")

        if self.current_turn and self.current_turn.eou_metrics:
            eou = self.current_turn.eou_metrics[-1]
            eou.eou_end_time = timestamp
            logger.info(f"[METRICS DEBUG] Set eou_end_time = {timestamp}")

            if probability is not None:
                eou.eou_probability = probability
                logger.info(f"[METRICS DEBUG] Set eou_probability = {probability}")

            # Calculate latency
            if eou.eou_start_time:
                eou.eou_latency = timestamp - eou.eou_start_time
                logger.info(f"[METRICS DEBUG] Calculated eou_latency = {eou.eou_latency}")
            else:
                logger.warning(f" [METRICS DEBUG] Cannot calculate eou_latency - start_time is None")
        else:
            logger.warning(f" [METRICS DEBUG] Cannot set eou_end_time - turn or eou_metrics missing")

    def on_wait_for_additional_speech(
        self,
        wait_duration: float,
        eou_probability: float = None
    ) -> None:
        """Called when waiting for additional speech based on EOU."""
        if self.current_turn and self.current_turn.eou_metrics:
            eou = self.current_turn.eou_metrics[-1]
            eou.waited_for_additional_speech = True
            eou.wait_for_additional_speech_duration = wait_duration
            if eou_probability is not None:
                eou.eou_probability = eou_probability

    def _populate_eou_config(self, eou: "EouMetrics", pipeline: Any = None) -> None:
        """Populate EOU config from pipeline or parent component."""
        logger.info(f"[METRICS DEBUG] _populate_eou_config called")

        # Try to get config from parent component manager's pipeline
        config = None
        if hasattr(self, '_pipeline_config'):
            config = self._pipeline_config
            logger.info(f"[METRICS DEBUG] Got config from _pipeline_config")
        elif pipeline:
            config = pipeline
            logger.info(f"[METRICS DEBUG] Got config from pipeline parameter")
        else:
            logger.warning(f"[METRICS DEBUG] No config source available")
            return

        logger.info(f"[METRICS DEBUG] Config object: {type(config).__name__}")
        logger.info(f"[METRICS DEBUG] Config has eou_config: {hasattr(config, 'eou_config')}")
        logger.info(f"[METRICS DEBUG] Config has orchestrator: {hasattr(config, 'orchestrator')}")

        # Populate EOU timing config
        if hasattr(config, 'eou_config'):
            eou_config = config.eou_config
            logger.info(f"[METRICS DEBUG] eou_config type: {type(eou_config).__name__}")
            logger.info(f"[METRICS DEBUG] eou_config has min_max_speech_wait_timeout: {hasattr(eou_config, 'min_max_speech_wait_timeout')}")

            if hasattr(eou_config, 'min_max_speech_wait_timeout') and len(eou_config.min_max_speech_wait_timeout) >= 2:
                eou.min_speech_wait_timeout = eou_config.min_max_speech_wait_timeout[0]
                eou.max_speech_wait_timeout = eou_config.min_max_speech_wait_timeout[1]
                logger.info(f"[METRICS DEBUG] Set from eou_config: min={eou.min_speech_wait_timeout}, max={eou.max_speech_wait_timeout}")
        elif hasattr(config, 'orchestrator') and hasattr(config.orchestrator, 'speech_understanding'):
            speech_understanding = config.orchestrator.speech_understanding
            logger.info(f"[METRICS DEBUG] speech_understanding type: {type(speech_understanding).__name__}")
            logger.info(f"[METRICS DEBUG] has min_speech_wait_timeout: {hasattr(speech_understanding, 'min_speech_wait_timeout')}")
            logger.info(f"[METRICS DEBUG] has max_speech_wait_timeout: {hasattr(speech_understanding, 'max_speech_wait_timeout')}")

            if hasattr(speech_understanding, 'min_speech_wait_timeout'):
                eou.min_speech_wait_timeout = speech_understanding.min_speech_wait_timeout
                logger.info(f"[METRICS DEBUG] Set min_speech_wait_timeout = {eou.min_speech_wait_timeout}")
            if hasattr(speech_understanding, 'max_speech_wait_timeout'):
                eou.max_speech_wait_timeout = speech_understanding.max_speech_wait_timeout
                logger.info(f"[METRICS DEBUG] Set max_speech_wait_timeout = {eou.max_speech_wait_timeout}")
        else:
            logger.warning(f" [METRICS DEBUG] No valid EOU config path found")

    def _populate_interrupt_config(self, interrupt_metrics: "InterruptionMetrics") -> None:
        """Populate interrupt config from pipeline."""
        if not self._pipeline_config:
            return

        config = self._pipeline_config

        # Try to get from pipeline interrupt_config
        if hasattr(config, 'interrupt_config'):
            int_config = config.interrupt_config
            if hasattr(int_config, 'mode'):
                interrupt_metrics.interrupt_mode = int_config.mode
            if hasattr(int_config, 'interrupt_min_duration'):
                interrupt_metrics.interrupt_min_duration = int_config.interrupt_min_duration
            if hasattr(int_config, 'interrupt_min_words'):
                interrupt_metrics.interrupt_min_words = int_config.interrupt_min_words
            if hasattr(int_config, 'false_interrupt_pause_duration'):
                interrupt_metrics.false_interrupt_pause_duration = int_config.false_interrupt_pause_duration
            if hasattr(int_config, 'resume_on_false_interrupt'):
                interrupt_metrics.resume_on_false_interrupt = int_config.resume_on_false_interrupt
        # Fallback: try from orchestrator.speech_understanding
        elif hasattr(config, 'orchestrator') and hasattr(config.orchestrator, 'speech_understanding'):
            speech_understanding = config.orchestrator.speech_understanding
            if hasattr(speech_understanding, 'interrupt_mode'):
                interrupt_metrics.interrupt_mode = speech_understanding.interrupt_mode
            if hasattr(speech_understanding, 'interrupt_min_duration'):
                interrupt_metrics.interrupt_min_duration = speech_understanding.interrupt_min_duration
            if hasattr(speech_understanding, 'interrupt_min_words'):
                interrupt_metrics.interrupt_min_words = speech_understanding.interrupt_min_words
            if hasattr(speech_understanding, 'false_interrupt_pause_duration'):
                interrupt_metrics.false_interrupt_pause_duration = speech_understanding.false_interrupt_pause_duration
            if hasattr(speech_understanding, 'resume_on_false_interrupt'):
                interrupt_metrics.resume_on_false_interrupt = speech_understanding.resume_on_false_interrupt

    def add_tts_characters(self, count: int) -> None:
        """Add to the total character count for the current turn's TTS."""
        logger.info(f"[METRICS DEBUG] add_tts_characters called with count={count}, current_turn exists: {self.current_turn is not None}")

        if not self.current_turn:
            logger.warning(f" [METRICS DEBUG] Cannot add TTS characters - no current turn")
            return

        if self.current_turn.tts_metrics:
            tts = self.current_turn.tts_metrics[-1]
            if tts.tts_characters is not None:
                tts.tts_characters += count
                logger.info(f"[METRICS DEBUG] Added {count} chars, total now: {tts.tts_characters}")
            else:
                tts.tts_characters = count
                logger.info(f"[METRICS DEBUG] Initialized tts_characters to {count}")
        else:
            logger.warning(f" [METRICS DEBUG] Cannot add TTS characters - no tts_metrics")

    def on_vad_end_of_speech(self, timestamp: float = None) -> None:
        """Called when VAD detects end of speech."""
        if not self.current_turn:
            return

        if timestamp is None:
            timestamp = time.perf_counter()

        if self.current_turn.vad_metrics:
            vad = self.current_turn.vad_metrics[-1]
            vad.vad_end_of_speech_time = timestamp

    def on_interrupted(self) -> None:
        """Called when the user interrupts the agent."""
        if not self.current_turn:
            return

        # NEW LOGIC: Check if LLM generation has started using any LLM metric
        llm_started = False
        if self.current_turn.llm_metrics:
            if any(m.llm_start_time is not None for m in self.current_turn.llm_metrics):
                llm_started = True
            # Also check if TTS started (just in case LLM is skipped or fast)
            if not llm_started and self.current_turn.tts_metrics:
                 if any(m.tts_start_time is not None for m in self.current_turn.tts_metrics):
                     llm_started = True

        if not llm_started:
            logger.info(f"[METRICS DEBUG] Ignoring interruption for turn {self.current_turn.turn_id} - LLM/TTS has not started yet. Treating as continuous input.")
            return

        # Mark turn as interrupted
        self.current_turn.is_interrupted = True

        # Initialize interruption_metrics if not exists
        if not self.current_turn.interruption_metrics:
            self.current_turn.interruption_metrics = InterruptionMetrics()

        # Set interrupt start time
        self.current_turn.interruption_metrics.interrupt_start_time = time.perf_counter()

        # Populate interrupt config if not already set
        self._populate_interrupt_config(self.current_turn.interruption_metrics)

        logger.info(f"Turn {self.current_turn.turn_id} interrupted")

    def on_interrupt_trigger(
        self,
        word_count: int = None,
        duration: float = None
    ) -> None:
        """
        Called when interrupt is triggered, captures interrupt reason.

        Args:
            word_count: Number of words in interrupt transcript (STT-based)
            duration: Duration of speech that triggered interrupt (VAD-based)
        """
        if not self.current_turn or not self.current_turn.is_interrupted:
            return

        if not self.current_turn.interruption_metrics:
            self.current_turn.interruption_metrics = InterruptionMetrics()

        metrics = self.current_turn.interruption_metrics

        # Track word count and duration
        if word_count is not None:
            metrics.interrupt_words = word_count
            if "STT" not in metrics.interrupt_reason:
                metrics.interrupt_reason.append("STT")

        if duration is not None:
            metrics.interrupt_duration = duration
            if "VAD" not in metrics.interrupt_reason:
                metrics.interrupt_reason.append("VAD")

    def set_interrupt_config(
        self,
        mode: str,
        min_duration: float = None,
        min_words: int = None,
        false_interrupt_pause_duration: float = None,
        resume_on_false_interrupt: bool = None
    ) -> None:
        """Configure interrupt detection parameters."""
        if not self.current_turn:
            return

        if not self.current_turn.interruption_metrics:
            self.current_turn.interruption_metrics = InterruptionMetrics()

        metrics = self.current_turn.interruption_metrics
        metrics.interrupt_mode = mode
        metrics.interrupt_min_duration = min_duration
        metrics.interrupt_min_words = min_words
        metrics.false_interrupt_pause_duration = false_interrupt_pause_duration
        metrics.resume_on_false_interrupt = resume_on_false_interrupt

    def on_false_interrupt_start(self, duration: float) -> None:
        """Called when false interrupt timer starts (potential resume scenario)."""
        if not self.current_turn:
            return

        if not self.current_turn.interruption_metrics:
            self.current_turn.interruption_metrics = InterruptionMetrics()

        metrics = self.current_turn.interruption_metrics
        metrics.false_interrupt_start_time = time.perf_counter()

        logger.info(f"False interrupt started - waiting {duration}s to determine if real interrupt")

    def on_false_interrupt_resume(self) -> None:
        """Called when TTS resumes after false interrupt timeout (user didn't continue speaking)."""
        if not self.current_turn or not self.current_turn.interruption_metrics:
            return

        metrics = self.current_turn.interruption_metrics
        metrics.is_false_interrupt = True
        metrics.false_interrupt_end_time = time.perf_counter()

        # Calculate false interrupt duration
        if metrics.false_interrupt_start_time:
            duration_seconds = metrics.false_interrupt_end_time - metrics.false_interrupt_start_time
            metrics.false_interrupt_duration = self._to_milliseconds(duration_seconds)

        metrics.resumed_after_false_interrupt = True

        # Reset interrupted flag since this was NOT a true interrupt
        self.current_turn.is_interrupted = False
        metrics.interrupt_start_time = None
        metrics.interrupt_end_time = None
        metrics.interrupt_words = None
        metrics.interrupt_duration = None
        metrics.interrupt_reason = []

        logger.info(f"False interrupt ended - TTS resumed after {metrics.false_interrupt_duration}ms")

    def on_false_interrupt_escalated(self, word_count: int = None) -> None:
        """Called when a false interrupt escalates to a true interrupt (user continued speaking)."""
        if not self.current_turn or not self.current_turn.interruption_metrics:
            return

        metrics = self.current_turn.interruption_metrics
        metrics.is_false_interrupt = True
        metrics.false_interrupt_end_time = time.perf_counter()

        # Calculate false interrupt duration
        if metrics.false_interrupt_start_time:
            duration_seconds = metrics.false_interrupt_end_time - metrics.false_interrupt_start_time
            metrics.false_interrupt_duration = self._to_milliseconds(duration_seconds)

        metrics.resumed_after_false_interrupt = False

        if word_count is not None:
            metrics.false_interrupt_words = word_count

        logger.info(f"False interrupt escalated to true interrupt after {metrics.false_interrupt_duration}ms")

    def add_timeline_event(
        self,
        event_type: str,
        start_time: float,
        end_time: float = None,
        text: str = ""
    ) -> None:
        """Add a timeline event to current turn."""
        if not self.current_turn:
            return

        event = TimelineEvent(
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            text=text
        )

        # Calculate duration if end_time provided
        if end_time:
            event.duration_ms = round((end_time - start_time) * 1000, 4)

        self.current_turn.timeline_event_metrics.append(event)

    def add_function_tool_call(
        self,
        tool_name: str,
        params: Dict[str, Any],
        response: Dict[str, Any],
        start_time: float,
        end_time: float
    ) -> None:
        """Add function tool call metrics to current turn."""
        logger.info(f"[METRICS DEBUG] add_function_tool_call called - tool_name: {tool_name}, current_turn exists: {self.current_turn is not None}")

        if not self.current_turn:
            logger.warning(f" [METRICS DEBUG] Cannot add function tool call - no current turn")
            return

        tool_metric = FunctionToolMetrics(
            tool_name=tool_name,
            tool_params=params,
            tool_response=response,
            start_time=start_time,
            end_time=end_time,
            latency=end_time - start_time
        )

        self.current_turn.function_tool_metrics.append(tool_metric)
        self.current_turn.function_tools_called.append(tool_name)
        logger.info(f"[METRICS DEBUG] Added function tool call: {tool_name}, total tools: {len(self.current_turn.function_tools_called)}")

    def add_error(self, error: Dict[str, Any]) -> None:
        """Add error to current turn."""
        if self.current_turn:
            self.current_turn.errors.append(error)
