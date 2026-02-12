"""
Unified metrics collector for the modular component-based agent framework.

Supports all component combinations:
- LLM only, TTS only, STT only
- STT + LLM, LLM + TTS, STT + LLM + TTS
- Realtime only, Realtime + STT, Realtime + TTS
- Hybrid combinations

Architecture:
- UnifiedMetricsCollector: Main entry point, auto-discovers components
- ComponentMetricsManager: Registers and tracks component instances
- TurnLifecycleTracker: Manages turn boundaries and E2E calculations
- EventBridge: Integrates with PipelineHooks for event-driven collection
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, List, Any, Set
import time
import uuid
import logging
import asyncio
from dataclasses import asdict

from .metrics_schema import (
    SessionMetrics,
    TurnMetrics,
    VadMetrics,
    SttMetrics,
    EouMetrics,
    LlmMetrics,
    TtsMetrics,
    RealtimeMetrics,
    InterruptionMetrics,
    TimelineEvent,
    FallbackEvent,
    FunctionToolMetrics,
    McpToolMetrics,
    ParticipantMetrics,
)

if TYPE_CHECKING:
    from ..agent import Agent
    from ..pipeline import Pipeline
    from ..pipeline_hooks import PipelineHooks

logger = logging.getLogger(__name__)


class ComponentMetricsManager:
    """
    Manages component registration and metric instance creation.
    Tracks which components are active in the session.
    """

    def __init__(self):
        # Component type -> list of (id, provider_class, model_name) tuples
        self._registered_components: Dict[str, List[tuple]] = {}

        # Active component types
        self._active_types: Set[str] = set()

    def register_component(
        self,
        component_type: str,
        provider_class: str,
        model_name: str
    ) -> str:
        """
        Register a component at session start.

        Args:
            component_type: One of 'vad', 'stt', 'eou', 'llm', 'tts', 'realtime'
            provider_class: Provider class name (e.g., 'DeepgramSTT', 'OpenAILLM')
            model_name: Model identifier

        Returns:
            Unique component ID
        """
        component_id = f"{component_type}_{str(uuid.uuid4())[:8]}"

        if component_type not in self._registered_components:
            self._registered_components[component_type] = []

        self._registered_components[component_type].append(
            (component_id, provider_class, model_name)
        )
        self._active_types.add(component_type)

        logger.info(
            f"Registered component: {component_type} "
            f"(provider={provider_class}, model={model_name}, id={component_id})"
        )

        return component_id

    def create_turn_metrics(self, component_type: str) -> Optional[Any]:
        """
        Create a new metrics instance for this component type in a turn.

        Args:
            component_type: Component type to create metrics for

        Returns:
            New metrics instance (VadMetrics, SttMetrics, etc.) or None
        """
        if component_type not in self._registered_components:
            return None

        # Get the most recent registration (handles provider fallbacks)
        component_id, provider_class, model_name = self._registered_components[component_type][-1]

        # Create appropriate metrics instance
        metrics_map = {
            'vad': VadMetrics,
            'stt': SttMetrics,
            'eou': EouMetrics,
            'llm': LlmMetrics,
            'tts': TtsMetrics,
            'realtime': RealtimeMetrics,
        }

        metrics_class = metrics_map.get(component_type)
        if not metrics_class:
            logger.warning(f"Unknown component type: {component_type}")
            return None

        return metrics_class(
            id=component_id,
            provider_class=provider_class,
            model_name=model_name
        )

    def get_active_component_types(self) -> Set[str]:
        """Return set of active component types."""
        return self._active_types.copy()

    def is_component_active(self, component_type: str) -> bool:
        """Check if a component type is active."""
        return component_type in self._active_types


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
            metrics = self.component_manager.create_turn_metrics(component_type)
            if metrics:
                logger.info(f"[METRICS DEBUG] start_turn: Created {component_type} metrics: {type(metrics).__name__}")
                # Add to appropriate list in TurnMetrics
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

        self.current_turn = turn

        # Apply buffered STT start time if it exists
        if self._buffered_stt_start_time is not None and turn.stt_metrics:
            turn.stt_metrics[-1].stt_start_time = self._buffered_stt_start_time
            logger.info(f"[METRICS DEBUG] Applied buffered stt_start_time = {self._buffered_stt_start_time}")
            self._buffered_stt_start_time = None  # Clear the buffer

        logger.info(f"Started turn: {turn_id} (trigger={trigger})")
        logger.info(f"[METRICS DEBUG] start_turn: Created turn with STT metrics: {len(turn.stt_metrics)}, EOU: {len(turn.eou_metrics)}, TTS: {len(turn.tts_metrics)}")

        return turn

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

        # STT latency
        if turn.stt_metrics:
            for stt in turn.stt_metrics:
                if stt.stt_latency:
                    total_ms += self._to_milliseconds(stt.stt_latency)

        # EOU latency
        if turn.eou_metrics:
            for eou in turn.eou_metrics:
                if eou.eou_latency:
                    total_ms += self._to_milliseconds(eou.eou_latency)

        # LLM TTFT (time to first token)
        if turn.llm_metrics:
            for llm in turn.llm_metrics:
                if llm.llm_ttft:
                    total_ms += self._to_milliseconds(llm.llm_ttft)

        # TTS latency
        if turn.tts_metrics:
            for tts in turn.tts_metrics:
                if tts.tts_latency:
                    total_ms += self._to_milliseconds(tts.tts_latency)

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
            self.current_turn.user_speech_start_time = timestamp
            if transcript:
                self.current_turn.user_speech = transcript

    def on_user_speech_end(self, timestamp: float) -> None:
        """Called when user stops speaking."""
        if self.current_turn:
            self.current_turn.user_speech_end_time = timestamp
            if self.current_turn.user_speech_start_time:
                duration = timestamp - self.current_turn.user_speech_start_time
                self.current_turn.user_speech_duration = round(duration, 4)

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
            logger.info(f"[METRICS DEBUG] stt_metrics list: {len(self.current_turn.stt_metrics) if self.current_turn.stt_metrics else 0} items")

        if self.current_turn and self.current_turn.stt_metrics:
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

        if self.current_turn and self.current_turn.stt_metrics:
            stt = self.current_turn.stt_metrics[-1]
            stt.stt_end_time = timestamp
            stt.stt_transcript = transcript
            logger.info(f"[METRICS DEBUG] Set stt_end_time = {timestamp}")

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


class EventBridge:
    """
    Bridges component events to metrics collection.
    Registers callbacks for pipeline, orchestrator, and component events.
    """

    def __init__(
        self,
        hooks: PipelineHooks,
        turn_tracker: TurnLifecycleTracker,
        session_metrics: SessionMetrics,
        pipeline: Any = None,
        analytics_client: Any = None,
        playground_manager: Any = None
    ):
        self.hooks = hooks
        self.turn_tracker = turn_tracker
        self.session_metrics = session_metrics
        self.pipeline = pipeline
        self.analytics_client = analytics_client
        self.playground_manager = playground_manager
        self._attached = False
        self._orchestrator_attached = False
        self._component_handlers = []

    def attach(self) -> None:
        """Attach event handlers to pipeline hooks and components."""
        if self._attached:
            logger.warning("EventBridge already attached")
            return

        # Register basic lifecycle hooks
        @self.hooks.on("user_turn_start")
        async def on_user_turn_start(transcript: str) -> None:
            timestamp = time.perf_counter()
            self.turn_tracker.on_user_speech_start(timestamp, transcript)

        @self.hooks.on("user_turn_end")
        async def on_user_turn_end() -> None:
            timestamp = time.perf_counter()
            self.turn_tracker.on_user_speech_end(timestamp)

        @self.hooks.on("agent_turn_start")
        async def on_agent_turn_start() -> None:
            timestamp = time.perf_counter()
            self.turn_tracker.on_agent_speech_start(timestamp)

        @self.hooks.on("agent_turn_end")
        async def on_agent_turn_end() -> None:
            timestamp = time.perf_counter()
            self.turn_tracker.on_agent_speech_end(timestamp)
            # Complete turn when agent finishes speaking
            completed_turn = self.turn_tracker.complete_turn()
            if completed_turn:
                logger.info(f"Turn completed: {completed_turn.turn_id}")
                await self._send_turn_analytics(completed_turn)

        @self.hooks.on("content_generated")
        async def on_content_generated(data: dict) -> None:
            text = data.get("text", "")
            if text and self.turn_tracker.current_turn:
                self.turn_tracker.current_turn.agent_speech = text

        self._attached = True
        logger.info("EventBridge attached to PipelineHooks")

        # Attach to orchestrator and component events
        if self.pipeline:
            self._attach_orchestrator_events()
            self._attach_component_events()

    def _attach_orchestrator_events(self) -> None:
        """Attach to PipelineOrchestrator events for detailed metrics."""
        if not self.pipeline or not hasattr(self.pipeline, 'orchestrator'):
            return

        orchestrator = self.pipeline.orchestrator
        if not orchestrator:
            return

        try:
            # Transcript ready event (STT complete)
            def on_transcript_ready(data: dict):
                timestamp = time.perf_counter()
                text = data.get("text", "")
                is_final = data.get("is_final", False)
                is_preemptive = data.get("is_preemptive", False)
                metadata = data.get("metadata", {})

                if is_final and text:
                    # Extract STT-specific data
                    confidence = metadata.get("confidence")
                    duration = metadata.get("duration")

                    self.turn_tracker.on_stt_complete(
                        timestamp,
                        text,
                        confidence=confidence,
                        duration=duration,
                        metadata=metadata,
                        is_preemptive=is_preemptive
                    )
                    self.turn_tracker.add_timeline_event(
                        "transcript_ready",
                        timestamp,
                        text=text
                    )

            # Content generated event (LLM complete)
            def on_content_generated(data: dict):
                timestamp = time.perf_counter()
                text = data.get("text", "")
                if text:
                    self.turn_tracker.add_timeline_event(
                        "content_generated",
                        timestamp,
                        text=text[:100]  # First 100 chars
                    )

            # Synthesis complete event (TTS complete)
            def on_synthesis_complete(data: dict):
                timestamp = time.perf_counter()
                self.turn_tracker.on_tts_complete(timestamp)
                self.turn_tracker.add_timeline_event(
                    "synthesis_complete",
                    timestamp
                )

            orchestrator.on("transcript_ready", on_transcript_ready)
            orchestrator.on("content_generated", on_content_generated)
            orchestrator.on("synthesis_complete", on_synthesis_complete)

            self._orchestrator_attached = True
            logger.info("EventBridge attached to PipelineOrchestrator events")

        except Exception as e:
            logger.error(f"Error attaching orchestrator events: {e}", exc_info=True)

    def _attach_component_events(self) -> None:
        """Attach to individual component events for granular metrics."""
        if not self.pipeline:
            return

        try:
            # Attach to ContentGeneration events (LLM timing, token usage)
            if hasattr(self.pipeline, 'orchestrator') and self.pipeline.orchestrator:
                orch = self.pipeline.orchestrator

                if hasattr(orch, 'content_generation') and orch.content_generation:
                    content_gen = orch.content_generation

                    def on_generation_started(data: dict):
                        timestamp = time.perf_counter()
                        user_text = data.get("user_text", "")
                        self.turn_tracker.on_llm_start(timestamp, user_text)

                    def on_first_chunk(data: dict):
                        timestamp = time.perf_counter()
                        self.turn_tracker.on_llm_first_token(timestamp)

                    def on_generation_complete(data: dict):
                        timestamp = time.perf_counter()
                        # Token usage will be captured from usage_tracked event
                        self.turn_tracker.on_llm_complete(timestamp)

                    def on_usage_tracked(usage: dict):
                        # Capture token usage from LLM
                        if self.turn_tracker.current_turn and self.turn_tracker.current_turn.llm_metrics:
                            llm = self.turn_tracker.current_turn.llm_metrics[-1]
                            llm.prompt_tokens = usage.get("prompt_tokens")
                            llm.completion_tokens = usage.get("completion_tokens")
                            llm.total_tokens = usage.get("total_tokens")
                            llm.prompt_cached_tokens = usage.get("prompt_cached_tokens")

                            # Calculate tokens per second
                            if llm.llm_duration and llm.completion_tokens:
                                llm.tokens_per_second = round(
                                    llm.completion_tokens / llm.llm_duration, 2
                                )

                    content_gen.on("generation_started", on_generation_started)
                    content_gen.on("first_chunk", on_first_chunk)
                    content_gen.on("generation_complete", on_generation_complete)
                    content_gen.on("usage_tracked", on_usage_tracked)

                    logger.info("Attached to ContentGeneration events")

                # Attach to SpeechUnderstanding events (STT timing)
                if hasattr(orch, 'speech_understanding') and orch.speech_understanding:
                    speech_understanding = orch.speech_understanding

                    def on_speech_started(data: dict):
                        timestamp = time.perf_counter()
                        self.turn_tracker.on_user_speech_start(timestamp)

                    def on_speech_stopped(data: dict):
                        timestamp = time.perf_counter()
                        self.turn_tracker.on_user_speech_end(timestamp)

                    def on_transcript_preflight(data: dict):
                        """Handle preflight transcript for preemptive generation."""
                        timestamp = time.perf_counter()
                        text = data.get("text", "")
                        self.turn_tracker.on_stt_preflight_end(timestamp, text)

                    def on_transcript_interim(data: dict):
                        """Handle interim transcript."""
                        timestamp = time.perf_counter()
                        self.turn_tracker.on_stt_interim_end(timestamp)

                    def on_transcript_final(data: dict):
                        timestamp = time.perf_counter()
                        text = data.get("text", "")
                        is_preemptive = data.get("is_preemptive", False)
                        metadata = data.get("metadata", {})

                        # Extract STT-specific data if available
                        confidence = metadata.get("confidence")
                        duration = metadata.get("duration")

                        self.turn_tracker.on_stt_complete(
                            timestamp,
                            text,
                            confidence=confidence,
                            duration=duration,
                            metadata=metadata,
                            is_preemptive=is_preemptive
                        )

                    speech_understanding.on("speech_started", on_speech_started)
                    speech_understanding.on("speech_stopped", on_speech_stopped)
                    speech_understanding.on("transcript_preflight", on_transcript_preflight)
                    speech_understanding.on("transcript_interim", on_transcript_interim)
                    speech_understanding.on("transcript_final", on_transcript_final)

                    logger.info("Attached to SpeechUnderstanding events")

                # Attach to SpeechGeneration events (TTS timing)
                if hasattr(orch, 'speech_generation') and orch.speech_generation:
                    speech_gen = orch.speech_generation

                    def on_synthesis_started(data: dict):
                        timestamp = time.perf_counter()
                        self.turn_tracker.on_tts_start(timestamp)

                    def on_first_audio_byte(data: dict):
                        timestamp = time.perf_counter()
                        self.turn_tracker.on_tts_first_byte(timestamp)

                    def on_last_audio_byte(data: dict):
                        timestamp = time.perf_counter()
                        self.turn_tracker.on_tts_complete(timestamp)

                    speech_gen.on("synthesis_started", on_synthesis_started)
                    speech_gen.on("first_audio_byte", on_first_audio_byte)
                    speech_gen.on("last_audio_byte", on_last_audio_byte)

                    logger.info("Attached to SpeechGeneration events")

            # Attach to Realtime LLM events
            if hasattr(self.pipeline, 'llm') and self.pipeline.llm:
                llm = self.pipeline.llm

                # Check if it's a RealtimeLLMAdapter
                if hasattr(llm, '_is_realtime') and llm._is_realtime:
                    def on_user_speech_started_rt(data: dict):
                        timestamp = time.perf_counter()
                        self.turn_tracker.on_user_speech_start(timestamp)

                    def on_user_speech_ended_rt(data: dict):
                        timestamp = time.perf_counter()
                        self.turn_tracker.on_user_speech_end(timestamp)

                    def on_agent_speech_started_rt(data: dict):
                        timestamp = time.perf_counter()
                        self.turn_tracker.on_agent_speech_start(timestamp)

                    def on_agent_speech_ended_rt(data: dict):
                        timestamp = time.perf_counter()
                        self.turn_tracker.on_agent_speech_end(timestamp)
                        # Complete turn
                        completed_turn = self.turn_tracker.complete_turn()
                        if completed_turn:
                            logger.info(f"Realtime turn completed: {completed_turn.turn_id}")
                            asyncio.create_task(self._send_turn_analytics(completed_turn))

                    llm.on("user_speech_started", on_user_speech_started_rt)
                    llm.on("user_speech_ended", on_user_speech_ended_rt)
                    llm.on("agent_speech_started", on_agent_speech_started_rt)
                    llm.on("agent_speech_ended", on_agent_speech_ended_rt)

                    logger.info("Attached to RealtimeLLMAdapter events")

        except Exception as e:
            logger.error(f"Error attaching component events: {e}", exc_info=True)

    async def _send_turn_analytics(self, turn: TurnMetrics) -> None:
        """Send turn analytics to analytics client and playground."""
        try:
            # Convert to dict
            turn_dict = asdict(turn)

            # Send to analytics client if available
            if self.analytics_client and hasattr(self.analytics_client, 'send_interaction_analytics'):
                try:
                    # Session ID is already set on the client, just pass the turn data
                    await self.analytics_client.send_interaction_analytics(turn_dict)
                    logger.debug(f"Turn analytics sent: {turn.turn_id}")
                except Exception as e:
                    logger.error(f"Error sending turn analytics: {e}")

            # Send to playground if available
            if self.playground_manager and hasattr(self.playground_manager, 'send_metrics'):
                try:
                    await self.playground_manager.send_metrics(turn_dict)
                    logger.debug(f"Turn metrics sent to playground: {turn.turn_id}")
                except Exception as e:
                    logger.error(f"Error sending to playground: {e}")

        except Exception as e:
            logger.error(f"Error in _send_turn_analytics: {e}", exc_info=True)

    def detach(self) -> None:
        """Detach event handlers (cleanup)."""
        self._attached = False
        self._orchestrator_attached = False
        logger.info("EventBridge detached")


class UnifiedMetricsCollector:
    """
    Main metrics collector for the modular component-based agent framework.

    Auto-discovers active components from the pipeline and dynamically adapts
    metrics collection to the specific component configuration.

    Usage:
        collector = UnifiedMetricsCollector(
            session_id=session_id,
            agent=agent,
            pipeline=pipeline
        )
        await collector.start()

        # ... agent runs ...

        await collector.cleanup()
    """

    def __init__(
        self,
        session_id: str,
        agent: Agent,
        pipeline: Pipeline,
        analytics_client: Any = None,
        traces_flow_manager: Any = None,
        playground_manager: Any = None
    ):
        self.session_id = session_id
        self.agent = agent
        self.pipeline = pipeline
        self._analytics_client = analytics_client
        self.traces_flow_manager = traces_flow_manager
        self._playground_manager = playground_manager

        # Initialize session metrics
        self.session_metrics = SessionMetrics(
            session_id=session_id,
            session_start_time=time.time()
        )

        # Initialize core components
        self.component_manager = ComponentMetricsManager()
        self.turn_tracker = TurnLifecycleTracker(self.component_manager, pipeline)
        self.event_bridge = EventBridge(
            hooks=pipeline.hooks,
            turn_tracker=self.turn_tracker,
            session_metrics=self.session_metrics,
            pipeline=pipeline,  # Pass pipeline for component event access
            analytics_client=analytics_client,
            playground_manager=playground_manager
        )

        logger.info(f"UnifiedMetricsCollector initialized for session: {session_id}")

    @property
    def analytics_client(self) -> Any:
        """Get analytics client."""
        return self._analytics_client

    @analytics_client.setter
    def analytics_client(self, value: Any) -> None:
        """Set analytics client on both collector and event bridge."""
        self._analytics_client = value
        if hasattr(self, 'event_bridge'):
            self.event_bridge.analytics_client = value

    @property
    def playground_manager(self) -> Any:
        """Get playground manager."""
        return self._playground_manager

    @playground_manager.setter
    def playground_manager(self, value: Any) -> None:
        """Set playground manager on both collector and event bridge."""
        self._playground_manager = value
        if hasattr(self, 'event_bridge'):
            self.event_bridge.playground_manager = value

    async def start(self) -> None:
        """
        Start metrics collection.
        Auto-discovers components from pipeline and attaches event bridge.
        """
        # Discover and register components from pipeline
        self._discover_components()

        # Attach event bridge
        self.event_bridge.attach()

        logger.info(
            f"Metrics collection started "
            f"(components={list(self.component_manager.get_active_component_types())})"
        )

    async def cleanup(self) -> None:
        """
        Cleanup metrics collection.
        Finalizes session, completes any open turn, and sends analytics.
        """
        # Complete any open turn
        if self.turn_tracker.current_turn:
            self.turn_tracker.complete_turn()

        # Finalize session
        self.session_metrics.session_end_time = time.time()

        # Detach event bridge
        self.event_bridge.detach()

        # Send analytics if client provided
        if self._analytics_client:
            await self._send_session_analytics()

        logger.info(
            f"Metrics collection completed "
            f"(turns={len(self.turn_tracker.completed_turns)})"
        )

    def _discover_components(self) -> None:
        """
        Auto-discover active components from pipeline.
        Registers each discovered component with the component manager.
        Populates session-level provider information.

        Components can be in two places:
        1. Directly on pipeline (old architecture): pipeline.stt, pipeline.llm, pipeline.tts
        2. Inside orchestrator (new architecture): pipeline.orchestrator.speech_understanding.stt, etc.
        """
        # Try orchestrator-based pipeline first (new architecture)
        orchestrator = getattr(self.pipeline, 'orchestrator', None)

        if orchestrator:
            # Check for STT (inside orchestrator.speech_understanding)
            speech_understanding = getattr(orchestrator, 'speech_understanding', None)
            if speech_understanding:
                stt = getattr(speech_understanding, 'stt', None)
                if stt:
                    provider_class = stt.__class__.__name__
                    model_name = getattr(stt, 'model', 'unknown')
                    self.component_manager.register_component('stt', provider_class, model_name)
                    self.session_metrics.provider_per_component['stt'] = {
                        'provider_class': provider_class,
                        'model_name': model_name
                    }

                # Check for VAD
                vad = getattr(speech_understanding, 'vad', None)
                if vad:
                    provider_class = vad.__class__.__name__
                    self.component_manager.register_component('vad', provider_class, 'n/a')
                    self.session_metrics.provider_per_component['vad'] = {
                        'provider_class': provider_class,
                        'model_name': 'n/a'
                    }

                # Check for EOU
                eou = getattr(speech_understanding, 'turn_detector', None) or getattr(speech_understanding, 'eou', None)
                if eou:
                    provider_class = eou.__class__.__name__
                    model_name = getattr(eou, 'model', 'unknown')
                    self.component_manager.register_component('eou', provider_class, model_name)
                    self.session_metrics.provider_per_component['eou'] = {
                        'provider_class': provider_class,
                        'model_name': model_name
                    }

            # Check for LLM (inside orchestrator.content_generation)
            content_generation = getattr(orchestrator, 'content_generation', None)
            if content_generation:
                llm = getattr(content_generation, 'llm', None)
                if llm:
                    provider_class = llm.__class__.__name__
                    model_name = getattr(llm, 'model', 'unknown')
                    self.component_manager.register_component('llm', provider_class, model_name)
                    self.session_metrics.provider_per_component['llm'] = {
                        'provider_class': provider_class,
                        'model_name': model_name
                    }

            # Check for TTS (inside orchestrator.speech_generation)
            speech_generation = getattr(orchestrator, 'speech_generation', None)
            if speech_generation:
                tts = getattr(speech_generation, 'tts', None)
                if tts:
                    provider_class = tts.__class__.__name__
                    model_name = getattr(tts, 'model', 'unknown')
                    self.component_manager.register_component('tts', provider_class, model_name)
                    self.session_metrics.provider_per_component['tts'] = {
                        'provider_class': provider_class,
                        'model_name': model_name
                    }

        # Fallback: Check direct pipeline attributes (old architecture)
        else:
            # Check for STT directly on pipeline
            stt = getattr(self.pipeline, 'stt', None)
            if stt:
                provider_class = stt.__class__.__name__
                model_name = getattr(stt, 'model', 'unknown')
                self.component_manager.register_component('stt', provider_class, model_name)
                self.session_metrics.provider_per_component['stt'] = {
                    'provider_class': provider_class,
                    'model_name': model_name
                }

            # Check for LLM directly on pipeline
            llm = getattr(self.pipeline, 'llm', None)
            if llm:
                provider_class = llm.__class__.__name__
                model_name = getattr(llm, 'model', 'unknown')
                self.component_manager.register_component('llm', provider_class, model_name)
                self.session_metrics.provider_per_component['llm'] = {
                    'provider_class': provider_class,
                    'model_name': model_name
                }

            # Check for TTS directly on pipeline
            tts = getattr(self.pipeline, 'tts', None)
            if tts:
                provider_class = tts.__class__.__name__
                model_name = getattr(tts, 'model', 'unknown')
                self.component_manager.register_component('tts', provider_class, model_name)
                self.session_metrics.provider_per_component['tts'] = {
                    'provider_class': provider_class,
                    'model_name': model_name
                }

        # Check for Realtime (can be direct on pipeline)
        realtime = getattr(self.pipeline, 'realtime_llm', None)
        if realtime:
            provider_class = realtime.__class__.__name__
            model_name = getattr(realtime, 'model', 'unknown')
            self.component_manager.register_component('realtime', provider_class, model_name)
            self.session_metrics.provider_per_component['realtime'] = {
                'provider_class': provider_class,
                'model_name': model_name
            }

        # Update session metrics components list
        self.session_metrics.components = list(
            self.component_manager.get_active_component_types()
        )

        logger.info(f"Discovered components: {self.session_metrics.components}")
        logger.info(f"Provider info: {self.session_metrics.provider_per_component}")

    async def _send_session_analytics(self) -> None:
        """Send session-level analytics."""
        try:
            # Convert session to dict
            session_dict = asdict(self.session_metrics)

            # Add completed turns
            session_dict['turns'] = [
                asdict(turn) for turn in self.turn_tracker.completed_turns
            ]

            # Send to analytics client
            if self._analytics_client and hasattr(self._analytics_client, 'send_session_analytics'):
                await self._analytics_client.send_session_analytics(session_dict)

            # Send to playground if enabled
            if self._playground_manager and hasattr(self._playground_manager, 'send_session_metrics'):
                await self._playground_manager.send_session_metrics(session_dict)

            logger.info(f"Session analytics sent for session: {self.session_id}")

        except Exception as e:
            logger.error(f"Error sending session analytics: {e}", exc_info=True)

    def get_current_turn(self) -> Optional[TurnMetrics]:
        """Get the current active turn."""
        return self.turn_tracker.current_turn

    def get_completed_turns(self) -> List[TurnMetrics]:
        """Get all completed turns."""
        return self.turn_tracker.completed_turns

    def get_session_metrics(self) -> SessionMetrics:
        """Get session-level metrics."""
        return self.session_metrics
