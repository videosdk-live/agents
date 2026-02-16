from __future__ import annotations
from typing import Any, Optional, Dict
import time
import asyncio
import logging
from dataclasses import asdict

from .metrics_schema import SessionMetrics, TurnMetrics, TimelineEvent
from .turn_lifecycle_tracker import TurnLifecycleTracker
from .metrics_transformer import transform_turn

logger = logging.getLogger(__name__)

class EventBridge:
    """
    Bridges component events to metrics collection.
    Registers callbacks for pipeline, orchestrator, and component events.
    """

    def __init__(
        self,
        hooks: Any, # Typed as PipelineHooks in original but avoiding circular import
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
            pass  # Agent speech start is tracked at TTFB (first_audio_byte)

        @self.hooks.on("agent_turn_end")
        async def on_agent_turn_end() -> None:
            timestamp = time.perf_counter()
            if self.turn_tracker.current_turn:
                # Set agent speech end time (at last audio callback)
                if not self.turn_tracker.current_turn.agent_speech_end_time:
                    self.turn_tracker.on_agent_speech_end(timestamp)
                # Add timeline event for agent speech
                if self.turn_tracker.current_turn.agent_speech_start_time:
                    self.turn_tracker.add_timeline_event(
                        "agent_speech",
                        self.turn_tracker.current_turn.agent_speech_start_time,
                        end_time=timestamp,
                        text=self.turn_tracker.current_turn.agent_speech or ""
                    )
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
                pass  # Timeline only tracks user_speech and agent_speech

            # Synthesis complete event (TTS complete)
            def on_synthesis_complete(data: dict):
                timestamp = time.perf_counter()
                self.turn_tracker.on_tts_complete(timestamp)
                # Timeline event is added in agent_turn_end hook with proper start/end times

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
                        # Agent speech starts at TTFB (first audio byte reaches user)
                        self.turn_tracker.on_agent_speech_start(timestamp)

                    def on_last_audio_byte(data: dict):
                        timestamp = time.perf_counter()
                        self.turn_tracker.on_tts_complete(timestamp)
                        # Agent speech end + timeline are handled in agent_turn_end hook
                        # This is a fallback for say()-only turns where agent_turn_end
                        # might not fire
                        if (self.turn_tracker.current_turn and
                            self.turn_tracker.current_turn.user_speech_start_time is None):
                            # Set agent speech end if not already set
                            if not self.turn_tracker.current_turn.agent_speech_end_time:
                                self.turn_tracker.on_agent_speech_end(timestamp)
                            # Add timeline event
                            if self.turn_tracker.current_turn.agent_speech_start_time:
                                self.turn_tracker.add_timeline_event(
                                    "agent_speech",
                                    self.turn_tracker.current_turn.agent_speech_start_time,
                                    end_time=timestamp,
                                    text=self.turn_tracker.current_turn.agent_speech or ""
                                )
                            completed_turn = self.turn_tracker.complete_turn()
                            if completed_turn:
                                logger.info(f"TTS-only turn completed (say): {completed_turn.turn_id}")
                                asyncio.create_task(self._send_turn_analytics(completed_turn))

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
            # Convert to API schema format
            turn_dict = transform_turn(turn)

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
