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
from typing import TYPE_CHECKING, Optional, List, Any
import time
import logging
from dataclasses import asdict

from .metrics_schema import SessionMetrics, TurnMetrics
from .component_manager import ComponentMetricsManager
from .turn_lifecycle_tracker import TurnLifecycleTracker
from .event_bridge import EventBridge
from .metrics_transformer import transform_session
if TYPE_CHECKING:
    from ..agent import Agent
    from ..pipeline import Pipeline

logger = logging.getLogger(__name__)


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
        from . import component_metrics_manager, turn_lifecycle_tracker
        self.component_manager = component_metrics_manager
        self.turn_tracker = turn_lifecycle_tracker
        
        # Configure the global tracker with this session's pipeline
        self.turn_tracker.set_pipeline_config(pipeline)
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
                
                # Check for RealtimeLLMAdapter and inject tracker into inner model
                target_model = llm
                if hasattr(llm, 'realtime_model'):
                    target_model = llm.realtime_model
                
                if hasattr(target_model, 'set_metrics_collector'):
                    target_model.set_metrics_collector(self.turn_tracker)
                    logger.info(f"Injected shared TurnLifecycleTracker into {target_model.__class__.__name__} (via llm)")

                # If this is a realtime model (adapter or direct), also register it as 'realtime' component
                # This ensures start_turn initializes realtime_metrics, which is required for _validate_turn()
                if hasattr(llm, 'realtime_model') or hasattr(llm, 'set_metrics_collector'):
                    self.component_manager.register_component('realtime', provider_class, model_name)
                    self.session_metrics.provider_per_component['realtime'] = {
                        'provider_class': provider_class,
                        'model_name': model_name
                    }
                    logger.info(f"Registered {provider_class} as 'realtime' component (derived from llm)")

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

            # Inject shared tracker into Realtime component
            if hasattr(realtime, 'set_metrics_collector'):
                realtime.set_metrics_collector(self.turn_tracker)
                logger.info(f"Injected shared TurnLifecycleTracker into {provider_class}")

        # Update session metrics components list
        self.session_metrics.components = list(
            self.component_manager.get_active_component_types()
        )

        logger.info(f"Discovered components: {self.session_metrics.components}")
        logger.info(f"Provider info: {self.session_metrics.provider_per_component}")

    async def _send_session_analytics(self) -> None:
        """Send session-level analytics."""
        try:
            # Convert session + turns to API schema format
            session_dict = transform_session(
                self.session_metrics,
                self.turn_tracker.completed_turns,
                meeting_id=self.session_metrics.room_id
            )

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
