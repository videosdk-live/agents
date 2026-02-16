from __future__ import annotations
from typing import Dict, List, Any, Set, Optional
import uuid
import logging

from .metrics_schema import (
    VadMetrics,
    SttMetrics,
    EouMetrics,
    LlmMetrics,
    TtsMetrics,
    RealtimeMetrics,
)

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
