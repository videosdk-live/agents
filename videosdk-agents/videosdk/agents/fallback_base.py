import logging
import asyncio
import time
from typing import List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics.cascading_metrics_collector import CascadingMetricsCollector

logger = logging.getLogger(__name__)

class FallbackBase:
    """Shared logic for switching providers and cleanup."""
    def __init__(self, providers: List[Any], component_name: str, temporary_disable_sec: float = 60.0, permanent_disable_after_attempts: int = 3):
        if not providers:
            raise ValueError(f"{component_name} requires at least one provider")
        self.providers = providers
        self._current_index = 0
        self._component_name = component_name
        self._switch_lock = asyncio.Lock()
        self.temporary_disable_sec = temporary_disable_sec
        self.permanent_disable_after_attempts = permanent_disable_after_attempts
        self._failed_providers: dict[int, float] = {} 
        self._recovery_attempts: dict[int, int] = {}
        self._metrics_collector: Optional['CascadingMetricsCollector'] = None

    def set_metrics_collector(self, metrics_collector: 'CascadingMetricsCollector'):
        """Set the metrics collector for fallback event tracking"""
        self._metrics_collector = metrics_collector

    @property
    def active_provider(self):
        return self.providers[self._current_index]

    @property
    def active_provider_class(self) -> str:
        """Return the class name of the currently active provider"""
        return self.active_provider.__class__.__name__

    @property
    def label(self) -> str:
        return f"Fallback{self._component_name}(active={self.active_provider.label})"

    def _emit_fallback_event(self, event_data: dict):
        """Emit fallback event to metrics collector if available"""
        if self._metrics_collector:
            # Update provider info when fallback occurs
            if event_data.get("new_provider_label"):
                new_provider_class = self.active_provider_class
                self._metrics_collector.update_provider_class(self._component_name, new_provider_class)
            self._metrics_collector.on_fallback_event(event_data)

    async def _switch_provider(self, reason: str, failed_provider: Any = None):
        """
        Internal: Switch to the next provider in the list.
        Returns True if switched successfully, False if no providers left.
        """
        # Track timing for fallback event
        fallback_start_time = time.perf_counter()
        original_connection_start = fallback_start_time
        
        async with self._switch_lock:
            if failed_provider and failed_provider != self.active_provider:
                logger.info(f"[{self._component_name}] Provider {getattr(failed_provider, 'label', 'Unknown')} already switched. Current: {self.active_provider.label}")
                return True

            original_provider_label = self.active_provider.label
            logger.warning(f"[{self._component_name}] Provider {self.active_provider.label} failed: {reason}")
            
            original_connection_end = time.perf_counter()
            
            try:
                failed_idx = self._current_index
                if self.providers[failed_idx] == self.active_provider:
                     self._failed_providers[failed_idx] = time.time()
                     
                     current_attempts = self._recovery_attempts.get(failed_idx, 0)
                     self._recovery_attempts[failed_idx] = current_attempts + 1
                     
                     logger.warning(f"[{self._component_name}] Provider {failed_idx} failed. Recovery attempt {self._recovery_attempts[failed_idx]}/{self.permanent_disable_after_attempts}")

            except Exception as e:
                logger.warning(f"[{self._component_name}] Error recording failure timestamp: {e}")

            try:
                if hasattr(self.active_provider, "aclose"):
                    await self.active_provider.aclose()
            except Exception as e:
                logger.warning(f"[{self._component_name}] Error closing failed provider: {e}")

            if self._current_index >= len(self.providers) - 1:
                logger.error(f"[{self._component_name}] All providers failed. No fallback available.")
                # Emit fallback event even when no fallback available
                fallback_end_time = time.perf_counter()
                self._emit_fallback_event({
                    "component_type": self._component_name,
                    "temporary_disable_sec": self.temporary_disable_sec,
                    "permanent_disable_after_attempts": self.permanent_disable_after_attempts,
                    "recovery_attempt": self._recovery_attempts.get(self._current_index, 0),
                    "message": f"All providers failed. No fallback available. Last error: {reason}",
                    "start_time": fallback_start_time,
                    "end_time": fallback_end_time,
                    "duration_ms": (fallback_end_time - fallback_start_time) * 1000,
                    "original_provider_label": original_provider_label,
                    "original_connection_start": original_connection_start,
                    "original_connection_end": original_connection_end,
                    "original_connection_duration_ms": (original_connection_end - original_connection_start) * 1000,
                    "new_provider_label": None,
                    "new_connection_start": None,
                    "new_connection_end": None,
                    "new_connection_duration_ms": None,
                })
                return False

            self._current_index += 1
            new_connection_start = time.perf_counter()
            new_provider_label = self.active_provider.label
            logger.info(f"[{self._component_name}] Switched to backup: {new_provider_label}")
            new_connection_end = time.perf_counter()
            
            fallback_end_time = time.perf_counter()
            
            # Emit fallback event with timing data
            self._emit_fallback_event({
                "component_type": self._component_name,
                "temporary_disable_sec": self.temporary_disable_sec,
                "permanent_disable_after_attempts": self.permanent_disable_after_attempts,
                "recovery_attempt": self._recovery_attempts.get(self._current_index - 1, 0),
                "message": f"Provider {original_provider_label} failed: {reason}. Switched to {new_provider_label}",
                "start_time": fallback_start_time,
                "end_time": fallback_end_time,
                "duration_ms": (fallback_end_time - fallback_start_time) * 1000,
                "original_provider_label": original_provider_label,
                "original_connection_start": original_connection_start,
                "original_connection_end": original_connection_end,
                "original_connection_duration_ms": (original_connection_end - original_connection_start) * 1000,
                "new_provider_label": new_provider_label,
                "new_connection_start": new_connection_start,
                "new_connection_end": new_connection_end,
                "new_connection_duration_ms": (new_connection_end - new_connection_start) * 1000,
            })
            
            return True

    def check_recovery(self):
        """
        Checks if any higher-priority providers (lower index than current) 
        have passed their recovery cooldown. If so, switches back to the best one.
        """
        now = time.time()
        best_ready_index = self._current_index
        recovery_message = None
        
        for i in range(self._current_index):
            attempts = self._recovery_attempts.get(i, 0)
            if attempts >= self.permanent_disable_after_attempts:
                continue

            if i in self._failed_providers:
                elapsed = now - self._failed_providers[i]
                if elapsed > self.temporary_disable_sec:
                    recovery_message = f"Provider {i} (Label: {self.providers[i].label}) cooldown expired ({elapsed:.1f}s > {self.temporary_disable_sec}s). Attempting recovery."
                    logger.info(f"[{self._component_name}] {recovery_message}")
                    del self._failed_providers[i]
                    best_ready_index = i
                    break
            else:
                pass
        
        if best_ready_index < self._current_index:
            previous_provider_label = self.active_provider.label
            logger.info(f"[{self._component_name}] Restoring primary/higher priority provider: {self.providers[best_ready_index].label}")
            self._current_index = best_ready_index
            
            # Emit recovery attempt event
            self._emit_fallback_event({
                "component_type": self._component_name,
                "temporary_disable_sec": self.temporary_disable_sec,
                "permanent_disable_after_attempts": self.permanent_disable_after_attempts,
                "recovery_attempt": self._recovery_attempts.get(best_ready_index, 0),
                "message": recovery_message or f"Restoring primary/higher priority provider: {self.providers[best_ready_index].label}",
                "original_provider_label": previous_provider_label,
                "new_provider_label": self.providers[best_ready_index].label,
                "start_time": time.perf_counter(),
                "end_time": time.perf_counter(),
                "duration_ms": 0,
            })
            return True
        return False
