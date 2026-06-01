import logging
import asyncio
import time
from typing import List, Any, Optional, TYPE_CHECKING

from .metrics import metrics_collector
from .utils import format_provider_class

logger = logging.getLogger(__name__)

class FallbackBase:
    """Base class providing provider failover, recovery, and metrics emission for fallback-capable components."""
    def __init__(
        self,
        providers: List[Any],
        component_name: str,
        temporary_disable_sec: float = 60.0,
        permanent_disable_after_attempts: int = 3,
        latency_threshold_ms: Optional[float] = None,
        consecutive_latency_hits: int = 3,
    ):
        if not providers:
            raise ValueError(f"{component_name} requires at least one provider")
        self.providers = providers
        self._current_index = 0
        self._component_name = component_name
        self._switch_lock = asyncio.Lock()
        self.temporary_disable_sec = temporary_disable_sec
        self.permanent_disable_after_attempts = permanent_disable_after_attempts
        self.latency_threshold_ms = latency_threshold_ms
        self.consecutive_latency_hits = max(1, consecutive_latency_hits)
        self._consecutive_latency_hits: int = 0
        self._last_latency_ms: Optional[float] = None
        self._failed_providers: dict[int, float] = {}
        self._recovery_attempts: dict[int, int] = {}
        self._metrics_collector = metrics_collector

    @property
    def active_provider(self):
        return self.providers[self._current_index]

    @property
    def active_provider_class(self) -> str:
        """Return the display class name of the currently active provider"""
        return format_provider_class(self.active_provider)

    @property
    def label(self) -> str:
        return f"Fallback{self._component_name}(active={self.active_provider.label})"

    def _emit_fallback_event(self, event_data: dict):
        """Emit fallback event to metrics collector if available"""
        print(f"Emitting fallback event: {event_data}")  # Debug print for emitted event data
        if self._metrics_collector:
            # Update provider info when fallback occurs
            if event_data.get("new_provider_label") or event_data.get("is_recovery"):
                new_provider_class = self.active_provider_class
                new_provider_model = getattr(self.active_provider, 'model', getattr(self.active_provider, 'model_id', getattr(self.active_provider, 'speech_model', getattr(self.active_provider, 'voice_id', getattr(self.active_provider, 'voice', '')))))
                self._metrics_collector.update_provider_class(self._component_name, new_provider_class, str(new_provider_model))
            self._metrics_collector.on_fallback_event(event_data)

    async def _record_latency(self, latency_ms: Optional[float]) -> None:
        """
        Record an observed per-turn latency value for the active provider.
        When latency exceeds latency_threshold_ms for consecutive_latency_hits consecutive
        turns, trigger a fallback switch via the same path used for errors.
        """
        if self.latency_threshold_ms is None or latency_ms is None:
            return

        self._last_latency_ms = latency_ms

        if latency_ms <= self.latency_threshold_ms:
            if self._consecutive_latency_hits:
                logger.info(
                    f"[{self._component_name}] Latency recovered: {latency_ms:.1f}ms <= {self.latency_threshold_ms}ms"
                )
            self._consecutive_latency_hits = 0
            return

        self._consecutive_latency_hits += 1
        logger.info(
            f"[{self._component_name}] Latency hit {self._consecutive_latency_hits}/{self.consecutive_latency_hits}: "
            f"{latency_ms:.1f}ms > {self.latency_threshold_ms}ms (provider={self.active_provider.label})"
        )

        if self._consecutive_latency_hits >= self.consecutive_latency_hits:
            reason = (
                f"Latency degraded: {latency_ms:.1f}ms > {self.latency_threshold_ms}ms "
                f"for {self.consecutive_latency_hits} consecutive turns"
            )
            await self._switch_provider(reason, failed_provider=self.active_provider)

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
            self._consecutive_latency_hits = 0
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
                "observed_latency_ms": self._last_latency_ms,
                "latency_threshold_ms": self.latency_threshold_ms,
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
            new_provider_label = self.providers[best_ready_index].label
            logger.info(f"[{self._component_name}] Restoring primary/higher priority provider: {new_provider_label}")
            self._current_index = best_ready_index
            self._consecutive_latency_hits = 0
            
            # Emit recovery event - mark as recovery so it doesn't create child spans
            self._emit_fallback_event({
                "component_type": self._component_name,
                "temporary_disable_sec": self.temporary_disable_sec,
                "permanent_disable_after_attempts": self.permanent_disable_after_attempts,
                "recovery_attempt": self._recovery_attempts.get(best_ready_index, 0),
                "message": recovery_message or f"Restoring primary/higher priority provider: {new_provider_label}",
                "is_recovery": True,  # Flag to indicate this is a recovery, not a failure-switch
                "original_provider_label": previous_provider_label,  # The backup we're switching FROM
                "new_provider_label": new_provider_label,  # The primary we're restoring TO
                "start_time": time.perf_counter(),
                "end_time": time.perf_counter(),
                "duration_ms": 0,
            })
            return True
        return False
