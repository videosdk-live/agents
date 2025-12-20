import logging
import asyncio
import time
from typing import List, Any, Optional

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

    @property
    def active_provider(self):
        return self.providers[self._current_index]

    @property
    def label(self) -> str:
        return f"Fallback{self._component_name}(active={self.active_provider.label})"

    async def _switch_provider(self, reason: str, failed_provider: Any = None):
        """
        Internal: Switch to the next provider in the list.
        Returns True if switched successfully, False if no providers left.
        """
        async with self._switch_lock:
            if failed_provider and failed_provider != self.active_provider:
                logger.info(f"[{self._component_name}] Provider {getattr(failed_provider, 'label', 'Unknown')} already switched. Current: {self.active_provider.label}")
                return True

            logger.warning(f"[{self._component_name}] Provider {self.active_provider.label} failed: {reason}")
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
                return False

            self._current_index += 1
            logger.info(f"[{self._component_name}] Switched to backup: {self.active_provider.label}")
            return True

    def check_recovery(self):
        """
        Checks if any higher-priority providers (lower index than current) 
        have passed their recovery cooldown. If so, switches back to the best one.
        """
        now = time.time()
        best_ready_index = self._current_index
        
        for i in range(self._current_index):
            attempts = self._recovery_attempts.get(i, 0)
            if attempts >= self.permanent_disable_after_attempts:
                continue

            if i in self._failed_providers:
                elapsed = now - self._failed_providers[i]
                if elapsed > self.temporary_disable_sec:
                    logger.info(f"[{self._component_name}] Provider {i} (Label: {self.providers[i].label}) cooldown expired ({elapsed:.1f}s > {self.temporary_disable_sec}s). Attempting recovery.")
                    del self._failed_providers[i]
                    best_ready_index = i
                    break
            else:
                pass
        
        if best_ready_index < self._current_index:
             logger.info(f"[{self._component_name}] Restoring primary/higher priority provider: {self.providers[best_ready_index].label}")
             self._current_index = best_ready_index
             return True
        return False
