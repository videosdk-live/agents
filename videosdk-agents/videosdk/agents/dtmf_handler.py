import asyncio
from typing import Callable, Optional
from videosdk import PubSubSubscribeConfig
from .job import get_current_job_context
import logging

logger = logging.getLogger(__name__)


class DTMFHandler:
    """
    Handles DTMF events from PubSub and forwards digits to callbacks.
    """

    def __init__(self, callback: Optional[Callable] = None):
        self.ctx = get_current_job_context()
        self._callback = callback
        self._subscribed = False

    async def start(self):
        """
        Begins listening to DTMF_EVENT from pubsub.
        Called by AgentSession automatically.
        """
        if self._subscribed:
            return

        subscribe_config = PubSubSubscribeConfig(
        topic="DTMF_EVENT",
        cb=lambda msg: asyncio.create_task(self._on_pubsub_event(msg))
        )

        await self.ctx.room.subscribe_to_pubsub(subscribe_config)
        self._subscribed = True

    def set_callback(self, callback: Callable):
        """
        Allows developers to attach or update callback.
        """
        self._callback = callback

    async def _on_pubsub_event(self, message):
        """
        Internal PubSub handler - extracts digit and forwards to user callback.
        """
        try:
            digit = message.get("payload", {}).get("number")
            logger.info(f"[DTMFHandler] Received: {digit}")

            if not digit or not self._callback:
                return

            if asyncio.iscoroutinefunction(self._callback):
                await self._callback(digit)
            else:
                self._callback(digit)

        except Exception as e:
            logger.error(f"[DTMFHandler] Error processing message: {e}")
