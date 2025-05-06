from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, Literal
from dataclasses import dataclass

import aiohttp

from ...agent.realtime_base_model import RealtimeBaseModel

OPENAI_BASE_URL = "https://api.openai.com/v1"

OpenAIEventTypes = Literal[
    "response_created",
    "response_completed",
    "audio_generated"
]

@dataclass
class OpenAISession:
    """Represents an OpenAI WebSocket session"""
    ws: aiohttp.ClientWebSocketResponse
    msg_queue: asyncio.Queue
    tasks: list[asyncio.Task]

class OpenAIRealtime(RealtimeBaseModel[OpenAIEventTypes]):
    """OpenAI's realtime model implementation."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = OPENAI_BASE_URL,
    ) -> None:
        """
        Initialize OpenAI realtime model.
        
        Args:
            api_key: OpenAI API key
            base_url: Base URL for OpenAI API
        """
        self.api_key = api_key
        self.base_url = base_url
        self._config: Dict[str, Any] | None = None
        self._session: Optional[OpenAISession] = None
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._closing = False

    def set_config(self, config: Dict[str, Any]) -> None:
        """Set configuration received from pipeline"""
        super().__init__(config)
        self._config = config
        
        # Extract config values
        self.response_modalities = config.get("response_modalities", ["audio"])
        self.silence_threshold_ms = config.get("silence_threshold_ms", 500)
        self.model = config.get("model", "gpt-4")

    async def _ensure_http_session(self) -> aiohttp.ClientSession:
        """Ensure we have an HTTP session"""
        if not self._http_session:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def _create_session(self) -> OpenAISession:
        """Create a new WebSocket session"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        http_session = await self._ensure_http_session()
        ws = await http_session.ws_connect(
            f"{self.base_url}/realtime",
            headers=headers,
            params={"model": self.model}
        )
        
        msg_queue: asyncio.Queue = asyncio.Queue()
        tasks: list[asyncio.Task] = []
        
        return OpenAISession(ws=ws, msg_queue=msg_queue, tasks=tasks)

    async def _handle_websocket(self, session: OpenAISession) -> None:
        """Start WebSocket send/receive tasks"""
        session.tasks.extend([
            asyncio.create_task(self._send_loop(session), name="send_loop"),
            asyncio.create_task(self._receive_loop(session), name="receive_loop")
        ])

    async def _send_loop(self, session: OpenAISession) -> None:
        """Send messages from queue to WebSocket"""
        try:
            while not self._closing:
                msg = await session.msg_queue.get()
                if isinstance(msg, dict):
                    await session.ws.send_json(msg)
                else:
                    await session.ws.send_str(str(msg))
        except asyncio.CancelledError:
            pass
        finally:
            await self._cleanup_session(session)

    async def _receive_loop(self, session: OpenAISession) -> None:
        """Receive and process WebSocket messages"""
        try:
            while not self._closing:
                msg = await session.ws.receive()
                
                if msg.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(json.loads(msg.data))
        except asyncio.CancelledError:
            pass
        finally:
            await self._cleanup_session(session)

    async def _handle_message(self, data: dict) -> None:
        """Handle incoming WebSocket messages"""
        event_type = data.get('type')
        # Handle different event types based on your needs
        # This will be called by the receive loop when messages arrive
        pass

    async def _cleanup_session(self, session: OpenAISession) -> None:
        """Clean up session resources"""
        self._closing = True
        
        # Cancel all tasks
        for task in session.tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close WebSocket
        if not session.ws.closed:
            await session.ws.close()

    async def process(self, **kwargs: Any) -> None:
        """
        Process data in realtime.
        This will be called by the pipeline's start method.
        """
        if self._config is None:
            raise RuntimeError("Config must be set via set_config before processing")

        try:
            # Create and store session
            self._session = await self._create_session()
            
            # Start WebSocket handling
            await self._handle_websocket(self._session)
            
            # Wait for tasks to complete
            if self._session.tasks:
                await asyncio.gather(*self._session.tasks)
                
        finally:
            # Cleanup
            if self._session:
                await self._cleanup_session(self._session)
            if self._http_session:
                await self._http_session.close()

    async def send_event(self, event: Dict[str, Any]) -> None:
        """Send an event to the WebSocket"""
        if self._session and not self._closing:
            await self._session.msg_queue.put(event)

    async def aclose(self) -> None:
        """Cleanup all resources"""
        self._closing = True
        if self._session:
            await self._cleanup_session(self._session)
        if self._http_session:
            await self._http_session.close()