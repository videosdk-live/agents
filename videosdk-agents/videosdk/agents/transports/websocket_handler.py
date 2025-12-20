import asyncio
import logging
import json
try:
    import websockets
except ImportError:
    websockets = None

from .base import BaseTransportHandler
from ..room.audio_stream import TeeCustomAudioStreamTrack

logger = logging.getLogger(__name__)

class WebSocketAudioTrack(TeeCustomAudioStreamTrack):
    def __init__(self, loop, websocket_handler, sinks=None, pipeline=None):
        super().__init__(loop, sinks, pipeline)
        self.websocket_handler = websocket_handler
        self._ignore_packets = False

    def interrupt(self):
        self._ignore_packets = True
        super().interrupt()
        if self.websocket_handler and self.websocket_handler.active_connection:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.websocket_handler.active_connection.send(json.dumps({"type": "interrupt"})),
                    self.loop
                )
            except Exception as e:
                logger.error(f"Error sending interruption signal: {e}")

    async def add_new_bytes(self, audio_data: bytes):
        if self._ignore_packets:
            return
        await super().add_new_bytes(audio_data)

    def enable_audio_input(self, manual_control: bool = False):
        self._ignore_packets = False
        super().enable_audio_input(manual_control)

class WebSocketTransportHandler(BaseTransportHandler):
    def __init__(self, loop, pipeline, port=8080, path="/ws"):
        super().__init__(loop, pipeline)
        self.port = port
        self.path = path
        self.server = None
        self.active_connection = None
        self._participant_joined_event = asyncio.Event()
        self._stop_event = asyncio.Event()        
        self.audio_track = WebSocketAudioTrack(loop=loop, websocket_handler=self, pipeline=pipeline)
        self._on_session_end = None

    async def connect(self):
        if not websockets:
            raise ImportError("websockets library is required for WebSocketConnectionHandler. Install it with `pip install websockets`.")
            
        logger.info(f"Starting WebSocket server on port {self.port} at {self.path}")
        self.server = await websockets.serve(self._handle_connection, "0.0.0.0", self.port)

    async def _handle_connection(self, websocket):        
        logger.info("New WebSocket connection established")
        self.active_connection = websocket

        self._participant_joined_event.set()
        
        async def audio_sink(data: bytes):
            try:
                await websocket.send(data)
            except Exception:
                pass 

        if self.pipeline and hasattr(self.pipeline, 'audio_track') and self.pipeline.audio_track:
             if hasattr(self.pipeline.audio_track, 'add_sink'):
                self.pipeline.audio_track.add_sink(audio_sink)
             else:
                 if hasattr(self.pipeline.audio_track, 'sinks'):
                     self.pipeline.audio_track.sinks.append(audio_sink)
        
        elif self.audio_track:
             if hasattr(self.audio_track, 'add_sink'):
                self.audio_track.add_sink(audio_sink)

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    if self.pipeline:
                        await self.pipeline.on_audio_delta(message)
                elif isinstance(message, str):
                    logger.debug(f"Ignored text message: {message[:50]}...")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        finally:
            if self.pipeline and hasattr(self.pipeline, 'audio_track'):
                if hasattr(self.pipeline.audio_track, 'remove_sink'):
                    self.pipeline.audio_track.remove_sink(audio_sink)
                elif hasattr(self.pipeline.audio_track, 'sinks') and audio_sink in self.pipeline.audio_track.sinks:
                    self.pipeline.audio_track.sinks.remove(audio_sink)
            
            self.active_connection = None

            if self._on_session_end:
                try:
                    self._on_session_end("websocket_disconnected")
                except Exception as e:
                    logger.error(f"Error in session end callback: {e}")

    async def wait_for_participant(self, participant_id=None):
        logger.info("Waiting for WebSocket client...")
        await self._participant_joined_event.wait()
        return "ws_client"

    async def disconnect(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def cleanup(self):
        await self.disconnect()