from abc import ABC, abstractmethod
from typing import Optional, Any

class BaseTransportHandler(ABC):
    """
    Abstract base class for all transport layers (VideoSDK, WebSocket, WebRTC).
    """
    def __init__(self, loop, pipeline):
        self.loop = loop
        self.pipeline = pipeline
        self.audio_track = None 
    @abstractmethod
    async def connect(self):
        """Establish the connection (join room or start server)"""
        pass

    @abstractmethod
    async def disconnect(self):
        """Close the connection"""
        pass

    @abstractmethod
    async def wait_for_participant(self, participant_id: Optional[str] = None) -> str:
        """Wait for a user to connect/join"""
        pass

    @abstractmethod
    async def cleanup(self):
        """Free resources"""
        pass

    def setup_session_end_callback(self, callback):
        """Set up the callback to be called when the session ends."""
        self._on_session_end = callback

