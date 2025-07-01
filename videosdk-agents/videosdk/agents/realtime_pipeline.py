from __future__ import annotations

from typing import Any, Literal
import asyncio

from .pipeline import Pipeline
from .event_emitter import EventEmitter
from .realtime_base_model import RealtimeBaseModel
from .room.room import VideoSDKHandler
from .agent import Agent
class RealTimePipeline(Pipeline, EventEmitter[Literal["realtime_start", "realtime_end","user_audio_input_data"]]):
    """
    RealTime pipeline implementation that processes data in real-time.
    Inherits from Pipeline base class and adds realtime-specific events.
    """
    
    def __init__(
        self,
        model: RealtimeBaseModel,
    ) -> None:
        """
        Initialize the realtime pipeline.
        
        Args:
            model: Instance of RealtimeBaseModel to process data
            config: Configuration dictionary with settings like:
                   - response_modalities: List of enabled modalities
                   - silence_threshold_ms: Silence threshold in milliseconds
        """
        self.model = model
        self.model.audio_track = None
        self.agent = None
        super().__init__()
    
    def set_agent(self, agent: Agent) -> None:
        self.agent = agent
        if hasattr(self.model, 'set_agent'):
            self.model.set_agent(agent)

    def _configure_components(self) -> None:
        """Configure pipeline components with the loop"""
        if self.loop and self.audio_track:
            self.model.loop = self.loop
            self.model.audio_track = self.audio_track

    async def start(self, **kwargs: Any) -> None:
        """
        Start the realtime pipeline processing.
        Overrides the abstract start method from Pipeline base class.
        
        Args:
            **kwargs: Additional arguments for pipeline configuration
        """
        await self.model.connect()

    async def send_message(self, message: str) -> None:
        """
        Send a message through the realtime model.
        Delegates to the model's send_message implementation.
        """

        await self.model.send_message(message)

    async def send_text_message(self, message: str) -> None:
        """
        Send a text message through the realtime model.
        This method specifically handles text-only input when modalities is ["text"].
        """
        if hasattr(self.model, 'send_text_message'):
            await self.model.send_text_message(message)
        else:
            await self.model.send_message(message)
    
    async def on_audio_delta(self, audio_data: bytes):
        """
        Handle incoming audio data from the user
        """
        await self.model.handle_audio_input(audio_data)

    async def on_video_delta(self, video_data: av.VideoFrame):
        """
        Handle incoming video data from the user
        The model's handle_video_input is now expected to handle the av.VideoFrame.
        """
        if self.vision and hasattr(self.model, 'handle_video_input'):
            await self.model.handle_video_input(video_data)

    async def leave(self) -> None:
        """
        Leave the realtime pipeline.
        """
        if self.room is not None:
            await self.room.leave()

    async def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'room') and self.room is not None:
            await self.room.leave()
            if hasattr(self.room, 'cleanup'):
                await self.room.cleanup()
        if hasattr(self, 'model'):
            await self.model.aclose()