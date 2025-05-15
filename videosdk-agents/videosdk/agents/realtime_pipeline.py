from __future__ import annotations

from typing import Any, Dict, Literal
import asyncio

from .pipeline import Pipeline
from .event_emitter import EventEmitter
from .realtime_base_model import RealtimeBaseModel
from .room.room import VideoSDKHandler

class RealTimePipeline(Pipeline, EventEmitter[Literal["realtime_start", "realtime_end","user_audio_input_data"]]):
    """
    RealTime pipeline implementation that processes data in real-time.
    Inherits from Pipeline base class and adds realtime-specific events.
    """
    
    def __init__(
        self,
        model: RealtimeBaseModel
    ) -> None:
        """
        Initialize the realtime pipeline.
        
        Args:
            model: Instance of RealtimeBaseModel to process data
            config: Configuration dictionary with settings like:
                   - response_modalities: List of enabled modalities
                   - silence_threshold_ms: Silence threshold in milliseconds
        """
        super().__init__()
        self.model = model
        self.loop = asyncio.get_event_loop()
        self.room = None
        self.model.loop = self.loop
        self.model.audio_track = None

    async def start(self, **kwargs: Any) -> None:
        """
        Start the realtime pipeline processing.
        Overrides the abstract start method from Pipeline base class.
        
        Args:
            meeting_id: The meeting ID to join
            name: The name of the agent in the meeting
            **kwargs: Additional arguments for pipeline configuration
        """
        try:
            meeting_id = kwargs.get('meeting_id')
            name = kwargs.get('name')
            self.room = VideoSDKHandler(
                meeting_id=meeting_id,
                name=name,
                pipeline=self,
                loop=self.loop
            )
            
            self.room.init_meeting()
            self.model.loop = self.loop
            self.model.audio_track = self.room.audio_track
            
            await self.model.connect()
            await self.room.join()
            
        except Exception as e:
            print(f"Error starting realtime connection: {e}")
            await self.cleanup()
            raise

    async def send_message(self, message: str) -> None:
        """
        Send a message through the realtime model.
        Delegates to the model's send_message implementation.
        """
        await self.model.send_message(message)
    
    async def on_audio_delta(self, audio_data: bytes):
        """
        Handle incoming audio data from the user
        """
        await self.model.handle_audio_input(audio_data)
        
    async def leave(self) -> None:
        """
        Leave the realtime pipeline.
        """
        await self.room.leave()
        

    async def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'room'):
            await self.room.leave()
            await self.room.cleanup()
        if hasattr(self, 'model'):
            await self.model.aclose()