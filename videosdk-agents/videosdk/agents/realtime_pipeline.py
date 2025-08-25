from __future__ import annotations

import logging
from typing import Any, Literal
import av
import time

from .pipeline import Pipeline
from .event_emitter import EventEmitter
from .realtime_base_model import RealtimeBaseModel
from .agent import Agent
from .job import get_current_job_context
from .metrics import realtime_metrics_collector
from .denoise import Denoise
import logging

logger = logging.getLogger(__name__)

class RealTimePipeline(Pipeline, EventEmitter[Literal["realtime_start", "realtime_end","user_audio_input_data", "user_speech_started", "realtime_model_transcription"]]):
    """
    RealTime pipeline implementation that processes data in real-time.
    Inherits from Pipeline base class and adds realtime-specific events.
    """
    
    def __init__(
        self,
        model: RealtimeBaseModel,
        avatar: Any | None = None,
        denoise: Denoise | None = None,
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
        self.avatar = avatar
        self.vision = False
        self.denoise = denoise
        super().__init__()
        self.model.on("error", self.on_model_error)
        self.model.on("realtime_model_transcription", self.on_realtime_model_transcription)

    
    def set_agent(self, agent: Agent) -> None:
        self.agent = agent
        if hasattr(self.model, 'set_agent'):
            self.model.set_agent(agent)

    def _configure_components(self) -> None:
        """Configure pipeline components with the loop"""
        if self.loop:
            self.model.loop = self.loop
            job_context = get_current_job_context()
            
            if job_context and job_context.room:
                requested_vision = getattr(job_context.room, 'vision', False)
                self.vision = requested_vision
                
                model_name = self.model.__class__.__name__
                if requested_vision and model_name != 'GeminiRealtime':
                    logger.warning(f"Vision mode requested but {model_name} doesn't support video input. Only GeminiRealtime supports vision. Disabling vision.")
                    self.vision = False
                
                if self.avatar:
                    self.model.audio_track = getattr(job_context.room, 'agent_audio_track', None) or job_context.room.audio_track
                elif self.audio_track:
                     self.model.audio_track = self.audio_track

    async def start(self, **kwargs: Any) -> None:
        """
        Start the realtime pipeline processing.
        Overrides the abstract start method from Pipeline base class.
        
        Args:
            **kwargs: Additional arguments for pipeline configuration
        """
        await self.model.connect()
        self.model.on("user_speech_started", self.on_user_speech_started)

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
        if self.denoise:
            audio_data = await self.denoise.denoise(audio_data)
        await self.model.handle_audio_input(audio_data)

    async def on_video_delta(self, video_data: av.VideoFrame):
        """
        Handle incoming video data from the user
        The model's handle_video_input is now expected to handle the av.VideoFrame.
        """
        if self.vision and hasattr(self.model, 'handle_video_input'):
            await self.model.handle_video_input(video_data)
    
    def on_user_speech_started(self, data: dict) -> None:
        """
        Handle user speech started event
        """
        self._notify_speech_started()

    async def leave(self) -> None:
        """
        Leave the realtime pipeline.
        """
        if self.room is not None:
            await self.room.leave()

    def on_model_error(self, error: Exception):
        """
        Handle errors emitted from the model and send to realtime metrics cascading_metrics_collector.
        """
        error_data = {"message": str(error), "timestamp": time.time()}
        realtime_metrics_collector.set_realtime_model_error(error_data)
        logger.error(f"Realtime model error: {error_data}")

    def on_realtime_model_transcription(self, data: dict) -> None:
        """
        Handle realtime model transcription event
        """
        try:
            self.emit("realtime_model_transcription", data)
        except Exception:
            logger.error(f"Realtime model transcription: {data}")
    

    async def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'room') and self.room is not None:
            try:
                await self.room.leave()
            except Exception as e:
                logger.error(f"Error while leaving room during cleanup: {e}")
            try:
                if hasattr(self.room, 'cleanup'):
                    await self.room.cleanup()
            except Exception as e:
                logger.error(f"Error while cleaning up room: {e}")
        if hasattr(self, 'model'):
            await self.model.aclose()