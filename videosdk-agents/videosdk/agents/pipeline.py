from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Callable
import asyncio

from .event_emitter import EventEmitter
from .room.audio_stream import CustomAudioStreamTrack

class Pipeline(EventEmitter[Literal["start"]], ABC):
    """
    Base Pipeline class that other pipeline types (RealTime, Cascading) will inherit from.
    Inherits from EventEmitter to provide event handling capabilities.
    """
    
    def __init__(self) -> None:
        """Initialize the pipeline with event emitter capabilities"""
        super().__init__()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.audio_track: CustomAudioStreamTrack | None = None
        self._wake_up_callback: Optional[Callable[[], None]] = None
        self._auto_register()
        
    def _auto_register(self) -> None:
        """Automatically register this pipeline with the current job context"""
        try:
            from .job import get_current_job_context
            job_context = get_current_job_context()
            if job_context:
                job_context._set_pipeline_internal(self)
        except ImportError:
            pass

    def _set_loop_and_audio_track(self, loop: asyncio.AbstractEventLoop, audio_track: CustomAudioStreamTrack) -> None:
        """Set the event loop and configure components"""
        self.loop = loop
        self.audio_track = audio_track
        self._configure_components()

    def _configure_components(self) -> None:
        """Configure pipeline components with the loop - to be overridden by subclasses"""
        pass

    def set_wake_up_callback(self, callback: Callable[[], None]) -> None:
        self._wake_up_callback = callback

    def _notify_speech_started(self) -> None:
        if self._wake_up_callback:
            self._wake_up_callback()

    @abstractmethod
    async def start(self, **kwargs: Any) -> None:
        """
        Start the pipeline processing.
        This is an abstract method that must be implemented by child classes.
        
        Args:
            **kwargs: Additional arguments that may be needed by specific pipeline implementations
        """
        pass
    
    @abstractmethod
    async def on_audio_delta(self, audio_data: bytes) -> None:
        """
        Handle incoming audio data from the user
        """
        pass
    
    @abstractmethod
    async def send_message(self, message: str) -> None:
        """
        Send a message to the pipeline.
        """
        pass