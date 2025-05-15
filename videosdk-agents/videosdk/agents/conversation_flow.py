from __future__ import annotations

from typing import Callable, Literal

from .event_emitter import EventEmitter

class ConversationFlow(EventEmitter[Literal["transcription"]]):
    """
    Manages the conversation flow by listening to transcription events.
    """
    
    def __init__(self) -> None:
        """Initialize conversation flow with event emitter capabilities"""
        super().__init__() 
        self.transcription_callback: Callable[[str], None] | None = None

    def on_transcription(self, callback: Callable[[str], None]) -> None:
        """
        Set the callback for transcription events.
        
        Args:
            callback: Function to call when transcription occurs, takes transcribed text as argument
        """
        self.on("transcription_event", lambda data: callback(data["text"]))