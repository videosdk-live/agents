from __future__ import annotations

from typing import AsyncIterator, Any, TYPE_CHECKING
import asyncio
import logging
from .llm.llm import ResponseChunk
from .llm.chat_context import ChatContext, ChatRole
from .event_emitter import EventEmitter
from .realtime_base_model import RealtimeBaseModel

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)


class RealtimeLLMAdapter(EventEmitter):
    """
    Wraps a RealtimeBaseModel to expose an LLM-compatible interface.
    
    This allows realtime models (like OpenAI Realtime API, Gemini Live) to be
    used in place of standard LLM components in the pipeline architecture.
    
    Key differences from standard LLMs:
    - Realtime models handle their own audio I/O (STT + TTS built-in)
    - They maintain their own conversation state
    - Function calling may work differently
    
    This wrapper primarily delegates to the underlying realtime model and
    provides adapter methods to make it look like an LLM from the pipeline's
    perspective.
    """
    
    def __init__(self, realtime_model: RealtimeBaseModel, agent: Agent | None = None):
        super().__init__()
        self.realtime_model = realtime_model
        self.agent = agent
        self._is_realtime = True
        self.audio_track = None
        self.loop = None
        
        self.realtime_model.on("error", lambda error: self.emit("error", error))
        self.realtime_model.on("user_speech_started", lambda data: self.emit("user_speech_started", data))
        self.realtime_model.on("user_speech_ended", lambda data: self.emit("user_speech_ended", data))
        self.realtime_model.on("agent_speech_started", lambda data: self.emit("agent_speech_started", data))
        self.realtime_model.on("agent_speech_ended", lambda data: self.emit("agent_speech_ended", data))
        self.realtime_model.on("realtime_model_transcription", lambda data: self.emit("realtime_model_transcription", data))
    
    def set_agent(self, agent: Agent) -> None:
        """Set the agent for this wrapper"""
        self.agent = agent
        if hasattr(self.realtime_model, 'set_agent'):
            self.realtime_model.set_agent(agent)
    
    async def connect(self) -> None:
        """Connect the realtime model"""
        await self.realtime_model.connect()
    
    async def chat(
        self,
        context: ChatContext,
        tools: list[Any] | None = None,
        conversational_graph: Any | None = None,
        **kwargs
    ) -> AsyncIterator[ResponseChunk]:
        """
        Adapter method for LLM compatibility.
        
        For realtime models, the chat method is less relevant since they handle
        audio I/O directly. This method exists for interface compatibility but
        yields minimal content since the actual response happens through audio.
        
        Args:
            context: Chat context (may be ignored by realtime models)
            tools: Available function tools
            conversational_graph: Optional conversational graph
            **kwargs: Additional arguments
            
        Yields:
            ResponseChunk objects (mostly empty for realtime models)
        """
        logger.info("RealtimeLLMAdapter.chat() called - realtime models handle I/O directly")
        
        async def empty_gen():
            yield ResponseChunk(content="", metadata={"realtime_mode": True}, role=ChatRole.ASSISTANT)
        
        async for chunk in empty_gen():
            yield chunk
    
    async def handle_audio_input(self, audio_data: bytes) -> None:
        """
        Process incoming audio through the realtime model.
        
        Args:
            audio_data: Raw audio bytes
        """
        await self.realtime_model.handle_audio_input(audio_data)
    
    async def handle_video_input(self, video_frame: Any) -> None:
        """
        Process incoming video through the realtime model (if supported).
        
        Args:
            video_frame: Video frame data
        """
        if hasattr(self.realtime_model, 'handle_video_input'):
            await self.realtime_model.handle_video_input(video_frame)
        else:
            logger.warning(f"Realtime model {type(self.realtime_model).__name__} does not support video input")
    
    async def send_message(self, message: str) -> None:
        """
        Send a text message to the realtime model.
        
        Args:
            message: Text message to send
        """
        await self.realtime_model.send_message(message)
    
    async def send_text_message(self, message: str) -> None:
        """
        Send a text-only message (for models supporting text modality).
        
        Args:
            message: Text message to send
        """
        if hasattr(self.realtime_model, 'send_text_message'):
            await self.realtime_model.send_text_message(message)
        else:
            await self.realtime_model.send_message(message)
    
    async def send_message_with_frames(self, message: str, frames: list[Any]) -> None:
        """
        Send a message with video frames (for vision-enabled models).
        
        Args:
            message: Text message
            frames: List of video frames
        """
        if hasattr(self.realtime_model, 'send_message_with_frames'):
            await self.realtime_model.send_message_with_frames(message, frames)
        else:
            logger.warning(f"Realtime model {type(self.realtime_model).__name__} does not support frames")
            await self.send_message(message)
    
    async def interrupt(self) -> None:
        """Interrupt the realtime model's current response"""
        if hasattr(self.realtime_model, 'interrupt'):
            await self.realtime_model.interrupt()
    
    async def cancel_current_generation(self) -> None:
        """Cancel the current generation (LLM compatibility method)"""
        await self.interrupt()
    
    def on_user_speech_started(self, callback) -> None:
        """Register callback for user speech started event"""
        self.realtime_model.on("user_speech_started", callback)
    
    def on_user_speech_ended(self, callback) -> None:
        """Register callback for user speech ended event"""
        self.realtime_model.on("user_speech_ended", callback)
    
    def on_agent_speech_started(self, callback) -> None:
        """Register callback for agent speech started event"""
        self.realtime_model.on("agent_speech_started", callback)
    
    def on_agent_speech_ended(self, callback) -> None:
        """Register callback for agent speech ended event"""
        self.realtime_model.on("agent_speech_ended", callback)
    
    def on_transcription(self, callback) -> None:
        """Register callback for transcription events"""
        self.realtime_model.on("realtime_model_transcription", callback)
    
    @property
    def current_utterance(self):
        """Get current utterance handle"""
        return getattr(self.realtime_model, 'current_utterance', None)
    
    @current_utterance.setter
    def current_utterance(self, value):
        """Set current utterance handle"""
        if hasattr(self.realtime_model, 'current_utterance'):
            self.realtime_model.current_utterance = value
    
    async def aclose(self) -> None:
        """Close and cleanup the realtime model"""
        await self.realtime_model.aclose()
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        await self.aclose()
