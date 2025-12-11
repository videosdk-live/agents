from __future__ import annotations

import logging
from typing import Any, Literal
import av
import time
import asyncio
from .pipeline import Pipeline
from .event_emitter import EventEmitter
from .realtime_base_model import RealtimeBaseModel
from .agent import Agent
from .job import get_current_job_context
from .metrics import realtime_metrics_collector
from .denoise import Denoise
import logging
from .utils import UserState, AgentState
from .utterance_handle import UtteranceHandle
from .voice_mail_detector import VoiceMailDetector

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
        self._vision_lock = asyncio.Lock()
        self.denoise = denoise
        self.voice_mail_detector: VoiceMailDetector | None = None
        self.voice_mail_detection_done = False
        self._vmd_buffer = ""
        self._vmd_check_task: asyncio.Task | None = None

        super().__init__()
        self.model.on("error", self.on_model_error)
        self.model.on("realtime_model_transcription", self.on_realtime_model_transcription)
        self.model.on("agent_speech_ended", self._on_agent_speech_ended)

    def set_voice_mail_detector(self, detector: VoiceMailDetector | None) -> None:
        """Called by AgentSession to configure VMD"""
        self.voice_mail_detector = detector
        self.voice_mail_detection_done = False
        self._vmd_buffer = ""
    
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
                if requested_vision and model_name != 'GeminiRealtime' and model_name != "OpenAIRealtime":
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
        self.model.on("user_speech_ended", lambda data: asyncio.create_task(self.on_user_speech_ended(data)))
        self.model.on("agent_speech_started", lambda data: asyncio.create_task(self.on_agent_speech_started(data)))
        self.model.on("agent_speech_ended", self._on_agent_speech_ended)

    async def send_message(self, message: str, handle: UtteranceHandle) -> None:
        """
        Send a message through the realtime pipeline and track the utterance handle.
        """
        self._current_utterance_handle = handle
        self.model.current_utterance = handle
        try:
            await self.model.send_message(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.model.current_utterance = None
            handle._mark_done()

    async def send_text_message(self, message: str) -> None:
        """
        Send a text message through the realtime model.
        This method specifically handles text-only input when modalities is ["text"].
        """
        if hasattr(self.model, 'send_text_message'):
            await self.model.send_text_message(message)
        else:
            await self.model.send_message(message)
    
    def _on_agent_speech_ended(self, data: dict) -> None:
        """
        Handle agent speech ended event and mark utterance as done, forwarding to agent if handler exists.
        """
        if self._current_utterance_handle and not self._current_utterance_handle.done():
            self._current_utterance_handle._mark_done()
        self.model.current_utterance = None
        if self.agent and hasattr(self.agent, 'on_agent_speech_ended'):
            self.agent.on_agent_speech_ended(data)
    
    async def on_audio_delta(self, audio_data: bytes):
        """
        Handle incoming audio data from the user
        """
        if self.denoise:
            audio_data = await self.denoise.denoise(audio_data)
        await self.model.handle_audio_input(audio_data)

    async def on_video_delta(self, video_data: av.VideoFrame):
        """Handle incoming video data from the user"""
        if self._vision_lock.locked():
            logger.info("Vision lock is locked, skipping video data")
            return
            
        if self.vision:
            self._recent_frames.append(video_data)
            if len(self._recent_frames) > self._max_frames_buffer:
                self._recent_frames.pop(0)
            await self.model.handle_video_input(video_data)

    def on_user_speech_started(self, data: dict) -> None:
        """
        Handle user speech started event
        """
        self._notify_speech_started()
        # self.interrupt() # Not sure yet whether this affects utterance handling.
        if self.agent.session:
            self.agent.session._emit_user_state(UserState.SPEAKING)
            self.agent.session._emit_agent_state(AgentState.LISTENING)
            
    def interrupt(self) -> None:
        """
        Interrupt the realtime pipeline
        """
        if self.model:
            if self.model.current_utterance and not self.model.current_utterance.is_interruptible:
                logger.info("Interruption is disabled for the current utterance. Not interrupting realtime pipeline.")
                return
            else:
                asyncio.create_task(self.model.interrupt())
        if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
            asyncio.create_task(self.agent.session.stop_thinking_audio())
        if self._current_utterance_handle and not self._current_utterance_handle.done():
            if self._current_utterance_handle.is_interruptible:
                self._current_utterance_handle.interrupt()
            else:
                logger.info("Current utterance handle is not interruptible. Skipping handle interrupt.")
        if self.avatar and hasattr(self.avatar, 'interrupt'):
            asyncio.create_task(self.avatar.interrupt())

        if self._vmd_check_task and not self._vmd_check_task.done():
            self._vmd_check_task.cancel()
        self._vmd_buffer = ""

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
            if self.voice_mail_detector and not self.voice_mail_detection_done:
                text = data.get("text", "")
                role = data.get("role") 
                
                if role == "user" and text and isinstance(text, str) and text.strip():
                    self._vmd_buffer += f" {text}"
                    
                    if not self._vmd_check_task:
                        
                        self._vmd_check_task = asyncio.create_task(self._run_vmd_check())
        except Exception:
            logger.error(f"Realtime model transcription: {data}")
    
    async def _run_vmd_check(self) -> None:
        """Waits, detects, emits result"""
        try:
            if not self.voice_mail_detector:
                return

            await asyncio.sleep(self.voice_mail_detector.duration)
            
            is_voicemail = await self.voice_mail_detector.detect(self._vmd_buffer.strip())
            self.voice_mail_detection_done = True
            
            if is_voicemail:
                logger.info("[RealTime] Voicemail Detected! Interrupting.")
                self.interrupt()

            self.emit("voicemail_result", {
                "is_voicemail": is_voicemail,
                "transcript": self._vmd_buffer.strip()
            })
            
        except Exception as e:
            logger.error(f"Error in VMD check: {e}")
            self.emit("voicemail_result", {"is_voicemail": False})
        finally:
            self._vmd_check_task = None
            self._vmd_buffer = ""
    

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up realtime pipeline")
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
            self.room = None
        
        if hasattr(self, 'model') and self.model is not None:
            try:
                await self.model.aclose()
            except Exception as e:
                logger.error(f"Error while closing model during cleanup: {e}")
            self.model = None
        
        if self._current_utterance_handle:
            self._current_utterance_handle.interrupt()
            self._current_utterance_handle = None
        
        if hasattr(self, 'avatar') and self.avatar is not None:
            try:
                if hasattr(self.avatar, 'cleanup'):
                    await self.avatar.cleanup()
                elif hasattr(self.avatar, 'aclose'):
                    await self.avatar.aclose()
            except Exception as e:
                logger.error(f"Error while cleaning up avatar: {e}")
            self.avatar = None
        
        if hasattr(self, 'denoise') and self.denoise is not None:
            try:
                await self.denoise.aclose()
            except Exception as e:
                logger.error(f"Error while cleaning up denoise: {e}")
            self.denoise = None
        
        self.agent = None
        self.vision = False
        self.model = None
        self.avatar = None
        self.denoise = None
        self._current_utterance_handle = None
        self.model.current_utterance = None
        if self._vmd_check_task and not self._vmd_check_task.done():
            self._vmd_check_task.cancel()
        self.voice_mail_detector = None
        self._vmd_buffer = ""

        
        logger.info("Realtime pipeline cleaned up")
        await super().cleanup()

    async def on_user_speech_ended(self, data: dict) -> None:
        """
        Handle agent turn started event
        """
        if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
            await self.agent.session.start_thinking_audio()

    async def on_agent_speech_started(self, data: dict) -> None:
        """
        Handle agent speech started event
        """
        if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
            await self.agent.session.stop_thinking_audio()

    async def reply_with_context(self, instructions: str, wait_for_playback: bool, handle: UtteranceHandle, frames: list[av.VideoFrame] | None = None) -> None:
        """
        Generate a reply using instructions and optional frames.
        
        Args:
            instructions: Instructions/text to send to the model
            wait_for_playback: If True, wait for playback to complete (for realtime, this is handled by the model)
            handle: UtteranceHandle to track the utterance
            frames: Optional list of VideoFrame objects to include in the reply
        """
        self._current_utterance_handle = handle
        self.model.current_utterance = handle
        
        if frames and hasattr(self.model, 'send_message_with_frames'):
            async with self._vision_lock:
                await self.model.send_message_with_frames(instructions, frames)
        elif hasattr(self.model, 'send_text_message'):
            await self.model.send_text_message(instructions)
        else:
            await self.model.send_message(instructions)
