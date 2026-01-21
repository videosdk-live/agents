from typing import Any, Literal, Optional, Callable, Dict, List, Tuple
import asyncio
import logging
import av
from dataclasses import dataclass, field
from .utterance_handle import UtteranceHandle
from .event_emitter import EventEmitter
from .room.output_stream import CustomAudioStreamTrack
from .pipeline_orchestrator import PipelineOrchestrator
from .pipeline_hooks import PipelineHooks
from .realtime_llm_adapter import RealtimeLLMAdapter
from .realtime_base_model import RealtimeBaseModel
from .stt.stt import STT
from .llm.llm import LLM
from .tts.tts import TTS
from .vad import VAD
from .eou import EOU
from .denoise import Denoise
from .voice_mail_detector import VoiceMailDetector
from .job import get_current_job_context

logger = logging.getLogger(__name__)


@dataclass
class EOUConfig:
    mode: Literal["ADAPTIVE", "DEFAULT"] = "DEFAULT"
    min_max_speech_wait_timeout: List[float] | Tuple[float, float] = field(default_factory=lambda: [0.5, 0.8])

    def __post_init__(self):
        if not (isinstance(self.min_max_speech_wait_timeout, (list, tuple)) and len(self.min_max_speech_wait_timeout) == 2):
            raise ValueError("min_max_speech_wait_timeout must be a list or tuple of two floats")
        min_val, max_val = self.min_max_speech_wait_timeout
        if not (isinstance(min_val, (int, float)) and isinstance(max_val, (int, float))):
            raise ValueError("min_max_speech_wait_timeout values must be numbers")
        if min_val <= 0 or max_val <= 0:
            raise ValueError("min_max_speech_wait_timeout values must be greater than 0")
        if min_val >= max_val:
            raise ValueError("min_speech_wait_timeout must be less than max_speech_wait_timeout")


@dataclass
class InterruptConfig:
    mode: Literal["VAD_ONLY", "STT_ONLY", "HYBRID"] = "HYBRID"
    interrupt_min_duration: float = 0.5
    interrupt_min_words: int = 2
    false_interrupt_pause_duration: float = 2.0
    resume_on_false_interrupt: bool = False

    def __post_init__(self):
        if self.interrupt_min_duration <= 0:
            raise ValueError("interrupt_min_duration must be greater than 0")
        if self.interrupt_min_words <= 0:
            raise ValueError("interrupt_min_words must be greater than 0")
        if self.false_interrupt_pause_duration <= 0:
            raise ValueError("false_interrupt_pause_duration must be greater than 0")


class Pipeline(EventEmitter[Literal["start", "error", "transcript_ready", "content_generated", "synthesis_complete"]]):
    """
    Unified Pipeline class supporting multiple component configurations.
    
    Supports:
    - Full cascading: VAD → STT → TurnD → LLM → TTS
    - Partial cascading: Any subset of components
    - Realtime: Speech-to-speech models (OpenAI Realtime, Gemini Live)
    - Hybrid: Components + user event callbacks
    
    Args:
        stt: Speech-to-Text processor (optional)
        llm: Language Model or RealtimeBaseModel (optional)
        tts: Text-to-Speech processor (optional)
        vad: Voice Activity Detector (optional)
        turn_detector: End-of-Utterance detector (optional)
        avatar: Avatar for visual output (optional)
        denoise: Audio denoiser (optional)
        eou_config: End of utterance configuration
        interrupt_config: Interruption configuration
        conversational_graph: Conversational graph for structured dialogs (optional)
        max_context_items: Maximum chat context items (auto-truncates when exceeded)
        voice_mail_detector: Voicemail detection (optional)
    """
    
    def __init__(
        self,
        stt: STT | None = None,
        llm: LLM | RealtimeBaseModel | None = None,
        tts: TTS | None = None,
        vad: VAD | None = None,
        turn_detector: EOU | None = None,
        avatar: Any | None = None,
        denoise: Denoise | None = None,
        eou_config: EOUConfig | None = None,
        interrupt_config: InterruptConfig | None = None,
        conversational_graph: Any | None = None,
        max_context_items: int | None = None,
        voice_mail_detector: VoiceMailDetector | None = None,
    ) -> None:
        super().__init__()
        
        # Store raw components
        self.stt = stt
        self.tts = tts
        self.vad = vad
        self.turn_detector = turn_detector
        self.avatar = avatar
        self.denoise = denoise
        self.conversational_graph = conversational_graph
        self.max_context_items = max_context_items
        self.voice_mail_detector = voice_mail_detector
        
        # Pipeline hooks for middleware/interception
        self.hooks = PipelineHooks()
        
        # Detect and handle realtime models
        self._is_realtime_mode = False
        self.llm: LLM | RealtimeLLMAdapter | None = None
        self._realtime_model: RealtimeBaseModel | None = None
        
        if isinstance(llm, RealtimeBaseModel):
            logger.info("Realtime model detected - wrapping as LLM")
            self._is_realtime_mode = True
            self._realtime_model = llm
            self.llm = RealtimeLLMAdapter(llm)
            
            if stt or tts:
                logger.warning("STT/TTS components ignored when using realtime model (model handles audio I/O)")
        else:
            self.llm = llm
        
        # Configuration
        self.eou_config = eou_config or EOUConfig()
        self.interrupt_config = interrupt_config or InterruptConfig()
        
        # Pipeline state
        self.agent = None
        self.orchestrator: PipelineOrchestrator | None = None
        self.vision = False
        self.loop: asyncio.AbstractEventLoop | None = None
        self.audio_track: CustomAudioStreamTrack | None = None
        self._wake_up_callback: Optional[Callable[[], None]] = None
        self._recent_frames: list[av.VideoFrame] = []
        self._max_frames_buffer = 5
        self._vision_lock = asyncio.Lock()
        self._current_utterance_handle: UtteranceHandle | None = None
        
        self._setup_error_handlers()
        
        self._auto_register()
    
    def _auto_register(self) -> None:
        """Automatically register this pipeline with the current job context"""
        try:
            job_context = get_current_job_context()
            if job_context:
                job_context._set_pipeline_internal(self)
        except Exception:
            pass
    
    def on(self, event: Literal["speech_in", "speech_out", "stt", "llm", "agent_response", "vision_frame", "user_turn_start", "user_turn_end", "agent_turn_start", "agent_turn_end"]) -> Callable:
        """
        Decorator to register a hook for pipeline events.
        
        Supported hooks:
        - speech_in: Process raw incoming user audio (async iterator)
        - speech_out: Process outgoing agent audio after TTS (async iterator)
        - stt: Process user transcript after STT, before LLM
        - llm: Control LLM invocation (can bypass with direct response)
        - agent_response: Process agent response after LLM, before TTS
        - vision_frame: Process video frames when vision is enabled (async iterator)
        - user_turn_start: Called when user turn starts
        - user_turn_end: Called when user turn ends
        - agent_turn_start: Called when agent processing starts
        - agent_turn_end: Called when agent finishes speaking
        
        Examples:
            @pipeline.on("speech_in")
            async def process_audio(audio_stream):
                '''Apply noise reduction to incoming audio'''
                async for audio_chunk in audio_stream:
                    # Process audio_chunk (bytes)
                    processed = apply_noise_reduction(audio_chunk)
                    yield processed
            
            @pipeline.on("stt")
            async def clean_transcript(transcript: str) -> str:
                '''Remove filler words from transcript'''
                return transcript.replace("um", "").replace("uh", "")
            
            @pipeline.on("agent_response")
            async def process_response(response: str):
                '''Stream modified response to TTS'''
                for word in response.split():
                    yield word.replace("API", "A P I") + " "
            
            @pipeline.on("vision_frame")
            async def process_frames(frame_stream):
                '''Apply filters to video frames'''
                async for frame in frame_stream:
                    # Process av.VideoFrame
                    filtered_frame = apply_filter(frame)
                    yield filtered_frame
            
            @pipeline.on("user_turn_start")
            async def on_user_turn_start(transcript: str) -> None:
                '''Log when user starts speaking'''
                print(f"User said: {transcript}")
            
            @pipeline.on("user_turn_end")
            async def on_user_turn_end() -> None:
                '''Log when user turn ends'''
                print("User turn ended")
            
            @pipeline.on("agent_turn_start")
            async def on_agent_turn_start() -> None:
                '''Log when agent starts processing'''
                print("Agent processing started")
            
            @pipeline.on("agent_turn_end")
            async def on_agent_turn_end() -> None:
                '''Log when agent finishes speaking'''
                print("Agent finished speaking")
            
            @pipeline.on("llm")
            async def custom_handler(transcript: str):
                '''Bypass LLM with streaming response or don't yield for normal flow'''
                if "hours" in transcript.lower():
                    # Yield to bypass LLM and stream response directly to TTS
                    for word in "We're open 24/7".split():
                        yield word + " "
                # If no yields, the generator will be empty and LLM will be used
        """
        return self.hooks.on(event)
    
    def _setup_error_handlers(self) -> None:
        """Setup error handlers for all components"""
        if self.stt:
            self.stt.on("error", lambda *args: self.on_component_error("STT", args[0] if args else "Unknown error"))
        if self.llm and not self._is_realtime_mode:
            self.llm.on("error", lambda *args: self.on_component_error("LLM", args[0] if args else "Unknown error"))
        if self.tts:
            self.tts.on("error", lambda *args: self.on_component_error("TTS", args[0] if args else "Unknown error"))
        if self.vad:
            self.vad.on("error", lambda *args: self.on_component_error("VAD", args[0] if args else "Unknown error"))
        if self.turn_detector:
            self.turn_detector.on("error", lambda *args: self.on_component_error("TURN-D", args[0] if args else "Unknown error"))
    
    def on_component_error(self, source: str, error_data: Any) -> None:
        """Handle error events from components"""
        logger.error(f"[{source}] Component error: {error_data}")
        self.emit("error", {"source": source, "error": str(error_data)})
    
    def _detect_pipeline_mode(self) -> str:
        """Detect the pipeline mode based on components"""
        if self._is_realtime_mode:
            return "realtime"
        elif self.stt and self.llm and self.tts and self.vad and self.turn_detector:
            return "full_cascading"
        elif self.llm and self.tts and not self.stt:
            return "llm_tts_only"
        elif self.stt and self.llm and not self.tts:
            return "stt_llm_only"
        elif not self.llm and self.stt and self.tts:
            return "hybrid"
        else:
            return "partial_cascading"
    
    def set_agent(self, agent: Any) -> None:
        """Set the agent for this pipeline"""
        self.agent = agent
        
        if not self._is_realtime_mode:
            self.orchestrator = PipelineOrchestrator(
                agent=agent,
                stt=self.stt,
                llm=self.llm,
                tts=self.tts,
                vad=self.vad,
                turn_detector=self.turn_detector,
                denoise=self.denoise,
                avatar=self.avatar,
                mode=self.eou_config.mode,
                min_speech_wait_timeout=self.eou_config.min_max_speech_wait_timeout,
                interrupt_mode=self.interrupt_config.mode,
                interrupt_min_duration=self.interrupt_config.interrupt_min_duration,
                interrupt_min_words=self.interrupt_config.interrupt_min_words,
                false_interrupt_pause_duration=self.interrupt_config.false_interrupt_pause_duration,
                resume_on_false_interrupt=self.interrupt_config.resume_on_false_interrupt,
                conversational_graph=self.conversational_graph,
                max_context_items=self.max_context_items,
                voice_mail_detector=self.voice_mail_detector,
                hooks=self.hooks, 
            )
            
            self.orchestrator.on("transcript_ready", lambda data: self.emit("transcript_ready", data))
            self.orchestrator.on("content_generated", lambda data: self.emit("content_generated", data))
            self.orchestrator.on("synthesis_complete", lambda data: self.emit("synthesis_complete", data))
            self.orchestrator.on("voicemail_result", lambda data: self.emit("voicemail_result", data))
        else:
            if isinstance(self.llm, RealtimeLLMAdapter):
                self.llm.set_agent(agent)
    
    def _set_loop_and_audio_track(self, loop: asyncio.AbstractEventLoop, audio_track: CustomAudioStreamTrack) -> None:
        """Set the event loop and configure components"""
        self.loop = loop
        self.audio_track = audio_track
        self._configure_components()
    
    def _configure_components(self) -> None:
        """Configure pipeline components with loop and audio track"""
        if not self.loop:
            return
        
        job_context = get_current_job_context()
        
        if job_context and job_context.room:
            requested_vision = getattr(job_context.room, 'vision', False)
            self.vision = requested_vision
            
            if requested_vision and self._is_realtime_mode:
                model_name = self._realtime_model.__class__.__name__ if self._realtime_model else "Unknown"
                if model_name not in ["GeminiRealtime", "OpenAIRealtime"]:
                    logger.warning(f"Vision requested but {model_name} doesn't support video input. Disabling vision.")
                    self.vision = False
        
        if not self._is_realtime_mode and self.tts:
            self.tts.loop = self.loop
            
            if self.avatar and job_context and job_context.room:
                self.tts.audio_track = getattr(job_context.room, "agent_audio_track", None) or job_context.room.audio_track
            elif self.audio_track:
                self.tts.audio_track = self.audio_track
            
            if self.tts.audio_track:
                logger.info(f"TTS audio track configured: {type(self.tts.audio_track).__name__}")
                # Set hooks on audio track for speech_out processing
                if hasattr(self.tts.audio_track, 'set_pipeline_hooks'):
                    self.tts.audio_track.set_pipeline_hooks(self.hooks)
            
            if self.orchestrator:
                self.orchestrator.set_audio_track(self.tts.audio_track)
        
        if self._is_realtime_mode and self._realtime_model:
            self._realtime_model.loop = self.loop
            
            if self.avatar and job_context and job_context.room:
                self._realtime_model.audio_track = getattr(job_context.room, 'agent_audio_track', None) or job_context.room.audio_track
            elif self.audio_track:
                self._realtime_model.audio_track = self.audio_track
            
            # Set hooks on audio track for speech_out processing
            if self._realtime_model.audio_track and hasattr(self._realtime_model.audio_track, 'set_pipeline_hooks'):
                self._realtime_model.audio_track.set_pipeline_hooks(self.hooks)
    
    def set_wake_up_callback(self, callback: Callable[[], None]) -> None:
        """Set wake-up callback for speech detection"""
        self._wake_up_callback = callback
    
    def _notify_speech_started(self) -> None:
        """Notify that user speech started (triggers wake-up)"""
        if self._wake_up_callback:
            self._wake_up_callback()
    
    async def start(self, **kwargs: Any) -> None:
        """
        Start the pipeline processing.
        
        Args:
            **kwargs: Additional arguments for pipeline configuration
        """
        mode = self._detect_pipeline_mode()
        logger.info(f"Starting pipeline in {mode} mode")
        
        if self._is_realtime_mode:
            if self._realtime_model:
                await self._realtime_model.connect()
                
                if isinstance(self.llm, RealtimeLLMAdapter):
                    self.llm.on_user_speech_started(lambda data: self._on_user_speech_started_realtime(data))
                    self.llm.on_user_speech_ended(lambda data: asyncio.create_task(self._on_user_speech_ended_realtime(data)))
                    self.llm.on_agent_speech_started(lambda data: asyncio.create_task(self._on_agent_speech_started_realtime(data)))
                    self.llm.on_agent_speech_ended(lambda data: self._on_agent_speech_ended_realtime(data))
                    self.llm.on_transcription(self._on_realtime_transcription)
        else:
            if self.orchestrator:
                await self.orchestrator.start()
    
    async def send_message(self, message: str, handle: UtteranceHandle) -> None:
        """
        Send a message to the pipeline.
        
        Args:
            message: Message text to send
            handle: Utterance handle to track
        """
        self._current_utterance_handle = handle
        
        if self._is_realtime_mode:
            if isinstance(self.llm, RealtimeLLMAdapter):
                self.llm.current_utterance = handle
                try:
                    await self.llm.send_message(message)
                except Exception as e:
                    logger.error(f"Error sending message: {e}")
                    handle._mark_done()
        else:
            if self.orchestrator:
                await self.orchestrator.say(message, handle)
            else:
                logger.warning("No orchestrator available")
                handle._mark_done()
    
    async def send_text_message(self, message: str) -> None:
        """
        Send a text message (for A2A or text-only scenarios).
        
        Args:
            message: Text message to send
        """
        if self._is_realtime_mode:
            if isinstance(self.llm, RealtimeLLMAdapter):
                await self.llm.send_text_message(message)
        else:
            if self.orchestrator:
                await self.orchestrator.process_text(message)
    
    async def on_audio_delta(self, audio_data: bytes) -> None:
        """
        Handle incoming audio data from the user.
        
        Args:
            audio_data: Raw audio bytes
        """
        if self._is_realtime_mode:
            if isinstance(self.llm, RealtimeLLMAdapter):
                await self.llm.handle_audio_input(audio_data)
        else:
            if self.orchestrator:
                await self.orchestrator.process_audio(audio_data)
    
    async def on_video_delta(self, video_data: av.VideoFrame) -> None:
        """
        Handle incoming video data from the user.
        
        Args:
            video_data: Video frame
        """
        if not self.vision:
            return
        
        if self._vision_lock.locked():
            return
        
        # Process through vision_frame hook if available
        if self.hooks and self.hooks.has_vision_frame_hooks():
            async def frame_stream():
                yield video_data
            
            processed_stream = self.hooks.process_vision_frame(frame_stream())
            async for processed_frame in processed_stream:
                video_data = processed_frame
        
        self._recent_frames.append(video_data)
        if len(self._recent_frames) > self._max_frames_buffer:
            self._recent_frames.pop(0)
        
        if self._is_realtime_mode:
            if isinstance(self.llm, RealtimeLLMAdapter):
                await self.llm.handle_video_input(video_data)
    
    def get_latest_frames(self, num_frames: int = 1) -> list[av.VideoFrame]:
        """
        Get the latest video frames from the pipeline.
        
        Args:
            num_frames: Number of frames to retrieve (default: 1, max: 5)
            
        Returns:
            List of VideoFrame objects
        """
        if not self.vision:
            logger.warning("Vision not enabled")
            return []
        
        num_frames = max(1, min(num_frames, self._max_frames_buffer))
        
        if not self._recent_frames:
            return []
        
        return self._recent_frames[-num_frames:]
    
    def interrupt(self) -> None:
        """Interrupt the pipeline"""
        if self._is_realtime_mode:
            if self._realtime_model:
                if self._realtime_model.current_utterance and not self._realtime_model.current_utterance.is_interruptible:
                    logger.info("Interruption disabled for current utterance")
                    return
                asyncio.create_task(self._realtime_model.interrupt())
            
            if self.avatar and hasattr(self.avatar, 'interrupt'):
                asyncio.create_task(self.avatar.interrupt())
            
            if self._current_utterance_handle and not self._current_utterance_handle.done():
                if self._current_utterance_handle.is_interruptible:
                    self._current_utterance_handle.interrupt()
        else:
            if self.orchestrator:
                asyncio.create_task(self.orchestrator.interrupt())
            
            if self.avatar and hasattr(self.avatar, 'interrupt'):
                asyncio.create_task(self.avatar.interrupt())
    
    async def reply_with_context(
        self,
        instructions: str,
        wait_for_playback: bool,
        handle: UtteranceHandle,
        frames: list[av.VideoFrame] | None = None
    ) -> None:
        """
        Generate a reply using instructions and current chat context.
        
        Args:
            instructions: Instructions to add to context
            wait_for_playback: If True, wait for playback to complete
            handle: Utterance handle
            frames: Optional video frames for vision
        """
        self._current_utterance_handle = handle
        
        if self._is_realtime_mode:
            if isinstance(self.llm, RealtimeLLMAdapter):
                self.llm.current_utterance = handle
                
                if frames and hasattr(self.llm, 'send_message_with_frames'):
                    async with self._vision_lock:
                        await self.llm.send_message_with_frames(instructions, frames)
                else:
                    await self.llm.send_text_message(instructions)
        else:
            if self.orchestrator:
                await self.orchestrator.reply_with_context(instructions, wait_for_playback, handle, frames)
            else:
                logger.warning("No orchestrator available")
                handle._mark_done()
    
    def _on_user_speech_started_realtime(self, data: dict) -> None:
        """Handle user speech started in realtime mode"""
        self._notify_speech_started()
        if self.agent and self.agent.session:
            from .utils import UserState, AgentState
            self.agent.session._emit_user_state(UserState.SPEAKING)
            self.agent.session._emit_agent_state(AgentState.LISTENING)
    
    async def _on_user_speech_ended_realtime(self, data: dict) -> None:
        """Handle user speech ended in realtime mode"""
        if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
            await self.agent.session.start_thinking_audio()
    
    async def _on_agent_speech_started_realtime(self, data: dict) -> None:
        """Handle agent speech started in realtime mode"""
        if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
            await self.agent.session.stop_thinking_audio()
    
    def _on_agent_speech_ended_realtime(self, data: dict) -> None:
        """Handle agent speech ended in realtime mode"""
        if self._current_utterance_handle and not self._current_utterance_handle.done():
            self._current_utterance_handle._mark_done()
        
        if self._realtime_model:
            self._realtime_model.current_utterance = None
        
        if self.agent and hasattr(self.agent, 'on_agent_speech_ended'):
            self.agent.on_agent_speech_ended(data)
    
    def _on_realtime_transcription(self, data: dict) -> None:
        """Handle realtime model transcription"""
        self.emit("realtime_model_transcription", data)
        
        if self.voice_mail_detector:
            pass
    
    def set_voice_mail_detector(self, detector: VoiceMailDetector | None) -> None:
        """Set voicemail detector"""
        self.voice_mail_detector = detector
        if self.orchestrator:
            self.orchestrator.set_voice_mail_detector(detector)
    
    async def process_text(self, text: str) -> None:
        """
        Process text input directly (bypasses STT).
        
        Args:
            text: User text input
        """
        if self._is_realtime_mode:
            if isinstance(self.llm, RealtimeLLMAdapter):
                await self.llm.send_text_message(text)
        else:
            if self.orchestrator:
                await self.orchestrator.process_text(text)
            else:
                logger.warning("No orchestrator available for text processing")
    
    async def inject_text_to_llm(self, text: str) -> None:
        """
        Inject processed text into LLM for generation (hybrid mode).
        
        Args:
            text: Processed text to send to LLM
        """
        if self.orchestrator:
            await self.orchestrator.inject_text_to_llm(text)
        else:
            logger.warning("inject_text_to_llm only available in cascading mode")
    
    async def inject_text_to_tts(self, text: str) -> None:
        """
        Inject text directly to TTS (bypassing LLM) for hybrid scenarios.
        
        Args:
            text: Text to synthesize
        """
        if self.orchestrator:
            await self.orchestrator.inject_text_to_tts(text)
        else:
            logger.warning("inject_text_to_tts only available in cascading mode")
    
    def get_component_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get component configurations"""
        configs: Dict[str, Dict[str, Any]] = {}
        
        for comp_name, comp in [
            ("stt", self.stt),
            ("llm", self.llm if not self._is_realtime_mode else self._realtime_model),
            ("tts", self.tts),
            ("vad", self.vad),
            ("eou", self.turn_detector),
        ]:
            if comp:
                try:
                    configs[comp_name] = {
                        k: v for k, v in comp.__dict__.items()
                        if not k.startswith("_") and not callable(v)
                    }
                except Exception:
                    configs[comp_name] = {}
        
        return configs
    
    async def cleanup(self) -> None:
        """Cleanup pipeline resources"""
        logger.info("Cleaning up pipeline")
        
        if self._is_realtime_mode:
            if self._realtime_model:
                await self._realtime_model.aclose()
                self._realtime_model = None
            
            if self.avatar:
                if hasattr(self.avatar, 'cleanup'):
                    await self.avatar.cleanup()
                elif hasattr(self.avatar, 'aclose'):
                    await self.avatar.aclose()
                self.avatar = None
            
            if self.denoise:
                await self.denoise.aclose()
                self.denoise = None
        else:
            if self.stt:
                await self.stt.aclose()
                self.stt = None
            if self.llm and not isinstance(self.llm, RealtimeLLMAdapter):
                await self.llm.aclose()
                self.llm = None
            if self.tts:
                await self.tts.aclose()
                self.tts = None
            if self.vad:
                await self.vad.aclose()
                self.vad = None
            if self.turn_detector:
                await self.turn_detector.aclose()
                self.turn_detector = None
            if self.denoise:
                await self.denoise.aclose()
                self.denoise = None
            if self.avatar:
                if hasattr(self.avatar, 'cleanup'):
                    await self.avatar.cleanup()
                elif hasattr(self.avatar, 'aclose'):
                    await self.avatar.aclose()
                self.avatar = None
            if self.orchestrator:
                await self.orchestrator.cleanup()
                self.orchestrator = None
        
        self.agent = None
        self.vision = False
        self.loop = None
        self.audio_track = None
        self._wake_up_callback = None
        self._recent_frames = []
        self._current_utterance_handle = None
        
        logger.info("Pipeline cleaned up")
    
    async def leave(self) -> None:
        """Leave the pipeline"""
        await self.cleanup()