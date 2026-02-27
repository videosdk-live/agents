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
from .speech_generation import SpeechGeneration
from .stt.stt import STT
from .llm.llm import LLM
from .tts.tts import TTS
from .vad import VAD
from .eou import EOU
from .denoise import Denoise
from .voice_mail_detector import VoiceMailDetector
from .job import get_current_job_context
from .utils import PipelineMode, RealtimeMode,PipelineConfig, build_pipeline_config
from .metrics import metrics_collector

logger = logging.getLogger(__name__)

from .pipeline_utils import (
    NO_CHANGE, 
    cleanup_pipeline, 
    check_mode_shift, 
    swap_component_in_orchestrator, 
    swap_llm, 
    swap_tts, 
    register_stt_transcript_listener
)


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


@dataclass
class RealtimeConfig:
    """Configuration for realtime model behavior"""
    mode: Literal["full_s2s", "hybrid_stt", "hybrid_tts", "llm_only"] | None = None
    response_modalities: List[str] | None = None


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
        realtime_config: RealtimeConfig | None = None,
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
        
        # Realtime configuration
        self.realtime_config = realtime_config

        # Detect and handle realtime models
        self.llm: LLM | RealtimeLLMAdapter | None = None
        self._realtime_model: RealtimeBaseModel | None = None

        if isinstance(llm, RealtimeBaseModel):
            self._realtime_model = llm
            self.llm = RealtimeLLMAdapter(llm)
        else:
            self.llm = llm

        self.config: PipelineConfig = build_pipeline_config(
            stt=self.stt,
            llm=self.llm,
            tts=self.tts,
            vad=self.vad,
            turn_detector=self.turn_detector,
            avatar=self.avatar,
            denoise=self.denoise,
            realtime_model=self._realtime_model,
            realtime_config_mode=(
                self.realtime_config.mode if self.realtime_config and self.realtime_config.mode else None
            ),
        )
        
        # Configuration
        self.eou_config = eou_config or EOUConfig()
        self.interrupt_config = interrupt_config or InterruptConfig()
        
        # Pipeline state
        self.agent = None
        self.orchestrator: PipelineOrchestrator | None = None
        self.speech_generation: SpeechGeneration | None = None
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
    
    @property
    def realtime_mode(self) -> str | None:
        """Backwards-compatible alias. Returns the string value or None."""
        return self.config.realtime_mode.value if self.config.realtime_mode else None

    @property
    def _is_realtime_mode(self) -> bool:
        """Backwards-compatible alias."""
        return self.config.is_realtime

    def _configure_text_only_mode(self) -> None:
        """Configure realtime model for text-only output (provider-specific)"""
        if not self._realtime_model or not hasattr(self._realtime_model, 'config'):
            return
        
        config = self._realtime_model.config
        
        if hasattr(config, 'response_modalities'):
            config.response_modalities = ["TEXT"]
            logger.info("Configured Gemini for TEXT-only mode")
        
        elif hasattr(config, 'modalities'):
            config.modalities = ["text"]
            logger.info("Configured OpenAI for text-only mode")
        
        else:
            logger.warning(f"Unknown realtime provider config, could not set text-only mode")
    
    def _wrap_async(self, async_func):
        """Wrap an async function to be compatible with EventEmitter's sync-only handlers"""
        def sync_wrapper(*args, **kwargs):
            asyncio.create_task(async_func(*args, **kwargs))
        return sync_wrapper
    
    async def _on_transcript_ready_hybrid_stt(self, data: dict) -> None:
        """Handle transcript in hybrid STT mode (external STT + KB + realtime LLM+TTS)"""
        transcript = data["text"]
        
        if not self.agent:
            logger.warning("No agent available for transcript processing")
            return
        
        logger.info(f"Processing transcript in hybrid_stt mode: {transcript}")
        
        enriched_text = transcript
        if self.agent.knowledge_base:
            try:
                logger.info(f"Querying knowledge base for: {transcript[:100]}...")
                kb_context = await self.agent.knowledge_base.process_query(transcript)
                if kb_context:
                    enriched_text = f"{kb_context}\n\nUser: {transcript}"
                    logger.info(f"Enriched transcript with KB context: {kb_context[:100]}...")
                else:
                    logger.info("No KB context returned")
            except Exception as e:
                logger.error(f"Error processing KB query: {e}", exc_info=True)
        
        if isinstance(self.llm, RealtimeLLMAdapter):
            try:
                await self.llm.send_text_message(enriched_text)
                logger.info("Sent enriched text to realtime model")
            except Exception as e:
                logger.error(f"Error sending text to realtime model: {e}")
    
    async def _on_realtime_transcription_hybrid_tts(self, data: dict) -> None:
        """Handle transcription from realtime model in hybrid TTS mode"""
        role = data.get("role")
        text = data.get("text")
        is_final = data.get("is_final", False)
        
        if role not in ["agent", "assistant", "model"] or not is_final or not text:
            return
        
        logger.info(f"Intercepted final text from realtime model (hybrid_tts): {text[:100]}...")
        
        if self.speech_generation:
            try:
                await self.speech_generation.synthesize(text)
                logger.info("Sent transcribed text to external TTS")
            except Exception as e:
                logger.error(f"Error synthesizing with external TTS: {e}")
    
    def on(
        self, 
        event: Literal["speech_in", "speech_out", "stt", "llm","tts","agent_response", "vision_frame", "user_turn_start", "user_turn_end", "agent_turn_start", "agent_turn_end", "content_generated"] | str,
        callback: Callable | None = None
    ) -> Callable:
        """
        Register a listener for pipeline events or a hook for processing stages.
        
        Can be used as a decorator or with a callback.
        
        Supported hooks (decorator only):
        - stt: Process user transcript after STT, before LLM (or stream STT hook)
        - tts: Stream TTS hook (text -> audio)
        - llm: Control LLM invocation (can bypass with direct response)
        - agent_response: Process agent response after LLM, before TTS
        - vision_frame: Process video frames when vision is enabled (async iterator)
        - user_turn_start: Called when user turn starts
        - user_turn_end: Called when user turn ends
        - agent_turn_start: Called when agent processing starts
        - agent_turn_end: Called when agent finishes speaking
        - content_generated: Called when LLM generates content (receives dict with "text" key)
        
        Supported events (listener):
        - transcript_ready
        - synthesis_complete
        - error
        
        Examples:
            @pipeline.on("content_generated")
            async def on_content(data):
                print(f"LLM generated: {data['text']}")
        """
        if event in ["stt", "tts", "llm", "agent_response", "vision_frame", "user_turn_start", "user_turn_end", "agent_turn_start", "agent_turn_end", "content_generated"]:
            return self.hooks.on(event)(callback) if callback else self.hooks.on(event)
            
        return super().on(event, callback)
    
    def _setup_error_handlers(self) -> None:
        """Setup error handlers for all components"""
        if self.stt:
            self.stt.on("error", lambda *args: self.on_component_error("STT", args[0] if args else "Unknown error"))
        if self.llm and not self.config.is_realtime:
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
        metrics_collector.add_error(source, error_data)
        self.emit("error", {"source": source, "error": str(error_data)})
    
    def get_session_metrics_snapshot(self) -> dict:
        """Return dict suitable for populating SessionMetrics fields."""
        return {
            "pipeline_type": self.config.pipeline_mode.value,
            "components": self.config.component_names,
        }

    def set_agent(self, agent: Any) -> None:
        """Set the agent for this pipeline"""
        self.agent = agent

        # Configure metrics with pipeline info
        metrics_collector.configure_pipeline(
            pipeline_mode=self.config.pipeline_mode,
            realtime_mode=self.config.realtime_mode,
            active_components=self.config.active_components,
        )
        metrics_collector.set_eou_config(self.eou_config)
        metrics_collector.set_interrupt_config(self.interrupt_config)

        if self.config.realtime_mode in (RealtimeMode.HYBRID_STT, RealtimeMode.LLM_ONLY):
            logger.info(f"Creating orchestrator for {self.config.realtime_mode.value} mode")
            self.orchestrator = PipelineOrchestrator(
                agent=agent,
                stt=self.stt,
                llm=None, 
                tts=None,  
                vad=self.vad,
                turn_detector=self.turn_detector,
                denoise=self.denoise,
                avatar=None,
                mode=self.eou_config.mode,
                min_speech_wait_timeout=self.eou_config.min_max_speech_wait_timeout,
                interrupt_mode=self.interrupt_config.mode,
                interrupt_min_duration=self.interrupt_config.interrupt_min_duration,
                interrupt_min_words=self.interrupt_config.interrupt_min_words,
                false_interrupt_pause_duration=self.interrupt_config.false_interrupt_pause_duration,
                resume_on_false_interrupt=self.interrupt_config.resume_on_false_interrupt,
                conversational_graph=None, 
                max_context_items=self.max_context_items,
                voice_mail_detector=self.voice_mail_detector,
                hooks=self.hooks,
            )
            

            self.orchestrator.on("transcript_ready", self._wrap_async(self._on_transcript_ready_hybrid_stt))
            logger.info("Registered hybrid_stt event listener on orchestrator")
            
            if isinstance(self.llm, RealtimeLLMAdapter):
                self.llm.set_agent(agent)
        
        elif self.config.realtime_mode == RealtimeMode.HYBRID_TTS:
            logger.info("Setting up hybrid_tts mode: realtime STT+LLM + external TTS")
            
            if hasattr(self._realtime_model, 'audio_track'):
                self._realtime_model.audio_track = None
                logger.info("Disconnected realtime model audio track (external TTS will be used)")
            
            if self.tts:
                self.speech_generation = SpeechGeneration(
                    agent=agent,
                    tts=self.tts,
                    avatar=self.avatar,
                    hooks=self.hooks,
                )
           
            if self._realtime_model and not hasattr(self, '_hybrid_tts_listeners_registered'):
                self._hybrid_tts_listeners_registered = True
                self._realtime_model.on("realtime_model_transcription", 
                    self._wrap_async(self._on_realtime_transcription_hybrid_tts))
                logger.info("Registered hybrid_tts event listener for realtime_model_transcription")
            
            if isinstance(self.llm, RealtimeLLMAdapter):
                self.llm.set_agent(agent)
        
        elif self.config.realtime_mode == RealtimeMode.FULL_S2S:
            if isinstance(self.llm, RealtimeLLMAdapter):
                self.llm.set_agent(agent)
        
        elif not self.config.is_realtime:
            if self.conversational_graph:
                self.conversational_graph.compile()
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
    
    def _set_loop_and_audio_track(self, loop: asyncio.AbstractEventLoop, audio_track: CustomAudioStreamTrack) -> None:
        """Set the event loop and configure components"""
        self.loop = loop
        self.audio_track = audio_track
        self._configure_components()
    
    async def change_pipeline(
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
        realtime_config: RealtimeConfig | None = None
        ) -> None:
        """
        Dynamically change pipeline configuration and components.
        
        This method allows switching between different modes (Realtime, Cascading, Hybrid)
        and updating individual components.
        """
        logger.info("Changing pipeline configuration...")
        
        # 1.Cleanup current execution
        await cleanup_pipeline(self, llm_changing=True)

        # 2.Update components
        await swap_component_in_orchestrator(
            self, 'stt', stt, 'speech_understanding', 'stt_lock', 
            register_stt_transcript_listener
        )
        await swap_tts(self, tts)
        await swap_component_in_orchestrator(self, 'vad', vad, 'speech_understanding')
        await swap_component_in_orchestrator(self, 'turn_detector', turn_detector, 'speech_understanding', 'turn_detector_lock')
        await swap_component_in_orchestrator(self, 'denoise', denoise, 'speech_understanding', 'denoise_lock')
            
        if self.avatar and self.avatar != avatar: await self.avatar.aclose()
        self.avatar = avatar

        # Update configs
        if eou_config is not None: self.eou_config = eou_config
        if interrupt_config is not None: self.interrupt_config = interrupt_config
        if max_context_items is not None: self.max_context_items = max_context_items
        if voice_mail_detector is not None: self.voice_mail_detector = voice_mail_detector
        if realtime_config is not None: self.realtime_config = realtime_config   
        if conversational_graph is not None:
            self.conversational_graph = conversational_graph
            if self.conversational_graph and hasattr(self.conversational_graph, 'compile'):
                self.conversational_graph.compile()
            
        # Update LLM / Realtime Model
        await swap_llm(self, llm)

        # 3. REBOOT: Detect mode and restart
        self.config = build_pipeline_config(
            stt=self.stt,
            llm=self.llm,
            tts=self.tts,
            vad=self.vad,
            turn_detector=self.turn_detector,
            avatar=self.avatar,
            denoise=self.denoise,
            realtime_model=self._realtime_model,
            realtime_config_mode=(
                self.realtime_config.mode if self.realtime_config and self.realtime_config.mode else None
            ),
        )
        new_mode = self.config.pipeline_mode.value
        logger.info(f"New pipeline mode: {new_mode}")
        
        if self.agent:
            logger.info("Restarting pipeline with updated components")
            self.set_agent(self.agent)
            
        self._configure_components()
        await self.start()

    async def change_component(
        self,
        stt: STT | None = NO_CHANGE,
        llm: LLM | RealtimeBaseModel | None = NO_CHANGE,
        tts: TTS | None = NO_CHANGE,
        vad: VAD | None = NO_CHANGE,
        turn_detector: EOU | None = NO_CHANGE,
        denoise: Denoise | None = NO_CHANGE,
        ) -> None:
        """Dynamically change components.
        This will close the old components and set the new ones.
        """
        logger.info("Changing pipeline component(s)...")
        

        # 0 Change components only if present earlier
        validation_map = {
            'STT': (stt, self.stt),
            'TTS': (tts, self.tts),
            'LLM': (llm, self.llm),
            'VAD': (vad, self.vad),
            'Turn Detector': (turn_detector, self.turn_detector),
            'Denoise': (denoise, self.denoise)
        }

        for name, (new_val, current_val) in validation_map.items():
            if new_val is not NO_CHANGE and current_val is None:
                raise ValueError(
                    f"Cannot change component '{name}' because it is not present in the current pipeline. "
                    "Use change_pipeline() for full reconfiguration."
                )

        logger.info(f"Performing swap in {self.config.pipeline_mode.value} mode")

        # Detect pipeline mode shift
        mode_shift = check_mode_shift(self, llm, stt, tts)
        if mode_shift:
            logger.info("Component change triggers mode shift. Delegating to change_pipeline for full reconfiguration.")
            
            # Resolve sentinels to current values for resettlement
            target_stt = self.stt if stt is NO_CHANGE else stt
            target_tts = self.tts if tts is NO_CHANGE else tts
            target_vad = self.vad if vad is NO_CHANGE else vad
            target_turn_detector = self.turn_detector if turn_detector is NO_CHANGE else turn_detector
            target_denoise = self.denoise if denoise is NO_CHANGE else denoise

            if llm is NO_CHANGE:
                target_llm = self._realtime_model if self._realtime_model else self.llm
            else:
                target_llm = llm

            await self.change_pipeline(
                stt=target_stt,
                llm=target_llm,
                tts=target_tts,
                vad=target_vad,
                turn_detector=target_turn_detector,
                denoise=target_denoise
            )
            return

        if stt is not NO_CHANGE and self.stt != stt:
            await swap_component_in_orchestrator(
                self, 'stt', stt, 'speech_understanding', 'stt_lock', 
                register_stt_transcript_listener
            )

        if llm is not NO_CHANGE and self.llm != llm:
            await swap_llm(self, llm)

        if tts is not NO_CHANGE and self.tts != tts:
            await swap_tts(self, tts)


        if vad is not NO_CHANGE and self.vad != vad:
            await swap_component_in_orchestrator(self, 'vad', vad, 'speech_understanding')


        if turn_detector is not NO_CHANGE and self.turn_detector != turn_detector:
            await swap_component_in_orchestrator(self, 'turn_detector', turn_detector, 'speech_understanding', 'turn_detector_lock')

        if denoise is not NO_CHANGE and self.denoise != denoise:
            await swap_component_in_orchestrator(self, 'denoise', denoise, 'speech_understanding', 'denoise_lock')

        # 3. REBOOT: Rebuild config with updated components
        self.config = build_pipeline_config(
            stt=self.stt,
            llm=self.llm,
            tts=self.tts,
            vad=self.vad,
            turn_detector=self.turn_detector,
            avatar=self.avatar,
            denoise=self.denoise,
            realtime_model=self._realtime_model,
            realtime_config_mode=(
                self.realtime_config.mode if self.realtime_config and self.realtime_config.mode else None
            ),
        )
        new_mode = self.config.pipeline_mode.value
        logger.info(f"New pipeline mode: {new_mode}")

        return

    def _configure_components(self) -> None:
        """Configure pipeline components with loop and audio track"""
        if not self.loop:
            return
        
        job_context = get_current_job_context()
        
        if job_context and job_context.room:
            requested_vision = getattr(job_context.room, 'vision', False)
            self.vision = requested_vision
            
            if requested_vision and self.config.is_realtime:
                model_name = self._realtime_model.__class__.__name__ if self._realtime_model else "Unknown"
                if model_name not in ["GeminiRealtime", "OpenAIRealtime"]:
                    logger.warning(f"Vision requested but {model_name} doesn't support video input. Disabling vision.")
                    self.vision = False
        
        if not self.config.is_realtime and self.tts:
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
        
        if self.config.is_realtime and self._realtime_model:
            self._realtime_model.loop = self.loop
            
            audio_track = None
            if self.avatar and job_context and job_context.room:
                audio_track = getattr(job_context.room, 'agent_audio_track', None) or job_context.room.audio_track
            elif self.audio_track:
                audio_track = self.audio_track
            
            if self.config.realtime_mode == RealtimeMode.HYBRID_TTS and self.tts:
                self._realtime_model.audio_track = None  
                self.tts.audio_track = audio_track  
                self.tts.loop = self.loop
                logger.info("hybrid_tts: Audio track connected to external TTS, disconnected from realtime model")
                
                if self.tts.audio_track and hasattr(self.tts.audio_track, 'set_pipeline_hooks'):
                    self.tts.audio_track.set_pipeline_hooks(self.hooks)
            else:
                self._realtime_model.audio_track = audio_track
                
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
        logger.info(
            f"Starting pipeline | mode={self.config.pipeline_mode.value} "
            f"| realtime={self.config.realtime_mode.value if self.config.realtime_mode else 'none'} "
            f"| components={self.config.component_names}"
        )
        
        if self.config.is_realtime:
            if self._realtime_model:
                await self._realtime_model.connect()
                
                if isinstance(self.llm, RealtimeLLMAdapter):
                    self.llm.on_user_speech_started(lambda data: self._on_user_speech_started_realtime(data))
                    self.llm.on_user_speech_ended(lambda data: asyncio.create_task(self._on_user_speech_ended_realtime(data)))
                    self.llm.on_agent_speech_started(lambda data: asyncio.create_task(self._on_agent_speech_started_realtime(data)))
                    self.llm.on_agent_speech_ended(lambda data: self._on_agent_speech_ended_realtime(data))
                    self.llm.on_transcription(self._on_realtime_transcription)            
            if self.config.realtime_mode == RealtimeMode.HYBRID_STT and self.orchestrator:
                await self.orchestrator.start()
                logger.info("Started orchestrator for hybrid_stt mode")
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
        
        if self.config.is_realtime:
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
        if self.config.is_realtime:
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
        if self.config.realtime_mode == RealtimeMode.HYBRID_STT and self.orchestrator:
            await self.orchestrator.process_audio(audio_data)
        elif self.config.is_realtime:
            if isinstance(self.llm, RealtimeLLMAdapter):
                await self.llm.handle_audio_input(audio_data)
        else:
            if self.orchestrator:
                await self.orchestrator.process_audio(audio_data)
        
        if not hasattr(self, '_first_audio_logged'):
            self._first_audio_logged = True
            if self.config.realtime_mode == RealtimeMode.HYBRID_STT:
                logger.info("Audio routing: hybrid_stt → orchestrator (external STT)")
            elif self.config.is_realtime:
                logger.info("Audio routing: realtime mode → realtime model")
            else:
                logger.info("Audio routing: traditional mode → orchestrator")
    
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
        
        if self.config.is_realtime:
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
        if self.config.is_realtime:
            if self._realtime_model:
                if self._realtime_model.current_utterance and not self._realtime_model.current_utterance.is_interruptible:
                    logger.info("Interruption disabled for current utterance")
                    return
                asyncio.create_task(self._realtime_model.interrupt())
            
            if self.config.realtime_mode == RealtimeMode.HYBRID_TTS and self.speech_generation:
                asyncio.create_task(self.speech_generation.interrupt())
            
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
        
        if self.config.is_realtime:
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
        metrics_collector.on_user_speech_start()

        if self.config.realtime_mode == RealtimeMode.HYBRID_TTS and self.speech_generation:
            asyncio.create_task(self.speech_generation.interrupt())

        if self.agent and self.agent.session:
            from .utils import UserState, AgentState
            self.agent.session._emit_user_state(UserState.SPEAKING)
            self.agent.session._emit_agent_state(AgentState.LISTENING)
    
    async def _on_user_speech_ended_realtime(self, data: dict) -> None:
        """Handle user speech ended in realtime mode"""
        metrics_collector.on_user_speech_end()
        if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
            await self.agent.session.start_thinking_audio()
    
    async def _on_agent_speech_started_realtime(self, data: dict) -> None:
        """Handle agent speech started in realtime mode"""
        metrics_collector.on_agent_speech_start()
        if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
            await self.agent.session.stop_thinking_audio()
    
    def _on_agent_speech_ended_realtime(self, data: dict) -> None:
        """Handle agent speech ended in realtime mode"""
        metrics_collector.on_agent_speech_end()
        metrics_collector.schedule_turn_complete(timeout=1.0)

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
        if self.config.is_realtime:
            if isinstance(self.llm, RealtimeLLMAdapter):
                await self.llm.send_text_message(text)
        else:
            if self.orchestrator:
                await self.orchestrator.process_text(text)
            else:
                logger.warning("No orchestrator available for text processing")
    
    
    def get_component_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get component configurations"""
        configs: Dict[str, Dict[str, Any]] = {}
        
        for comp_name, comp in [
            ("stt", self.stt),
            ("llm", self.llm if not self.config.is_realtime else self._realtime_model),
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
        
        if self.config.is_realtime:
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