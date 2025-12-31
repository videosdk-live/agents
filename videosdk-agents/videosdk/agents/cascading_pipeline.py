from __future__ import annotations
from typing import Any, Dict, Literal, List, Tuple
from dataclasses import dataclass, field

from .pipeline import Pipeline
from .event_emitter import EventEmitter
from .llm.llm import LLM
from .stt.stt import STT
from .tts.tts import TTS
from .vad import VAD
from .conversation_flow import ConversationFlow
from .agent import Agent
from .eou import EOU
from .job import get_current_job_context
from .denoise import Denoise
import logging
import asyncio
import av
from .utterance_handle import UtteranceHandle
    
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

@dataclass
class CascadingPipeline(Pipeline, EventEmitter[Literal["error"]]):
    """
    Cascading pipeline implementation that processes data in sequence (STT -> LLM -> TTS).
    Inherits from Pipeline base class and adds cascade-specific events.
    """

    def __init__(
        self,
        stt: STT | None = None,
        llm: LLM | None = None,
        tts: TTS | None = None,
        vad: VAD | None = None,
        turn_detector: EOU | None = None,
        avatar: Any | None = None,
        denoise: Denoise | None = None,
        eou_config: EOUConfig | None = None,
        interrupt_config: InterruptConfig | None = None,
        conversational_graph: Any | None = None,
        max_context_items: int | None = None,
    ) -> None:
        """
        Initialize the cascading pipeline.

        Args:
            stt: Speech-to-Text processor (optional)
            llm: Language Model processor (optional)
            tts: Text-to-Speech processor (optional)
            vad: Voice Activity Detector (optional)
            turn_detector: Turn Detector (optional)
            avatar: Avatar (optional)
            denoise: Denoise (optional)
            eou_config: End of utterance configuration (optional)
            interrupt_config: Interruption configuration (optional)
            max_context_items: Maximum number of context items to keep (auto-truncates when exceeded)
        """
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.vad = vad
        self.turn_detector = turn_detector
        self.agent = None
        self.conversation_flow = None
        self.avatar = avatar
        self.vision = False
        self.eou_config = eou_config or EOUConfig()
        self.interrupt_config = interrupt_config or InterruptConfig()
        self.max_context_items = max_context_items

        if self.stt:
            self.stt.on(
                "error",
                lambda *args: self.on_component_error(
                    "STT", args[0] if args else "Unknown error"
                ),
            )
        if self.llm:
            self.llm.on(
                "error",
                lambda *args: self.on_component_error(
                    "LLM", args[0] if args else "Unknown error"
                ),
            )
        if self.tts:
            self.tts.on(
                "error",
                lambda *args: self.on_component_error(
                    "TTS", args[0] if args else "Unknown error"
                ),
            )
        if self.vad:
            self.vad.on(
                "error",
                lambda *args: self.on_component_error(
                    "VAD", args[0] if args else "Unknown error"
                ),
            )
        if self.turn_detector:
            self.turn_detector.on(
                "error",
                lambda *args: self.on_component_error(
                    "TURN-D", args[0] if args else "Unknown error"
                ),
            )

        self.denoise = denoise
        self.conversational_graph= conversational_graph
        super().__init__()

    def set_agent(self, agent: Agent) -> None:
        self.agent = agent
        
    def _configure_components(self) -> None:
        if self.loop and self.tts:
            self.tts.loop = self.loop
            logger.info("TTS loop configured")
            job_context = get_current_job_context()

            if job_context and job_context.room:
                requested_vision = getattr(job_context.room, "vision", False)
                self.vision = requested_vision

            if self.avatar and job_context and job_context.room:
                self.tts.audio_track = (
                    getattr(job_context.room, "agent_audio_track", None)
                    or job_context.room.audio_track
                )
                logger.info(
                    f"TTS audio track configured from room (avatar mode)")
            elif hasattr(self, "audio_track"):
                self.tts.audio_track = self.audio_track
                logger.info(f"TTS audio track configured from pipeline")
            else:
                logger.warning(
                    "No audio track available for TTS configuration")

            if self.tts.audio_track:
                logger.info(
                    f"TTS audio track successfully configured: {type(self.tts.audio_track).__name__}"
                )
            else:
                logger.error(
                    "TTS audio track is None - this will prevent audio playback"
                )

    def set_conversation_flow(self, conversation_flow: ConversationFlow) -> None:
        logger.info("Setting conversation flow in pipeline")
        self.conversation_flow = conversation_flow
        self.conversation_flow.stt = self.stt
        self.conversation_flow._update_preemptive_generation_flag()  
        self.conversation_flow.llm = self.llm
        self.conversation_flow.tts = self.tts
        self.conversation_flow.agent = self.agent
        self.conversation_flow.vad = self.vad
        self.conversation_flow.turn_detector = self.turn_detector
        self.conversation_flow.denoise = self.denoise
        self.conversation_flow.avatar = self.avatar
        self.conversation_flow.user_speech_callback = self.on_user_speech_started
        self.conversation_flow.max_context_items = self.max_context_items
        
        if self.max_context_items:
            logger.info(f"Chat Context truncation enabled: max_context_items={self.max_context_items}")
        else:
            logger.info("Chat Context Auto-truncation disabled (max_context_items not set)")
            
        if hasattr(self.conversation_flow, "apply_flow_config"):
            self.conversation_flow.apply_flow_config(
                eou_config=self.eou_config,
                interrupt_config=self.interrupt_config
            )
        if self.conversation_flow.stt:
            self.conversation_flow.stt.on_stt_transcript(
                self.conversation_flow.on_stt_transcript
            )
        if self.conversation_flow.vad:
            self.conversation_flow.vad.on_vad_event(
                self.conversation_flow.on_vad_event)
            
        if self.conversational_graph:
            self.conversational_graph.compile()
            self.conversation_flow.conversational_graph = self.conversational_graph

    async def change_component(
        self,
        stt: STT | None = None,
        llm: LLM | None = None,
        tts: TTS | None = None,
    ) -> None:
        """Dynamically change pipeline components.
        This will close the old components and set the new ones.
        """
        if stt and self.stt:
            async with self.conversation_flow.stt_lock:
                await self.stt.aclose()
                self.stt = stt
                self.conversation_flow.stt = stt
                if self.conversation_flow.stt:
                    self.conversation_flow.stt.on_stt_transcript(
                        self.conversation_flow.on_stt_transcript
                    )
        if llm and self.llm:
            async with self.conversation_flow.llm_lock:
                await self.llm.aclose()
                self.llm = llm
                self.conversation_flow.llm = llm
        if tts and self.tts:
            async with self.conversation_flow.tts_lock:
                await self.tts.aclose()
                self.tts = tts
                self._configure_components()
                self.conversation_flow.tts = tts

    async def start(self, **kwargs: Any) -> None:
        if self.conversation_flow:
            await self.conversation_flow.start()

    async def send_message(self, message: str, handle: UtteranceHandle) -> None:
        if self.conversation_flow:
            await self.conversation_flow.say(message, handle)
        else:
            logger.warning("No conversation flow found in pipeline")
            handle._mark_done()

    async def send_text_message(self, message: str) -> None:
        """
        Send a text message directly to the LLM (for A2A communication).
        This bypasses STT and directly processes the text through the conversation flow.
        """
        if self.conversation_flow:
            await self.conversation_flow.process_text_input(message)
        else:
            await self.send_message(message)

    async def on_audio_delta(self, audio_data: bytes) -> None:
        """
        Handle incoming audio data from the user
        """
        await self.conversation_flow.send_audio_delta(audio_data)

    async def on_video_delta(self, video_data: av.VideoFrame) -> None:
        """Handle incoming video data from the user"""
        if self.vision:
            self._recent_frames.append(video_data)
            if len(self._recent_frames) > self._max_frames_buffer:
                self._recent_frames.pop(0)
        else:
            raise ValueError("Vision not enabled")      
        
    def on_user_speech_started(self) -> None:
        """
        Handle user speech started event
        """
        self._notify_speech_started()
    
    def interrupt(self) -> None:
        """
        Interrupt the pipeline
        """
        if self.conversation_flow:
            asyncio.create_task(self.conversation_flow._interrupt_tts())
        if self.avatar and hasattr(self.avatar, 'interrupt'):
            asyncio.create_task(self.avatar.interrupt())
    
    async def cleanup(self) -> None:
        """Cleanup all pipeline components"""
        logger.info("Cleaning up cascading pipeline")
        if self.stt:
            await self.stt.aclose()
            self.stt = None
        if self.llm:
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
        if self.conversation_flow:
            try:
                await self.conversation_flow.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up conversation flow: {e}")
        
        self.agent = None
        self.vision = False
        self.conversation_flow = None
        self.avatar = None
        logger.info("Cascading pipeline cleaned up")
        await super().cleanup()
    
    async def leave(self) -> None:
        """Leave the cascading pipeline"""
        await self.cleanup()

    def get_component_configs(self) -> dict[str, dict[str, Any]]:
        """Return a dictionary of component configurations (STT, LLM, TTS) with their instance attributes.

        Returns:
            A nested dictionary with keys 'stt', 'llm', 'tts', each containing a dictionary of
            public instance attributes and extracted model information.
        """

        def extract_model_info(config_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Helper to extract model-related info from a dictionary with limited nesting."""
            model_info = {}
            model_keys = [
                "model",
                "model_id",
                "model_name",
                "voice",
                "voice_id",
                "name",
            ]
            try:
                for k, v in config_dict.items():
                    if k in model_keys and v is not None:
                        model_info[k] = v
                    elif k in ["config", "_config", "voice_config"] and isinstance(
                        v, dict
                    ):
                        for nk, nv in v.items():
                            if nk in model_keys and nv is not None:
                                model_info[nk] = nv
                    elif k in ["voice_config", "config"] and hasattr(v, "__dict__"):
                        for nk, nv in v.__dict__.items():
                            if (
                                nk in model_keys
                                and nv is not None
                                and not nk.startswith("_")
                            ):
                                model_info[nk] = nv
            except Exception as e:
                pass
            return model_info

        configs: Dict[str, Dict[str, Any]] = {}
        for comp_name, comp in [
            ("stt", self.stt),
            ("llm", self.llm),
            ("tts", self.tts),
            ("vad", self.vad),
            ("eou", self.turn_detector),
        ]:
            if comp:
                try:
                    configs[comp_name] = {
                        k: v
                        for k, v in comp.__dict__.items()
                        if not k.startswith("_") and not callable(v)
                    }

                    model_info = extract_model_info(comp.__dict__)
                    if model_info:
                        if "model" not in configs[comp_name] and "model" in model_info:
                            configs[comp_name]["model"] = model_info["model"]
                        elif "model" not in configs[comp_name] and "name" in model_info:
                            configs[comp_name]["model"] = model_info["name"]
                        configs[comp_name].update(
                            {
                                k: v
                                for k, v in model_info.items()
                                if k != "model"
                                and k != "name"
                                and k not in configs[comp_name]
                            }
                        )

                    if comp_name == "vad" and "model" not in configs[comp_name]:
                        if hasattr(comp, "_model_sample_rate"):
                            configs[comp_name][
                                "model"
                            ] = f"silero_vad_{comp._model_sample_rate}hz"
                        else:
                            configs[comp_name]["model"] = "silero_vad"
                    elif comp_name == "eou" and "model" not in configs[comp_name]:
                        class_name = comp.__class__.__name__
                        if "VideoSDK" in class_name:
                            configs[comp_name]["model"] = "videosdk_turn_detector"
                        elif "TurnDetector" in class_name:
                            configs[comp_name]["model"] = "turnsense_model"
                        else:
                            configs[comp_name]["model"] = "turn_detector"

                except Exception as e:
                    configs[comp_name] = configs.get(comp_name, {})

        sensitive_keys = ["api_key", "token",
                          "secret", "key", "password", "credential"]
        for comp in configs.values():
            for key in sensitive_keys:
                comp.pop(key, None)
        return configs

    def on_component_error(self, source: str, error_data: Any) -> None:
        """Handle error events from components (STT, LLM, TTS, VAD, TURN-D)"""
        from .metrics import cascading_metrics_collector

        cascading_metrics_collector.add_error(source, str(error_data))
        logger.error(f"[{source}] Component error: {error_data}")

    async def reply_with_context(self, instructions: str, wait_for_playback: bool, handle: UtteranceHandle, frames: list[av.VideoFrame] | None = None) -> None:
        """
        Generate a reply using instructions and current chat context.
        
        Args:
            instructions: Instructions to add to chat context
            wait_for_playback: If True, disable VAD and STT interruptions during response and wait for
            handle: UtteranceHandle to track the utterance
            frames: Optional list of VideoFrame objects to include in the reply
        """
        if self.conversation_flow:
            await self.conversation_flow._process_reply_instructions(instructions, wait_for_playback, handle, frames)
        else:
            logger.warning("No conversation flow found in pipeline")  
