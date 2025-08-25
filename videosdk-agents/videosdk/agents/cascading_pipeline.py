from __future__ import annotations
from typing import Any, Dict, Literal

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

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


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
    ) -> None:
        """
        Initialize the cascading pipeline.

        Args:
            stt: Speech-to-Text processor (optional)
            llm: Language Model processor (optional)
            tts: Text-to-Speech processor (optional)
        """
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.vad = vad
        self.turn_detector = turn_detector
        self.agent = None
        self.conversation_flow = None
        self.avatar = avatar

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
        super().__init__()

    def set_agent(self, agent: Agent) -> None:
        self.agent = agent

    def _configure_components(self) -> None:
        if self.loop and self.tts:
            self.tts.loop = self.loop
            logger.info("TTS loop configured")
            job_context = get_current_job_context()
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
        self.conversation_flow.llm = self.llm
        self.conversation_flow.tts = self.tts
        self.conversation_flow.agent = self.agent
        self.conversation_flow.vad = self.vad
        self.conversation_flow.turn_detector = self.turn_detector
        self.conversation_flow.denoise = self.denoise
        self.conversation_flow.user_speech_callback = self.on_user_speech_started
        if self.conversation_flow.stt:
            self.conversation_flow.stt.on_stt_transcript(
                self.conversation_flow.on_stt_transcript
            )
        if self.conversation_flow.vad:
            self.conversation_flow.vad.on_vad_event(
                self.conversation_flow.on_vad_event)

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

    async def send_message(self, message: str) -> None:
        if self.conversation_flow:
            await self.conversation_flow.say(message)
        else:
            logger.warning("No conversation flow found in pipeline")

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

    def on_user_speech_started(self) -> None:
        """
        Handle user speech started event
        """
        self._notify_speech_started()

    async def cleanup(self) -> None:
        """Cleanup all pipeline components"""
        if self.stt:
            await self.stt.aclose()
        if self.llm:
            await self.llm.aclose()
        if self.tts:
            await self.tts.aclose()

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
