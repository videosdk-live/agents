from __future__ import annotations

from typing import Any, Dict, Literal
import asyncio

from .pipeline import Pipeline
from .event_emitter import EventEmitter
from .llm.llm import LLM
from .stt.stt import STT
from .tts.tts import TTS
from .vad import VAD
from .conversation_flow import ConversationFlow
from .agent import Agent
from .room.room import VideoSDKHandler
from .eou import EOU

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
        turn_detector: EOU | None = None
    ) -> None:
        """
        Initialize the cascading pipeline.
        
        Args:
            stt: Speech-to-Text processor (optional)
            llm: Language Model processor (optional)
            tts: Text-to-Speech processor (optional)
        """
        super().__init__()
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.vad = vad
        self.turn_detector = turn_detector
        self.loop = asyncio.get_event_loop()
        self.room = None
        self.agent = None
        self.conversation_flow = None
        
    def set_agent(self, agent: Agent) -> None:
        self.agent = agent
    
    def set_conversation_flow(self, conversation_flow: ConversationFlow) -> None:
        self.conversation_flow = conversation_flow
        self.conversation_flow.stt = self.stt
        self.conversation_flow.llm = self.llm
        self.conversation_flow.tts = self.tts
        self.conversation_flow.agent = self.agent
        self.conversation_flow.vad = self.vad
        self.conversation_flow.turn_detector = self.turn_detector
        if self.conversation_flow.stt:
            self.conversation_flow.stt.on_stt_transcript(self.conversation_flow.on_stt_transcript)
        if self.conversation_flow.vad:
            self.conversation_flow.vad.on_vad_event(self.conversation_flow.on_vad_event)
        
    async def start(self, **kwargs: Any) -> None:
        if self.conversation_flow:
            await self.conversation_flow.start()
        try:
            meeting_id = kwargs.get('meeting_id')
            name = kwargs.get('name')
            token = kwargs.get('token')
            self.room = VideoSDKHandler(
                    meeting_id=meeting_id,
                    auth_token=token,
                    name=name,
                    pipeline=self,
                    loop=self.loop
                )

            self.room.init_meeting()
            self.tts.loop = self.loop
            self.tts.audio_track = self.room.audio_track

            await self.room.join()
        except Exception as e:
            print(f"Error starting realtime connection: {e}")
            await self.cleanup()
            raise

    async def send_message(self, message: str) -> None:
        await self.conversation_flow.say(message)


    async def on_audio_delta(self, audio_data: bytes) -> None:
        """
        Handle incoming audio data from the user
        """
        await self.conversation_flow.send_audio_delta(audio_data)

    async def cleanup(self) -> None:
        """Cleanup all pipeline components"""
        if self.stt:
            await self.stt.aclose()
        if self.llm:
            await self.llm.aclose()
        if self.tts:
            await self.tts.aclose()
