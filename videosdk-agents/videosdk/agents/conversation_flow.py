from __future__ import annotations

from typing import Callable, Literal

from .event_emitter import EventEmitter
from .stt.stt import STT, STTResponse
from .llm.llm import LLM, LLMResponse
from .llm.chat_context import ChatRole, ChatContext, ChatMessage
from .tts.tts import TTS
from .stt.stt import SpeechEventType
from .agent import Agent

class ConversationFlow(EventEmitter[Literal["transcription"]]):
    """
    Manages the conversation flow by listening to transcription events.
    """

    def __init__(self, agent: Agent, stt: STT | None = None, llm: LLM | None = None, tts: TTS | None = None) -> None:
        """Initialize conversation flow with event emitter capabilities"""
        super().__init__() 
        self.transcription_callback: Callable[[str], None] | None = None
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.agent = agent
    async def start(self) -> None:
            
        pass
    
    async def send_audio_delta(self, audio_data: bytes) -> None:
        """
        Send audio delta to the STT
        """
        if self.stt:
            async for stt_response in self.stt.process_audio(audio_data):
                if stt_response.event_type == SpeechEventType.FINAL:
                    user_text = stt_response.data.text
                    print(f"Transcription: {user_text}")
                    
                    self.agent.chat_context.add_message(
                        role=ChatRole.USER,
                        content=user_text
                    )
                    
                    if self.llm:
                        full_response = ""
                        prev_content_length = 0
                        
                        async for llm_chunk_resp in self.llm.chat(self.agent.chat_context):
                            new_content = llm_chunk_resp.content[prev_content_length:]
                            print(f"LLM Response Chunk: {new_content}")
                            full_response = llm_chunk_resp.content
                            prev_content_length = len(llm_chunk_resp.content)
                        
                        if self.tts and full_response:
                            await self.tts.synthesize(full_response)
                        
                        if full_response:
                            self.agent.chat_context.add_message(
                                role=ChatRole.ASSISTANT,
                                content=full_response
                            )

    def on_transcription(self, callback: Callable[[str], None]) -> None:
        """
        Set the callback for transcription events.
        
        Args:
            callback: Function to call when transcription occurs, takes transcribed text as argument
        """
        self.on("transcription_event", lambda data: callback(data["text"]))