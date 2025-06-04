from __future__ import annotations

from typing import Awaitable, Callable, Literal
import time
import json
import asyncio
from .event_emitter import EventEmitter
from .stt.stt import STT, STTResponse
from .llm.llm import LLM, LLMResponse
from .llm.chat_context import ChatRole, ChatContext, ChatMessage
from .utils import is_function_tool, get_tool_info, FunctionTool, FunctionToolInfo
from .tts.tts import TTS
from .stt.stt import SpeechEventType
from .agent import Agent
from .event_bus import global_event_emitter

class ConversationFlow(EventEmitter[Literal["transcription"]]):
    """
    Manages the conversation flow by listening to transcription events.
    """

    def __init__(self, agent: Agent, stt: STT | None = None, llm: LLM | None = None, tts: TTS | None = None) -> None:
        """Initialize conversation flow with event emitter capabilities"""
        super().__init__() 
        self.transcription_callback: Callable[[STTResponse], Awaitable[None]] | None = None
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.agent = agent
        
        if self.stt:
            self.stt.on_stt_transcript(self.on_stt_transcript)
        
    async def start(self) -> None:
        global_event_emitter.on("speech_started", self.on_speech_started)
        global_event_emitter.on("speech_stopped", self.on_speech_stopped)

    def on_transcription(self, callback: Callable[[str], None]) -> None:
        """
        Set the callback for transcription events.
        
        Args:
            callback: Function to call when transcription occurs, takes transcribed text as argument
        """
        self.on("transcription_event", lambda data: callback(data["text"]))

    async def send_audio_delta(self, audio_data: bytes) -> None:
        """
        Send audio delta to the STT
        """
        if self.stt:
            await self.stt.process_audio(audio_data)
                        
    async def on_stt_transcript(self, stt_response: STTResponse) -> None:
        if stt_response.event_type == SpeechEventType.FINAL:
            user_text = stt_response.data.text
            self.agent.chat_context.add_message(
                        role=ChatRole.USER,
                        content=user_text
                    )       
            if self.llm:
                full_response = ""
                prev_content_length = 0
                async def stream_new_content():
                    nonlocal full_response, prev_content_length
                    async for llm_chunk_resp in self.llm.chat(
                        self.agent.chat_context,
                        tools=self.agent._tools
                    ):
                        if llm_chunk_resp.metadata and "function_call" in llm_chunk_resp.metadata:
                            func_call = llm_chunk_resp.metadata["function_call"]
                            
                            self.agent.chat_context.add_function_call(
                                name=func_call["name"],
                                arguments=json.dumps(func_call["arguments"]),
                                call_id=func_call.get("call_id", f"call_{int(time.time())}")
                            )
                            try:
                                tool = next(
                                    (t for t in self.agent.tools if hasattr(t, '_tool_info') and t._tool_info.name == func_call["name"]),
                                    None
                                )
                            except Exception as e:
                                print(f"Error while selecting tool: {e}")
                            if tool:
                                try:
                                    result = await tool(**func_call["arguments"])
                                    
                                    self.agent.chat_context.add_function_output(
                                        name=func_call["name"],
                                        output=json.dumps(result),
                                        call_id=func_call.get("call_id", f"call_{int(time.time())}")
                                    )
                                    
                                    async for new_resp in self.llm.chat(self.agent.chat_context):
                                        new_content = new_resp.content[prev_content_length:]
                                        if new_content:
                                            # new_content = await self.agent.process_llm_output(new_content)
                                            yield new_content
                                        full_response = new_resp.content
                                        prev_content_length = len(new_resp.content)
                                except Exception as e:
                                    print(f"Error executing function {func_call['name']}: {e}")
                                    continue
                        else:
                            new_content = llm_chunk_resp.content[prev_content_length:]
                            if new_content: 
                                new_content = await self.agent.process_llm_output(new_content)
                                yield new_content
                            full_response = llm_chunk_resp.content
                            prev_content_length = len(llm_chunk_resp.content)
                if self.tts:
                    await self.tts.synthesize(stream_new_content())
                
                if full_response:
                    self.agent.chat_context.add_message(
                        role=ChatRole.ASSISTANT,
                        content=full_response
                    )
                            
    async def say(self, message: str) -> None:
        if self.tts:
            await self.tts.synthesize(message)

    def on_speech_started(self) -> None:
        if self.tts:
            asyncio.create_task(self.tts.interrupt())

    def on_speech_stopped(self) -> None:
        print("Speech stopped")