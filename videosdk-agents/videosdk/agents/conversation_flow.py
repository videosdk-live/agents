from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Literal, AsyncIterator
import time
import json
import asyncio
from .event_emitter import EventEmitter
from .stt.stt import STT, STTResponse
from .llm.llm import LLM
from .llm.chat_context import ChatRole
from .tts.tts import TTS
from .stt.stt import SpeechEventType
from .agent import Agent
from .event_bus import global_event_emitter
from .vad import VAD, VADResponse, VADEventType
from .eou import EOU
from .denoise import Denoise

class ConversationFlow(EventEmitter[Literal["transcription"]], ABC):
    """
    Manages the conversation flow by listening to transcription events.
    """

    def __init__(self, agent: Agent, stt: STT | None = None, llm: LLM | None = None, tts: TTS | None = None, vad: VAD | None = None, turn_detector: EOU | None = None, denoise: Denoise | None = None) -> None:
        """Initialize conversation flow with event emitter capabilities"""
        super().__init__() 
        self.transcription_callback: Callable[[STTResponse], Awaitable[None]] | None = None
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.vad = vad
        self.turn_detector = turn_detector
        self.agent = agent   
        self.denoise = denoise
        
        self.stt_lock = asyncio.Lock()
        self.llm_lock = asyncio.Lock()
        self.tts_lock = asyncio.Lock()
        
        self.user_speech_callback: Callable[[], None] | None = None
        if self.stt:
            self.stt.on_stt_transcript(self.on_stt_transcript)
        if self.vad:
            self.vad.on_vad_event(self.on_vad_event)
        
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
        if self.denoise:
            audio_data = await self.denoise.denoise(audio_data)
        if self.stt:
            async with self.stt_lock:
                await self.stt.process_audio(audio_data)
        if self.vad:
            await self.vad.process_audio(audio_data)
                        
    async def on_vad_event(self, vad_response: VADResponse) -> None:
        if vad_response.event_type == VADEventType.START_OF_SPEECH:
            self.on_speech_started()
        elif vad_response.event_type == VADEventType.END_OF_SPEECH:
            self.on_speech_stopped()
            
    async def on_stt_transcript(self, stt_response: STTResponse) -> None:
        if stt_response.event_type == SpeechEventType.FINAL:
            user_text = stt_response.data.text
            
            self.agent.chat_context.add_message(
                role=ChatRole.USER,
                content=user_text
            )
            
            await self.on_turn_start(user_text)
            
            if self.turn_detector and self.turn_detector.detect_end_of_utterance(self.agent.chat_context):
                if self.tts:
                    async with self.tts_lock:
                        await self._synthesize_with_tts(self.run(user_text))
                else:
                    async for _ in self.run(user_text):
                        pass
            if not self.turn_detector:
                if self.tts:
                    async with self.tts_lock:
                        await self._synthesize_with_tts(self.run(user_text))
                else:
                    async for _ in self.run(user_text):
                        pass
                        
            await self.on_turn_end()
            
    async def process_with_llm(self) -> AsyncIterator[str]:
        """
        Process the current chat context with LLM and yield response chunks.
        This method can be called by user implementations to get LLM responses.
        """
        async with self.llm_lock:
            if not self.llm:
                return
            full_response = ""
            prev_content_length = 0
            
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
                        continue
                        
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
                                    yield new_content
                                full_response = new_resp.content
                                prev_content_length = len(new_resp.content)
                        except Exception as e:
                            print(f"Error executing function {func_call['name']}: {e}")
                            continue
                else:
                    new_content = llm_chunk_resp.content[prev_content_length:]
                    if new_content: 
                        yield new_content
                    full_response = llm_chunk_resp.content
                    prev_content_length = len(llm_chunk_resp.content)
            
            if full_response:
                self.agent.chat_context.add_message(
                    role=ChatRole.ASSISTANT,
                    content=full_response
                )
                            
    async def say(self, message: str) -> None:
        async with self.tts_lock:
            if self.tts:
                await self.tts.synthesize(message)
                
    async def process_text_input(self, text: str) -> None:
        """
        Process text input directly (for A2A communication).
        This bypasses STT and directly processes the text through the LLM.
        """
        self.agent.chat_context.add_message(
            role=ChatRole.USER,
            content=text
        )
        
        full_response = ""
        async for response_chunk in self.process_with_llm():
            full_response += response_chunk
        
        if full_response:
            global_event_emitter.emit("text_response", {"text": full_response})
    
    
    async def run(self, transcript: str) -> AsyncIterator[str]:
        """
        Main conversation loop: handle a user turn.
        Users should implement this method to preprocess transcripts and yield response chunks.
        """
        async for response in self.process_with_llm():
            yield response
    
    async def on_turn_start(self, transcript: str) -> None:
        """Called at the start of a user turn."""
        pass
    
    async def on_turn_end(self) -> None:
        """Called at the end of a user turn."""
        pass

    def on_speech_started(self) -> None:
        if self.user_speech_callback:
            self.user_speech_callback()
        if self.tts:
            asyncio.create_task(self._interrupt_tts())

    async def _interrupt_tts(self) -> None:
        async with self.tts_lock:
            await self.tts.interrupt()
    
    def on_speech_stopped(self) -> None:
        pass

    async def _synthesize_with_tts(self, response_gen: AsyncIterator[str]) -> None:
        """
        Stream LLM response chunks to TTS with buffering to sentences or pauses.
        Synthesizes at natural breaks (., !, ?, ;).
        """
        buffer = ""
        delimiters = {'.', '!', '?', ';'}
        
        async for chunk in response_gen:
            buffer += chunk
            while buffer:
                delimiter_pos = min((buffer.find(d) for d in delimiters if buffer.find(d) != -1), default=-1)
                if delimiter_pos != -1:
                    to_speak = buffer[:delimiter_pos + 1]
                    buffer = buffer[delimiter_pos + 1:].lstrip() 
                    
                    await self.tts.synthesize(to_speak)
                else:
                    break
        
        if buffer:
            await self.tts.synthesize(buffer)