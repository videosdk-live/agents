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
from .utils import is_function_tool, get_tool_info
from .tts.tts import TTS
from .stt.stt import SpeechEventType
from .agent import Agent
from .event_bus import global_event_emitter
from .vad import VAD, VADResponse, VADEventType
from .eou import EOU
from .metrics import cascading_metrics_collector
from .denoise import Denoise
import logging

logger = logging.getLogger(__name__)

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
        self._stt_started = False
        
        self.stt_lock = asyncio.Lock()
        self.llm_lock = asyncio.Lock()
        self.tts_lock = asyncio.Lock()
        
        self.user_speech_callback: Callable[[], None] | None = None
        if self.stt:
            self.stt.on_stt_transcript(self.on_stt_transcript)
        if self.vad:
            self.vad.on_vad_event(self.on_vad_event)
        
    async def start(self) -> None:
        global_event_emitter.on("speech_started", self.on_speech_started_stt)
        global_event_emitter.on("speech_stopped", self.on_speech_stopped_stt)
        
        if self.agent and self.agent.instructions:
            cascading_metrics_collector.set_system_instructions(self.agent.instructions)

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
        """Handle STT transcript events"""
        if stt_response.event_type == SpeechEventType.FINAL:
            user_text = stt_response.data.text
            await self._process_final_transcript(user_text)
    
    async def _process_final_transcript(self, user_text: str) -> None:
        """Process final transcript with EOU detection and response generation"""
        
        # Fallback: If VAD is missing, this can start the turn. Otherwise, the collector handles it.
        if not cascading_metrics_collector.data.current_turn:
            cascading_metrics_collector.on_user_speech_start()
        
        cascading_metrics_collector.set_user_transcript(user_text)
        cascading_metrics_collector.on_stt_complete()
        
        # Fallback: If VAD is present but hasn't called on_user_speech_end yet,
        if self.vad and cascading_metrics_collector.data.is_user_speaking:
            cascading_metrics_collector.on_user_speech_end()
        elif not self.vad:
            cascading_metrics_collector.on_user_speech_end()
        
        self.agent.chat_context.add_message(
            role=ChatRole.USER,
            content=user_text
        )
        
        await self.on_turn_start(user_text)
        
        if self.turn_detector:
            cascading_metrics_collector.on_eou_start()
            eou_detected = self.turn_detector.detect_end_of_utterance(self.agent.chat_context)
            cascading_metrics_collector.on_eou_complete()
            
            if eou_detected:
                await self._generate_and_synthesize_response(user_text)
            else:
                cascading_metrics_collector.complete_current_turn()
        else:
            await self._generate_and_synthesize_response(user_text)
        
        await self.on_turn_end()
    
    async def _generate_and_synthesize_response(self, user_text: str) -> None:
        """Generate agent response and synthesize with TTS if available"""
        try:
            if self.tts:
                await self._synthesize_with_tts(self.run(user_text))
            else:
                agent_response = ""
                async for response_chunk in self.run(user_text):
                    agent_response += response_chunk
                cascading_metrics_collector.set_agent_response(agent_response)
        finally:
            cascading_metrics_collector.complete_current_turn()
            
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
            
            cascading_metrics_collector.on_llm_start()
            first_chunk_received = False
            
            async for llm_chunk_resp in self.llm.chat(
                self.agent.chat_context,
                tools=self.agent._tools
            ):
                if not first_chunk_received:
                    first_chunk_received = True
                    cascading_metrics_collector.on_llm_complete()
                    
                if llm_chunk_resp.metadata and "function_call" in llm_chunk_resp.metadata:
                    func_call = llm_chunk_resp.metadata["function_call"]
                    
                    cascading_metrics_collector.add_function_tool_call(func_call["name"])
                    
                    self.agent.chat_context.add_function_call(
                        name=func_call["name"],
                        arguments=json.dumps(func_call["arguments"]),
                        call_id=func_call.get("call_id", f"call_{int(time.time())}")
                    )
                    
                    try:
                        tool = next(
                            (t for t in self.agent.tools if is_function_tool(t) and get_tool_info(t).name == func_call["name"]),
                            None
                        )
                    except Exception as e:
                        logger.error(f"Error while selecting tool: {e}")
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
                            logger.error(f"Error executing function {func_call['name']}: {e}")
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
        """
        Direct TTS synthesis (used for initial messages)
        """
        if self.tts:
            cascading_metrics_collector.start_new_interaction("")
            cascading_metrics_collector.set_agent_response(message)
            
            try:
                await self._synthesize_with_tts(message)
            finally:
                cascading_metrics_collector.complete_current_turn()

    async def process_text_input(self, text: str) -> None:
        """
        Process text input directly (for A2A communication).
        This bypasses STT and directly processes the text through the LLM.
        """
        cascading_metrics_collector.start_new_interaction(text)
        
        self.agent.chat_context.add_message(
            role=ChatRole.USER,
            content=text
        )
        
        full_response = ""
        async for response_chunk in self.process_with_llm():
            full_response += response_chunk
        
        if full_response:
            cascading_metrics_collector.set_agent_response(full_response)
            cascading_metrics_collector.complete_current_turn()
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

    def on_speech_started_stt(self) -> None:
        if self.user_speech_callback:
            self.user_speech_callback()
    
    def on_speech_stopped_stt(self) -> None:
        pass

    def on_speech_started(self) -> None:
        cascading_metrics_collector.on_user_speech_start()
        
        if self.user_speech_callback:
            self.user_speech_callback()
            
        if self._stt_started:
            self._stt_started = False
            
        if self.tts:
            asyncio.create_task(self._interrupt_tts())

    async def _interrupt_tts(self) -> None:
        async with self.tts_lock:
            await self.tts.interrupt()
    
    def on_speech_stopped(self) -> None:
        
        if not self._stt_started:
            cascading_metrics_collector.on_stt_start()
            self._stt_started = True
        
        cascading_metrics_collector.on_user_speech_end()

    async def _synthesize_with_tts(self, response_gen: AsyncIterator[str] | str) -> None:
        """
        Stream LLM response chunks to TTS with buffering to sentences or pauses.
        Synthesizes at natural breaks (., !, ?, ;) with TTFB monitoring.
        """
        if not self.tts:
            return
        
        cascading_metrics_collector.on_tts_start()
        
        async def on_first_audio_byte():
            cascading_metrics_collector.on_tts_first_byte()
            cascading_metrics_collector.on_agent_speech_start()
        
        self.tts.on_first_audio_byte(on_first_audio_byte)
        
        self.tts.reset_first_audio_tracking()
        
        try:
            buffer = ""
            full_response = ""
            delimiters = {'.', '!', '?', ';', ',', '\n'}
            
            if isinstance(response_gen, str):
                async def string_to_iterator(text: str):
                    yield text
                response_gen = string_to_iterator(response_gen)
            
            async for chunk in response_gen:
                buffer += chunk
                full_response += chunk
                while buffer:
                    delimiter_pos = min((buffer.find(d) for d in delimiters if buffer.find(d) != -1), default=-1)
                    if delimiter_pos != -1:
                        to_speak = buffer[:delimiter_pos + 1]
                        buffer = buffer[delimiter_pos + 1:].lstrip() 
                        async with self.tts_lock:
                            await self.tts.synthesize(to_speak)
                    else:
                        break
            
            if buffer:
                async with self.tts_lock:
                    await self.tts.synthesize(buffer)
            
            if full_response:
                cascading_metrics_collector.set_agent_response(full_response)
                
        finally:
            cascading_metrics_collector.on_agent_speech_end()
