from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Literal, AsyncIterator
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
from .vad import VAD, VADResponse, VADEventType
from .eou import EOU
from .metrics import metrics_collector

class ConversationFlow(EventEmitter[Literal["transcription"]], ABC):
    """
    Manages the conversation flow by listening to transcription events.
    """

    def __init__(self, agent: Agent, stt: STT | None = None, llm: LLM | None = None, tts: TTS | None = None, vad: VAD | None = None, turn_detector: EOU | None = None) -> None:
        """Initialize conversation flow with event emitter capabilities"""
        super().__init__() 
        self.transcription_callback: Callable[[STTResponse], Awaitable[None]] | None = None
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.vad = vad
        self.turn_detector = turn_detector
        self.agent = agent
        self.is_turn_active = False
        
        if self.stt:
            self.stt.on_stt_transcript(self.on_stt_transcript)
        if self.vad:
            self.vad.on_vad_event(self.on_vad_event)
        
    async def start(self) -> None:
        global_event_emitter.on("speech_started", self.on_speech_started)
        global_event_emitter.on("speech_stopped", self.on_speech_stopped)
        
        if self.agent and self.agent.instructions:
            metrics_collector.set_system_instructions(self.agent.instructions)

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
        if self.vad:
            await self.vad.process_audio(audio_data)
                        
    async def on_vad_event(self, vad_response: VADResponse) -> None:
        if vad_response.event_type == VADEventType.START_OF_SPEECH:
            metrics_collector.on_user_speech_start()
            self.on_speech_started()
        elif vad_response.event_type == VADEventType.END_OF_SPEECH:
            metrics_collector.on_user_speech_end()
            self.on_speech_stopped()
            
    async def _run_and_collect_response(self, transcript: str) -> AsyncIterator[str]:
        """
        Helper method to run LLM and collect response for metrics while yielding chunks
        """
        agent_response = ""
        async for response_chunk in self.process_with_llm():
            agent_response += response_chunk
            yield response_chunk
        
        metrics_collector.set_agent_response(agent_response)

    async def _synthesize_with_ttfb_monitoring(self, text_generator: AsyncIterator[str]) -> None:
        """
        🔧 Enhanced TTS synthesis with TTFB monitoring
        """
        if not self.tts:
            return
            
        metrics_collector.on_tts_start()
        
        try:
            first_audio_detected = False
            
            async def ttfb_monitoring_wrapper():
                nonlocal first_audio_detected
                async for chunk in text_generator:
                    yield chunk
                    if not first_audio_detected:
                        first_audio_detected = True
                        metrics_collector.on_tts_first_byte()  
                        metrics_collector.on_agent_speech_start()
            
            await self.tts.synthesize(ttfb_monitoring_wrapper())
            
        finally:
            metrics_collector.on_agent_speech_end()

    async def on_stt_transcript(self, stt_response: STTResponse) -> None:
        if stt_response.event_type == SpeechEventType.FINAL:
            user_text = stt_response.data.text
            
            metrics_collector.set_user_transcript(user_text)
            metrics_collector.on_stt_complete()
            
            if not self.vad:
                metrics_collector.on_user_speech_end()
            
            self.agent.chat_context.add_message(
                role=ChatRole.USER,
                content=user_text
            )
            
            if self.turn_detector and self.turn_detector.detect_end_of_utterance(self.agent.chat_context):
                if self.tts:
                    try:
                        await self._synthesize_with_ttfb_monitoring(self._run_and_collect_response(user_text))
                    finally:
                        metrics_collector.complete_current_interaction()
                else:
                    agent_response = ""
                    async for response_chunk in self.run(user_text):
                        agent_response += response_chunk
                    metrics_collector.set_agent_response(agent_response)
                    metrics_collector.complete_current_interaction()
            
            if not self.turn_detector:
                if self.tts:
                    try:
                        await self._synthesize_with_ttfb_monitoring(self._run_and_collect_response(user_text))
                    finally:
                        metrics_collector.complete_current_interaction()
                else:
                    agent_response = ""
                    async for response_chunk in self.run(user_text):
                        agent_response += response_chunk
                    metrics_collector.set_agent_response(agent_response)
                    metrics_collector.complete_current_interaction()

    async def process_with_llm(self) -> AsyncIterator[str]:
        """
        Process the current chat context with LLM and yield response chunks.
        This method can be called by user implementations to get LLM responses.
        """
        if not self.llm:
            return
            
        metrics_collector.on_llm_start()
        
        full_response = ""
        prev_content_length = 0
        
        async for llm_chunk_resp in self.llm.chat(
            self.agent.chat_context,
            tools=self.agent._tools
        ):
            if llm_chunk_resp.metadata and "function_call" in llm_chunk_resp.metadata:
                func_call = llm_chunk_resp.metadata["function_call"]
                
                metrics_collector.add_function_tool_call(func_call["name"])
                
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
                    print(f"  Error while selecting tool: {e}")
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
                        print(f"  Error executing function {func_call['name']}: {e}")
                        continue
            else:
                new_content = llm_chunk_resp.content[prev_content_length:]
                if new_content: 
                    yield new_content
                full_response = llm_chunk_resp.content
                prev_content_length = len(llm_chunk_resp.content)
        
        metrics_collector.on_llm_complete()
        
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
            metrics_collector.start_new_interaction("")
            metrics_collector.set_agent_response(message)
            
            async def single_message_generator():
                yield message
            
            try:
                await self._synthesize_with_ttfb_monitoring(single_message_generator())
            finally:
                metrics_collector.complete_current_interaction()

    async def process_text_input(self, text: str) -> None:
        """
        Process text input directly (for A2A communication).
        This bypasses STT and directly processes the text through the LLM.
        """
        metrics_collector.start_new_interaction(text)
        
        self.agent.chat_context.add_message(
            role=ChatRole.USER,
            content=text
        )
        
        full_response = ""
        async for response_chunk in self.process_with_llm():
            full_response += response_chunk
        
        if full_response:
            metrics_collector.set_agent_response(full_response)
            metrics_collector.complete_current_interaction()
            global_event_emitter.emit("text_response", {"text": full_response})
    
    
    async def run(self, transcript: str) -> AsyncIterator[str]:
        """
        Main conversation loop: handle a user turn.
        Users should implement this method to preprocess transcripts and yield response chunks.
        """
        async for response in self.process_with_llm():
            yield response
    
    @abstractmethod
    async def on_turn_start(self, transcript: str) -> None:
        """Called at the start of a user turn."""
        pass
    
    @abstractmethod
    async def on_turn_end(self) -> None:
        """Called at the end of a user turn."""
        pass

    def on_speech_started(self) -> None:
        if not self.vad:
            metrics_collector.on_user_speech_start()
        
        if self.tts:
            asyncio.create_task(self.tts.interrupt())

    def on_speech_stopped(self) -> None:
        metrics_collector.start_new_interaction()
        metrics_collector.on_stt_start()
        
        if not self.vad:
            metrics_collector.on_user_speech_end()
        
        if not self.vad:
            current_time = time.perf_counter()
            
            if (metrics_collector.data.current_interaction and 
                metrics_collector.data.current_interaction.user_speech_start_time):
                pass
            else:
                estimated_speech_duration = 2.0
                estimated_start_time = current_time - estimated_speech_duration
                
                if not metrics_collector.data.is_user_speaking:
                    metrics_collector.data.user_input_start_time = estimated_start_time
                    metrics_collector.data.is_user_speaking = True
                    
                    if metrics_collector.data.current_interaction:
                        metrics_collector.data.current_interaction.user_speech_start_time = estimated_start_time
            
            if metrics_collector.data.current_interaction:
                metrics_collector.data.current_interaction.user_speech_end_time = current_time