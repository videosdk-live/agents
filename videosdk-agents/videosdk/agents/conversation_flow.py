from __future__ import annotations

from abc import ABC
from typing import Awaitable, Callable, Literal, AsyncIterator, Any
import time
import json
import asyncio
from .event_emitter import EventEmitter
from .stt.stt import STT, STTResponse
from .llm.llm import LLM
from .llm.chat_context import ChatRole
from .utils import is_function_tool, get_tool_info, graceful_cancel
from .tts.tts import TTS
from .stt.stt import SpeechEventType
from .agent import Agent
from .event_bus import global_event_emitter
from .vad import VAD, VADResponse, VADEventType
from .eou import EOU
from .metrics import cascading_metrics_collector
from .denoise import Denoise
from .utils import UserState, AgentState
import logging
import wave
from .background_audio import BackgroundAudio, BackgroundAudioConfig

logger = logging.getLogger(__name__)


class ConversationFlow(EventEmitter[Literal["transcription"]], ABC):
    """
    Manages the conversation flow by listening to transcription events.
    """

    def __init__(self, agent: Agent, stt: STT | None = None, llm: LLM | None = None, tts: TTS | None = None, vad: VAD | None = None, turn_detector: EOU | None = None, denoise: Denoise | None = None) -> None:
        """Initialize conversation flow with event emitter capabilities"""
        super().__init__()
        self.transcription_callback: Callable[[
            STTResponse], Awaitable[None]] | None = None
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.vad = vad
        self.turn_detector = turn_detector
        self.agent = agent
        self.denoise = denoise
        self._stt_started = False
        self.background_audio: BackgroundAudioConfig | None = None
        self._background_audio_player: BackgroundAudio | None = None
        self.stt_lock = asyncio.Lock()
        self.llm_lock = asyncio.Lock()
        self.tts_lock = asyncio.Lock()

        self.user_speech_callback: Callable[[], None] | None = None
        if self.stt:
            self.stt.on_stt_transcript(self.on_stt_transcript)
        if self.vad:
            self.vad.on_vad_event(self.on_vad_event)

        self._current_tts_task: asyncio.Task | None = None
        self._current_llm_task: asyncio.Task | None = None
        self._partial_response = ""
        self._is_interrupted = False

        # Enhanced transcript accumulation system
        self._accumulated_transcript = ""
        self._waiting_for_more_speech = False
        self._speech_wait_timeout = 0.8  # 800ms timeout
        self._wait_timer: asyncio.TimerHandle | None = None
        self._transcript_processing_lock = asyncio.Lock()

        self.min_interruption_words = 1 # 2 

    async def start(self) -> None:
        global_event_emitter.on("speech_started", self.on_speech_started_stt)
        global_event_emitter.on("speech_stopped", self.on_speech_stopped_stt)

        if self.agent and self.agent.instructions:
            cascading_metrics_collector.set_system_instructions(
                self.agent.instructions)

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
        asyncio.create_task(self._process_audio_delta(audio_data))

    async def _process_audio_delta(self, audio_data: bytes) -> None:
        """Background processing of audio delta"""
        try:
            if self.denoise:
                audio_data = await self.denoise.denoise(audio_data)
            if self.stt:
                async with self.stt_lock:
                    await self.stt.process_audio(audio_data)
            if self.vad:
                await self.vad.process_audio(audio_data)
        except Exception as e:
            self.emit("error", f"Audio processing failed: {str(e)}")

    async def on_vad_event(self, vad_response: VADResponse) -> None:
        """Handle VAD events"""
        if vad_response.event_type == VADEventType.START_OF_SPEECH:
            # If we're waiting for more speech and user starts speaking again
            if self._waiting_for_more_speech:
                await self._handle_continued_speech()
            await self.on_speech_started()
        elif vad_response.event_type == VADEventType.END_OF_SPEECH:
            self.on_speech_stopped()

    async def _handle_continued_speech(self) -> None:
        """Handle when user continues speaking while we're waiting"""
        # Cancel the wait timer
        if self._wait_timer:
            self._wait_timer.cancel()
            self._wait_timer = None
        
        self._waiting_for_more_speech = False

    async def on_stt_transcript(self, stt_response: STTResponse) -> None:
        """Handle STT transcript events with enhanced EOU logic"""
            
        if self._waiting_for_more_speech:
            await self._handle_continued_speech()
            
        text = stt_response.data.text if stt_response.data else ""
        
        if self.agent.session:
            if self.agent.session.agent_state in (AgentState.SPEAKING, AgentState.THINKING): 
                await self.handle_stt_event(text)
            else:
                logger.info(f"STT TRANSCRIPT EVENT: Agent is not SPEAKING, skipping interruption, Agent state: {self.agent.session.agent_state}")
                
        if self.agent.session:
            self.agent.session._emit_user_state(UserState.SPEAKING)
            
        if stt_response.event_type == SpeechEventType.FINAL:
            user_text = stt_response.data.text
            await self._process_transcript_with_eou(user_text)

    async def _process_transcript_with_eou(self, new_transcript: str) -> None:
        """Enhanced transcript processing with EOU-based decision making"""
        async with self._transcript_processing_lock:
            if self.agent.session:
                self.agent.session._emit_agent_state(AgentState.LISTENING) 
            # Append new transcript to accumulated transcript
            if self._accumulated_transcript:
                self._accumulated_transcript += " " + new_transcript
            else:
                self._accumulated_transcript = new_transcript
            
            # Check EOU with accumulated transcript
            is_eou = await self._check_end_of_utterance(self._accumulated_transcript)
            
            if is_eou:
                await self._finalize_transcript_and_respond()
            else:
                await self._wait_for_additional_speech()

    async def _check_end_of_utterance(self, transcript: str) -> bool:
        """Check if the current transcript represents end of utterance"""
        if not self.turn_detector:
            # If no EOU detector, assume it's always end of utterance
            return True
        
        # Create temporary chat context for EOU detection
        temp_context = self.agent.chat_context.copy()
        temp_context.add_message(role=ChatRole.USER, content=transcript)
        
        cascading_metrics_collector.on_eou_start()
        is_eou = self.turn_detector.detect_end_of_utterance(temp_context)
        cascading_metrics_collector.on_eou_complete()
        
        return is_eou

    async def _wait_for_additional_speech(self) -> None:
        """Wait for additional speech within the timeout period"""

        if self._waiting_for_more_speech:
            # Already waiting, extend the timer
            if self._wait_timer:
                self._wait_timer.cancel()
        
        self._waiting_for_more_speech = True
        
        # Set timer for speech timeout
        loop = asyncio.get_event_loop()
        self._wait_timer = loop.call_later(
            self._speech_wait_timeout,
            lambda: asyncio.create_task(self._on_speech_timeout())
        )
        

    async def _on_speech_timeout(self) -> None:
        """Handle timeout when no additional speech is detected"""
        async with self._transcript_processing_lock:
            if not self._waiting_for_more_speech:
                return  # Already processed or cancelled
            
            self._waiting_for_more_speech = False
            self._wait_timer = None
            
            await self._finalize_transcript_and_respond()

    async def _finalize_transcript_and_respond(self) -> None:
        """Finalize the accumulated transcript and generate response"""
        if not self._accumulated_transcript.strip():
            return
        
        final_transcript = self._accumulated_transcript.strip()
        logger.info(f"Finalizing transcript: '{final_transcript}'")
        
        # Reset accumulated transcript
        self._accumulated_transcript = ""
        
        # Process the final transcript
        await self._process_final_transcript(final_transcript)

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

        # Generate response
        asyncio.create_task(self._generate_and_synthesize_response(user_text))

        # Async helper: waits before generating a response (used if utterance isn't clearly ended)
        # async def generate_response_after_delay(delay: float):
        #     await asyncio.sleep(delay)
        #     if not asyncio.current_task().done():
        #         await self._generate_and_synthesize_response(user_text)

        # If turn detection is enabled
        # if self.turn_detector:
        #     cascading_metrics_collector.on_eou_start()
        #     eou_detected = self.turn_detector.detect_end_of_utterance(
        #         self.agent.chat_context)
        #     cascading_metrics_collector.on_eou_complete()

        #     If user finished speaking → respond immediately
        #     if eou_detected:
        #         asyncio.create_task(
        #             self._generate_and_synthesize_response(user_text))
        #     Else → start a 2s timer, then respond if no speech continues
        #     else:
        #         self._eou_timer_task = asyncio.create_task(generate_response_after_delay(2.0))
        #         # cascading_metrics_collector.complete_current_turn()
        # else:
        #     # If no turn detection, always respond immediately
        #     asyncio.create_task(
        #         self._generate_and_synthesize_response(user_text))

        await self.on_turn_end()

    async def _process_reply_instructions(self, instructions: str, wait_for_playback: bool = True) -> None:
        """Process reply instructions and generate response using existing flow"""
        
        original_vad_handler = None
        original_stt_handler = None
        
        if wait_for_playback:
            # Temporarily disable VAD events
            if self.vad:
                original_vad_handler = self.on_vad_event
                self.on_vad_event = lambda x: None
            
            # Temporarily disable STT transcript processing
            if self.stt:
                original_stt_handler = self.on_stt_transcript
                self.on_stt_transcript = lambda x: None
        
        try:
            self.agent.chat_context.add_message(
                role=ChatRole.USER,
                content=instructions
            )

            await self.on_turn_start(instructions)
            await self._generate_and_synthesize_response(instructions)

            await self.on_turn_end()
            
            if wait_for_playback:
                while (hasattr(cascading_metrics_collector.data, 'is_agent_speaking') and 
                    cascading_metrics_collector.data.is_agent_speaking):
                    await asyncio.sleep(0.1)
                    
        finally:
            if wait_for_playback:
                if original_vad_handler is not None:
                    self.on_vad_event = original_vad_handler
                
                if original_stt_handler is not None:
                    self.on_stt_transcript = original_stt_handler

    async def _generate_and_synthesize_response(self, user_text: str) -> None:
        """Generate agent response"""
        self._is_interrupted = False

        full_response = ""
        self._partial_response = ""

        try:
            if self.background_audio and self.tts and self.tts.audio_track:
                self._background_audio_player = BackgroundAudio(self.background_audio, self.tts.audio_track)
                await self._background_audio_player.start()

            llm_stream = self.run(user_text)

            q = asyncio.Queue(maxsize=50)

            async def collector():
                response_parts = []
                try:
                    async for chunk in llm_stream:
                        if self._is_interrupted:
                            logger.info("LLM collection interrupted")
                            await q.put(None)
                            return "".join(response_parts)

                        self._partial_response = "".join(response_parts)
                        
                        if not self._is_interrupted:  # Only queue if not interrupted
                            await q.put(chunk)
                            response_parts.append(chunk)

                    if not self._is_interrupted:
                        await q.put(None)
                    full_response = "".join(response_parts)
                    return full_response
                except asyncio.CancelledError:
                    logger.info("LLM collection cancelled")
                    await q.put(None)
                    return "".join(response_parts)

            async def tts_consumer():
                async def tts_stream_gen():
                    while True:
                        if self._is_interrupted:
                            break

                        try:
                            chunk = await asyncio.wait_for(q.get(), timeout=0.1)
                            if chunk is None:
                                break
                            yield chunk
                        except asyncio.TimeoutError:
                            if self._is_interrupted:
                                break
                            continue

                if self.tts:
                    try:
                        await self._synthesize_with_tts(tts_stream_gen())
                    except asyncio.CancelledError:
                        await self.tts.interrupt()


            collector_task = asyncio.create_task(collector())
            tts_task = asyncio.create_task(tts_consumer())

            self._current_llm_task = collector_task
            self._current_tts_task = tts_task

            try:
                await asyncio.gather(collector_task, tts_task, return_exceptions=True)
            except asyncio.CancelledError:
                # Ensure proper cleanup on cancellation
                if not collector_task.done():
                    collector_task.cancel()
                if not tts_task.done():
                    tts_task.cancel()

            if not collector_task.cancelled() and not self._is_interrupted:
                full_response = collector_task.result()
            else:
                full_response = self._partial_response

            if full_response and not self._is_interrupted:
                cascading_metrics_collector.set_agent_response(full_response)
                self.agent.chat_context.add_message(
                    role=ChatRole.ASSISTANT,
                    content=full_response
                )

        finally:
            self._current_tts_task = None
            self._current_llm_task = None
            cascading_metrics_collector.complete_current_turn()

    async def process_with_llm(self) -> AsyncIterator[str]:
        """
        Process the current chat context with LLM and yield response chunks.
        This method can be called by user implementations to get LLM responses.
        """
        async with self.llm_lock:
            if not self.llm:
                return

            cascading_metrics_collector.on_llm_start()
            first_chunk_received = False
            
            if self.agent.session:
                self.agent.session._emit_user_state(UserState.IDLE)
                self.agent.session._emit_agent_state(AgentState.THINKING)

            async for llm_chunk_resp in self.llm.chat(
                self.agent.chat_context,
                tools=self.agent._tools
            ):
                if self._is_interrupted:
                    logger.info("LLM processing interrupted")
                    break

                if not first_chunk_received:
                    first_chunk_received = True
                    cascading_metrics_collector.on_llm_complete()

                if llm_chunk_resp.metadata and "function_call" in llm_chunk_resp.metadata:
                    func_call = llm_chunk_resp.metadata["function_call"]

                    cascading_metrics_collector.add_function_tool_call(
                        func_call["name"])

                    self.agent.chat_context.add_function_call(
                        name=func_call["name"],
                        arguments=json.dumps(func_call["arguments"]),
                        call_id=func_call.get(
                            "call_id", f"call_{int(time.time())}")
                    )

                    try:
                        tool = next(
                            (t for t in self.agent.tools if is_function_tool(
                                t) and get_tool_info(t).name == func_call["name"]),
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
                                call_id=func_call.get(
                                    "call_id", f"call_{int(time.time())}")
                            )

                            async for new_resp in self.llm.chat(self.agent.chat_context):
                                if self._is_interrupted:
                                    break
                                if new_resp.content:
                                    yield new_resp.content
                        except Exception as e:
                            logger.error(
                                f"Error executing function {func_call['name']}: {e}")
                            continue
                else:
                    if llm_chunk_resp.content:
                        yield llm_chunk_resp.content

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

    def on_speech_started_stt(self, event_data: Any) -> None:
        if self.user_speech_callback:
            self.user_speech_callback()
        
        if self.agent.session:
            self.agent.session._emit_user_state(UserState.SPEAKING)

    def on_speech_stopped_stt(self, event_data: Any) -> None:
        pass
    
    async def handle_stt_event(self, text: str) -> None:
        """Handle STT event"""
        if not text or not text.strip():
            return
        
        word_count = len(text.strip().split())
        logger.info(f"handle_stt_event: Word count: {word_count}")
        if word_count >= self.min_interruption_words:
            await self._trigger_interruption()
            
    async def _trigger_interruption(self) -> None:
        """Trigger interruption once"""
        logger.info(f"Triggering interruption From STT")
        if self._is_interrupted:
            logger.info(f"Already interrupted, skipping")
            return
            
        if self.tts:
            try:
                await self._interrupt_tts()
            except Exception as e:
                logger.error(f" _trigger_interruption Callback error: {e}")
    
    async def on_speech_started(self) -> None:       
        cascading_metrics_collector.on_user_speech_start()

        if self.user_speech_callback:
            self.user_speech_callback()

        if self._stt_started:
            self._stt_started = False

        if self.tts:
            await self._interrupt_tts()
        
        if self.agent.session:
            self.agent.session._emit_user_state(UserState.SPEAKING)

    async def _interrupt_tts(self) -> None:
        self._is_interrupted = True

        if self._background_audio_player:
            await self._background_audio_player.stop()
            self._background_audio_player = None

        # Cancel any waiting timers
        if self._wait_timer:
            self._wait_timer.cancel()
            self._wait_timer = None
        self._waiting_for_more_speech = False

        if self.tts:
            await self.tts.interrupt()

        if self.llm:
            await self._cancel_llm()

        tasks_to_cancel = []
        if self._current_tts_task and not self._current_tts_task.done():
            tasks_to_cancel.append(self._current_tts_task)
        if self._current_llm_task and not self._current_llm_task.done():
            tasks_to_cancel.append(self._current_llm_task)

        if tasks_to_cancel:
            # Force cancel tasks to ensure immediate stop
            for task in tasks_to_cancel:
                task.cancel()
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            
        # Reset conversation state
        self._partial_response = ""
        self._is_interrupted = False

        cascading_metrics_collector.on_interrupted()

    async def _cancel_llm(self) -> None:
        """Cancel LLM generation"""
        try:
            await self.llm.cancel_current_generation()
        except Exception as e:
            logger.error(f"LLM cancellation failed: {e}")

    def on_speech_stopped(self) -> None:
        if not self._stt_started:
            cascading_metrics_collector.on_stt_start()
            self._stt_started = True

        cascading_metrics_collector.on_user_speech_end()
        
        if self.agent.session:
            self.agent.session._emit_user_state(UserState.IDLE)

    async def _synthesize_with_tts(self, response_gen: AsyncIterator[str] | str) -> None:
        """
        Stream LLM response directly to TTS.
        """
        if not self.tts:
            return
        
        self.agent.session._pause_wake_up_timer()

        async def on_first_audio_byte():
            if self._background_audio_player:
                await self._background_audio_player.stop()
                self._background_audio_player = None
            cascading_metrics_collector.on_tts_first_byte()
            cascading_metrics_collector.on_agent_speech_start()
            
            if self.agent.session:
                self.agent.session._emit_agent_state(AgentState.SPEAKING)
                self.agent.session._emit_user_state(UserState.LISTENING)

        self.tts.on_first_audio_byte(on_first_audio_byte)
        self.tts.reset_first_audio_tracking()

        cascading_metrics_collector.on_tts_start()
        try:
            response_iterator: AsyncIterator[str]
            if isinstance(response_gen, str):
                async def string_to_iterator(text: str):
                    yield text
                response_iterator = string_to_iterator(response_gen)
            else:
                response_iterator = response_gen

            await self.tts.synthesize(response_iterator)
            

        finally:
            if self._background_audio_player:
                await self._background_audio_player.stop()
                self._background_audio_player = None

            self.agent.session._reply_in_progress = False
            self.agent.session._reset_wake_up_timer()
            cascading_metrics_collector.on_agent_speech_end()
            
        # TODO: Need to work on IDLE state 
            # if self.agent.session:
            #     self.agent.session._emit_agent_state(AgentState.IDLE)
            #     self.agent.session._emit_user_state(UserState.IDLE)
    
    async def cleanup(self) -> None:
        """Cleanup conversation flow resources"""
        logger.info("Cleaning up conversation flow")
        if self._current_tts_task and not self._current_tts_task.done():
            self._current_tts_task.cancel()
            try:
                await self._current_tts_task
            except asyncio.CancelledError:
                pass
            self._current_tts_task = None
        
        if self._current_llm_task and not self._current_llm_task.done():
            self._current_llm_task.cancel()
            try:
                await self._current_llm_task
            except asyncio.CancelledError:
                pass
            self._current_llm_task = None
        
        if self._background_audio_player:
            await self._background_audio_player.stop()
            self._background_audio_player = None

        if self._eou_timer_task and not self._eou_timer_task.done():
            self._eou_timer_task.cancel()
            try:
                await self._eou_timer_task
            except asyncio.CancelledError:
                pass
            self._eou_timer_task = None
        
        if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'chat_context') and self.agent.chat_context:
            try:
                self.agent.chat_context.cleanup()
                logger.info("Agent chat context cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up agent chat context: {e}")
        
        self.transcription_callback = None
        self.user_speech_callback = None
        self.stt = None
        self.llm = None
        self.tts = None
        self.vad = None
        self.turn_detector = None
        self.agent = None
        self.denoise = None
        self._stt_started = False
        self._partial_response = ""
        self._is_interrupted = False
        self.background_audio = None
        self._background_audio_player = None
        logger.info("Conversation flow cleaned up")