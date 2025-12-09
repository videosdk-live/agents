from __future__ import annotations

from abc import ABC
from typing import Awaitable, Callable, Literal, AsyncIterator, Any
import time
import json
import asyncio
from .event_emitter import EventEmitter
from .stt.stt import STT, STTResponse
from .llm.llm import LLM
from .llm.chat_context import ChatRole, ImageContent
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
import uuid
from .utterance_handle import UtteranceHandle
import logging
import av
from typing import TYPE_CHECKING
from .voice_mail_detector import VoiceMailDetector
if TYPE_CHECKING:
    from .knowledge_base.base import KnowledgeBase
    
logger = logging.getLogger(__name__)


class ConversationFlow(EventEmitter[Literal["transcription"]], ABC):
    """
    Manages the conversation flow by listening to transcription events.
    """

    def __init__(self, agent: Agent, stt: STT | None = None, llm: LLM | None = None, tts: TTS | None = None, vad: VAD | None = None, turn_detector: EOU | None = None, denoise: Denoise | None = None, avatar: Any | None = None) -> None:
        """Initialize conversation flow with event emitter capabilities"""
        super().__init__()
        self.transcription_callback: Callable[[
            STTResponse], Awaitable[None]] | None = None
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.audio_track = None
        self.vad = vad
        self.turn_detector = turn_detector
        self.agent = agent
        self.denoise = denoise
        self.avatar = avatar
        self._stt_started = False
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

        self.min_interruption_words = 2 # 2 
        
        # Preemptive generation state
        self._preemptive_transcript: str | None = None
        self._preemptive_lock = asyncio.Lock()
        
        self._preemptive_generation_task: asyncio.Task | None = None
        self._preemptive_authorized = asyncio.Event()  # Authorization gate
        self._preemptive_cancelled = False
        
        # Voice Mail detection state
        self.voice_mail_detector: VoiceMailDetector | None = None
        self.voice_mail_detection_done = False
        self._vmd_buffer = ""
        self._vmd_check_task: asyncio.Task | None = None


    def _update_preemptive_generation_flag(self) -> None:
        """Update the preemptive generation flag based on current STT instance"""
        self._enable_preemptive_generation = getattr(self.stt, 'enable_preemptive_generation', False) if self.stt else False
        
    async def start(self) -> None:
        global_event_emitter.on("speech_started", self.on_speech_started_stt)
        global_event_emitter.on("speech_stopped", self.on_speech_stopped_stt)

        if self.agent and self.agent.instructions:
            cascading_metrics_collector.set_system_instructions(
                self.agent.instructions)

    def set_voice_mail_detector(self, detector: VoiceMailDetector | None) -> None:
        """Configures voicemail detection. Called by AgentSession."""
        self.voice_mail_detector = detector
        self.voice_mail_detection_done = False
        self._vmd_buffer = ""

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

        if self.voice_mail_detector and not self.voice_mail_detection_done and text.strip():
            self._vmd_buffer += f" {text}"
            if not self._vmd_check_task:
                logger.info("Starting Voice Mail Detection Timer")
                self._vmd_check_task = asyncio.create_task(self._run_vmd_check())

        if self.agent.session:
            state = self.agent.session.agent_state
            if state == AgentState.SPEAKING:
                logger.info(f"Agent is speaking, handling STT event")          
                await self.handle_stt_event(text)
                
            elif state == AgentState.THINKING:
               if not self._enable_preemptive_generation:
                 await self.handle_stt_event(text)

        if self.agent.session:
            self.agent.session._emit_user_state(UserState.SPEAKING)

        # Handle different event types
        if stt_response.event_type == SpeechEventType.PREFLIGHT:
            await self._handle_preflight_transcript(text)
            
        elif stt_response.event_type == SpeechEventType.FINAL:
            user_text = stt_response.data.text
            if self._enable_preemptive_generation:
                await self._authorize_or_process_final_transcript(user_text)
            else:
                await self._process_transcript_with_eou(user_text)
            
        elif stt_response.event_type == SpeechEventType.INTERIM:
            if stt_response.metadata and stt_response.metadata.get("turn_resumed"):
                await self._handle_turn_resumed(text)
    
    async def _run_vmd_check(self) -> None:
        """Internal task to wait and check LLM, then emit result."""
        try:
            if not self.voice_mail_detector:
                return
            await asyncio.sleep(self.voice_mail_detector.duration)
            
            is_voicemail = await self.voice_mail_detector.detect(self._vmd_buffer.strip())
            
            self.voice_mail_detection_done = True
            
            if is_voicemail:
                await self._interrupt_tts()
                await self._cancel_llm()

            self.emit("voicemail_result", {"is_voicemail": is_voicemail})
                
        except Exception as e:
            logger.error(f"Error in VMD check: {e}")
            self.voice_mail_detection_done = True
            self.emit("voicemail_result", {"is_voicemail": False})
        finally:
            self._vmd_check_task = None
            self._vmd_buffer = ""

    async def _handle_preflight_transcript(self, preflight_text: str) -> None:
        """
        Handle preflight transcript - start generation but wait for authorization.
        """
        async with self._preemptive_lock:
            self._preemptive_transcript = preflight_text.strip()
            self._preemptive_authorized.clear()  # Not authorized yet
            self._preemptive_cancelled = False
            
            user_text = preflight_text.strip()
            if self.agent.knowledge_base:
                kb_context = await self.agent.knowledge_base.process_query(user_text)
                if kb_context:
                    user_text = f"{kb_context}\n\nUser: {user_text}"
            
            # Add preflight transcript to temporary context
            self.agent.chat_context.add_message(
                role=ChatRole.USER,
                content=user_text
            )
            
            if self.agent and self.agent.session:
                if self.agent.session.current_utterance and not self.agent.session.current_utterance.done():
                    self.agent.session.current_utterance.interrupt()
                
                handle = UtteranceHandle(utterance_id=f"utt_{uuid.uuid4().hex[:8]}")
                self.agent.session.current_utterance = handle
            else:
                handle = UtteranceHandle(utterance_id="utt_fallback")
                handle._mark_done()
            
            self._preemptive_generation_task = asyncio.create_task(
                self._generate_and_synthesize_response(
                    user_text,
                    handle,
                    wait_for_authorization=True
                )
            )
            
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
        
        if not cascading_metrics_collector.data.current_turn: 
            cascading_metrics_collector.start_new_interaction()

        cascading_metrics_collector.set_user_transcript(user_text)
        cascading_metrics_collector.on_stt_complete()

        if self.vad and cascading_metrics_collector.data.is_user_speaking: 
            cascading_metrics_collector.on_user_speech_end()
        elif not self.vad:
            cascading_metrics_collector.on_user_speech_end()

        final_user_text = user_text
        if self.agent.knowledge_base:
            kb_context = await self.agent.knowledge_base.process_query(user_text)
            if kb_context:
                final_user_text = f"{kb_context}\n\nUser: {user_text}"

        self.agent.chat_context.add_message(
            role=ChatRole.USER,
            content=final_user_text
        )

        if self.agent and self.agent.session:
            if self.agent.session.current_utterance and not self.agent.session.current_utterance.done():
                self.agent.session.current_utterance.interrupt()
            
            handle = UtteranceHandle(utterance_id=f"utt_{uuid.uuid4().hex[:8]}")
            self.agent.session.current_utterance = handle
        else:
            handle = UtteranceHandle(utterance_id="utt_fallback")
            handle._mark_done()

        await self.on_turn_start(final_user_text)

        # Generate response
        asyncio.create_task(self._generate_and_synthesize_response(final_user_text, handle))


        await self.on_turn_end()

    async def _process_reply_instructions(self, instructions: str, wait_for_playback: bool, handle: UtteranceHandle, frames: list[av.VideoFrame] | None = None) -> None:
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
            final_instructions = instructions
            if self.agent.knowledge_base:
                kb_context = await self.agent.knowledge_base.process_query(instructions)
                if kb_context:
                    final_instructions = f"{kb_context}\n\nUser: {instructions}"
                
            content_parts = [final_instructions]
            if frames:
                for frame in frames:
                    image_part = ImageContent(image=frame, inference_detail="auto")
                    content_parts.append(image_part)
            
            self.agent.chat_context.add_message(
                role=ChatRole.USER,
                content=content_parts if len(content_parts) > 1 else final_instructions
            )

            await self.on_turn_start(final_instructions)
            await self._generate_and_synthesize_response(final_instructions, handle)
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
            
            if not handle.done():
                handle._mark_done()

    async def _authorize_or_process_final_transcript(self, final_text: str) -> None:
        """
        Handle final transcript - authorize preemptive generation or start new.
        """
        async with self._preemptive_lock:
            final_text_normalized = final_text.strip()
            
            if self._preemptive_transcript:
                preflight_normalized = self._preemptive_transcript.strip()
                
                # Compare transcripts
                if final_text_normalized == preflight_normalized:
                    logger.info(f"MATCH! Authorizing preemptive generation")
                    
                    # Authorize the waiting TTS to play audio
                    self._preemptive_authorized.set()
                    
                    # Wait for preemptive task to complete
                    if self._preemptive_generation_task:
                        try:
                            await asyncio.wait_for(
                                self._preemptive_generation_task, 
                                timeout=30.0  # Generous timeout for playback
                            )
                            logger.info("Preemptive generation completed successfully")
                        except asyncio.TimeoutError:
                            logger.error("Preemptive playback timeout")
                        except Exception as e:
                            logger.error(f"Error in preemptive playback: {e}")
                else:
                    logger.info(f"MISMATCH! Cancelling Preemptive Generation")
                    
                    # Cancel preemptive generation
                    await self._cancel_preemptive_generation()
                    
                    # Remove the wrong user message from context
                    if self.agent.chat_context.messages and \
                    self.agent.chat_context.messages[-1].role == ChatRole.USER:
                        self.agent.chat_context.messages.pop()
                    
                    # Follow normal flow with correct transcript
                    await self._process_transcript_with_eou(final_text_normalized)
            else:
                # No preflight, normal flow
                logger.info(f"No preflight, processing normally: '{final_text_normalized}'")
                await self._process_transcript_with_eou(final_text_normalized)
            
            # Cleanup
            self._preemptive_transcript = None
            self._preemptive_generation_task = None

    async def _cancel_preemptive_generation(self) -> None:
        """Cancel preemptive generation"""
        logger.info("Cancelling preemptive generation...")
        if  self._enable_preemptive_generation:
            self._preemptive_cancelled = True
            self._preemptive_authorized.set()  # Unblock to allow cancellation
            
            # Cancel the task
            if self._preemptive_generation_task and not self._preemptive_generation_task.done():
                self._preemptive_generation_task.cancel()
                try:
                    await self._preemptive_generation_task
                except asyncio.CancelledError:
                    logger.info("Preemptive task cancelled successfully")
                self._preemptive_generation_task = None
            
            # Cancel LLM/TTS
            if self.llm:
                try:
                    await self.llm.cancel_current_generation()
                except Exception as e:
                    logger.debug(f"LLM cancellation: {e}")
            
            if self.tts:
                await self.tts.interrupt()
            
            self._preemptive_transcript = None
            logger.info("Preemptive generation cancelled and cleaned up")
        
    async def _handle_turn_resumed(self, resumed_text: str) -> None:
        """
        Handle TurnResumed event (user continued speaking).
        Edge case: Cancel preemptive generation immediately.
        """
        await self._cancel_preemptive_generation()
        
        # Update accumulated transcript
        if self._accumulated_transcript:
            self._accumulated_transcript += " " + resumed_text
        else:
            self._accumulated_transcript = resumed_text

    async def _generate_and_synthesize_response(self, user_text: str, handle: UtteranceHandle, wait_for_authorization: bool = False) -> None:
        """Generate agent response and manage handle lifecycle"""
        self._is_interrupted = False

        full_response = ""
        self._partial_response = ""

        try:
            if self.agent.session and self.agent.session.is_background_audio_enabled:
                await self.agent.session.start_thinking_audio()

            llm_stream = self.run(user_text)

            q = asyncio.Queue(maxsize=50)

            async def collector():
                response_parts = []
                try:
                    async for chunk in llm_stream:
                        if handle.interrupted or (wait_for_authorization and self._preemptive_cancelled):
                            logger.info("LLM collection interrupted")
                            await q.put(None)
                            return "".join(response_parts)

                        self._partial_response = "".join(response_parts)
                        
                        if not handle.interrupted:
                            await q.put(chunk)
                            response_parts.append(chunk)

                    if not handle.interrupted:
                        await q.put(None)
                    full_response = "".join(response_parts)
                    return full_response
                except asyncio.CancelledError:
                    logger.info("LLM collection cancelled")
                    await q.put(None)
                    return "".join(response_parts)

            async def tts_consumer():
                """Consumes LLM chunks and sends to TTS with authorization gate"""
                
                # NEW: Wait for authorization if this is preemptive generation
                if wait_for_authorization:
                    try:
                        # Wait for authorization or cancellation
                        await asyncio.wait_for(
                            self._preemptive_authorized.wait(), 
                            timeout=10.0  # Safety timeout
                        )
                        
                        if self._preemptive_cancelled:
                            logger.info("Preemptive generation cancelled during authorization wait")
                            return
                        
                    except asyncio.TimeoutError:
                        logger.error("Authorization timeout - cancelling preemptive generation")
                        self._preemptive_cancelled = True
                        return
                    
                async def tts_stream_gen():
                    while True:
                        if handle.interrupted or (wait_for_authorization and self._preemptive_cancelled):
                            break

                        try:
                            chunk = await asyncio.wait_for(q.get(), timeout=0.1)
                            if chunk is None:
                                break
                            yield chunk
                        except asyncio.TimeoutError:
                            if handle.interrupted or (wait_for_authorization and self._preemptive_cancelled):
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

            if (
                full_response
                and self.agent
                and getattr(self.agent, "chat_context", None)
            ):
                cascading_metrics_collector.set_agent_response(full_response)
                self.agent.chat_context.add_message(
                        role=ChatRole.ASSISTANT,
                        content=full_response
                )

        finally:
            self._current_tts_task = None
            self._current_llm_task = None
            cascading_metrics_collector.complete_current_turn()
            if not handle.done():
                handle._mark_done()

    async def process_with_llm(self) -> AsyncIterator[str]:
        """
        Process the current chat context with LLM and yield response chunks.
        This method can be called by user implementations to get LLM responses.
        """
        async with self.llm_lock:
            if not self.llm:
                return

            if not self.agent or not getattr(self.agent, "chat_context", None):
                logger.info("Agent not available for LLM processing, exiting")
                return

            cascading_metrics_collector.on_llm_start()
            first_chunk_received = False
            
            agent_session = getattr(self.agent, "session", None) if self.agent else None
            if agent_session:
                agent_session._emit_user_state(UserState.IDLE)
                agent_session._emit_agent_state(AgentState.THINKING)

            async for llm_chunk_resp in self.llm.chat(
                self.agent.chat_context,
                tools=self.agent._tools
            ):
                if self._is_interrupted:
                    logger.info("LLM processing interrupted")
                    break

                if not self.agent or not getattr(self.agent, "chat_context", None):
                    logger.info("Agent context unavailable, stopping LLM processing")
                    break

                if not first_chunk_received:
                    first_chunk_received = True
                    cascading_metrics_collector.on_llm_complete()

                if llm_chunk_resp.metadata and "function_call" in llm_chunk_resp.metadata:
                    func_call = llm_chunk_resp.metadata["function_call"]

                    cascading_metrics_collector.add_function_tool_call(func_call["name"])

                    chat_context = getattr(self.agent, "chat_context", None)
                    if not chat_context:
                        logger.info("Chat context missing while handling function call, aborting")
                        return

                    chat_context.add_function_call(
                        name=func_call["name"],
                        arguments=json.dumps(func_call["arguments"]),
                        call_id=func_call.get(
                            "call_id", f"call_{int(time.time())}")
                    )

                    try:
                        if not self.agent:
                            logger.info("Agent cleaned up before selecting tool, aborting")
                            return

                        tool = next(
                            (t for t in self.agent.tools if is_function_tool(
                                t) and get_tool_info(t).name == func_call["name"]),
                            None
                        )
                    except Exception as e:
                        logger.error(f"Error while selecting tool: {e}")
                        continue

                    if tool:
                        agent_session = getattr(self.agent, "session", None) if self.agent else None
                        if agent_session:
                            agent_session._is_executing_tool = True
                        try:
                            result = await tool(**func_call["arguments"])

                            if isinstance(result, Agent):
                                new_agent = result
                                current_session = self.agent.session
                                
                                logger.info(f"Switching from agent {type(self.agent).__name__} to {type(new_agent).__name__}")

                                if getattr(new_agent, 'inherit_context', True):
                                    logger.info(f"Inheriting context from {type(self.agent).__name__} to {type(new_agent).__name__}")
                                    logger.info(f"Chat context: {self.agent.chat_context.items}")
                                    new_agent.chat_context = self.agent.chat_context
                                    new_agent.chat_context.add_message(
                                        role=ChatRole.SYSTEM,
                                        content=new_agent.instructions,
                                        replace=True
                                    )

                                if hasattr(self.agent, 'on_speech_in'):
                                    current_session.off("on_speech_in", self.agent.on_speech_in)
                                if hasattr(self.agent, 'on_speech_out'):
                                    current_session.off("on_speech_out", self.agent.on_speech_out)

                                new_agent.session = current_session
                                self.agent = new_agent
                                current_session.agent = new_agent

                                if hasattr(current_session.pipeline, 'set_agent'):
                                    current_session.pipeline.set_agent(new_agent)
                                if hasattr(current_session.pipeline, 'set_conversation_flow'):
                                     current_session.pipeline.set_conversation_flow(self)

                                if hasattr(new_agent, 'on_speech_in'):
                                    current_session.on("on_speech_in", new_agent.on_speech_in)
                                if hasattr(new_agent, 'on_speech_out'):
                                    current_session.on("on_speech_out", new_agent.on_speech_out)

                                if hasattr(new_agent, 'on_enter') and asyncio.iscoroutinefunction(new_agent.on_enter):
                                    await new_agent.on_enter()
                                
                                return

                            chat_context = getattr(self.agent, "chat_context", None)
                            if not chat_context:
                                logger.info("Agent chat context missing after tool execution, stopping LLM processing")
                                return
                            chat_context.add_function_output(
                                name=func_call["name"],
                                output=json.dumps(result),
                                call_id=func_call.get(
                                    "call_id", f"call_{int(time.time())}")
                            )

                            async for new_resp in self.llm.chat(chat_context):
                                if self._is_interrupted:
                                    break
                                if new_resp.content:
                                    yield new_resp.content
                        except Exception as e:
                            logger.error(
                                f"Error executing function {func_call['name']}: {e}")
                            continue
                        finally:
                            agent_session = getattr(self.agent, "session", None) if self.agent else None
                            if agent_session:
                                agent_session._is_executing_tool = False
                else:
                    if llm_chunk_resp.content:
                        yield llm_chunk_resp.content

    async def say(self, message: str, handle: UtteranceHandle) -> None:
        """
        Direct TTS synthesis (used for initial messages) and manage handle lifecycle.
        """
        if self.tts:
            cascading_metrics_collector.start_new_interaction("")
            cascading_metrics_collector.set_agent_response(message)

            try:
                await self._synthesize_with_tts(message)
            finally:
                cascading_metrics_collector.complete_current_turn()
                handle._mark_done()

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

        if self.agent and self.agent.session and self.agent.session.current_utterance:
            self.agent.session.current_utterance.interrupt()

        if self.agent.session and self.agent.session.is_background_audio_enabled:
            await self.agent.session.stop_thinking_audio()

        # Cancel any waiting timers
        if self._wait_timer:
            self._wait_timer.cancel()
            self._wait_timer = None
        self._waiting_for_more_speech = False

        if self.tts:
            await self.tts.interrupt()

        if self.avatar and hasattr(self.avatar, 'interrupt'):
            await self.avatar.interrupt()

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

        if self.agent and self.agent.session:
            self.agent.session._pause_wake_up_timer()

        # Ensure audio track exists before callback registration
        if not self.audio_track:
            if hasattr(self.agent.session, "pipeline") and hasattr(self.agent.session.pipeline, "audio_track"):
                self.audio_track = self.agent.session.pipeline.audio_track
            else:
                logger.warning("[ConversationFlow] Audio track not found in pipeline — last audio callback will be skipped.")

        # Define first/last audio byte callbacks
        async def on_first_audio_byte():
            if self.agent.session and self.agent.session.is_background_audio_enabled:
                await self.agent.session.stop_thinking_audio()
            cascading_metrics_collector.on_tts_first_byte()
            cascading_metrics_collector.on_agent_speech_start()

            if self.agent.session:
                self.agent.session._emit_agent_state(AgentState.SPEAKING)
                self.agent.session._emit_user_state(UserState.LISTENING)

        async def on_last_audio_byte():
            if self.agent.session:
                self.agent.session._emit_agent_state(AgentState.IDLE)
                self.agent.session._emit_user_state(UserState.IDLE)
            logger.info("[TTS] Last audio byte processed — Agent and User set to IDLE")

        # Register the callbacks
        self.tts.on_first_audio_byte(on_first_audio_byte)

        if self.audio_track:
            if hasattr(self.audio_track, "on_last_audio_byte"):
                self.audio_track.on_last_audio_byte(on_last_audio_byte)
            else:
                logger.warning(f"[ConversationFlow] Audio track '{type(self.audio_track).__name__}' does not have 'on_last_audio_byte' method — skipping callback registration.")
        else:
            logger.warning("[ConversationFlow] Audio track not initialized — skipping last audio callback registration.")

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
            if self.agent.session and self.agent.session.is_background_audio_enabled:
                await self.agent.session.stop_thinking_audio()

            if self.agent and self.agent.session:
                self.agent.session._reply_in_progress = False
                self.agent.session._reset_wake_up_timer()
            cascading_metrics_collector.on_agent_speech_end()
    
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
            
        if self._vmd_check_task and not self._vmd_check_task.done():
            self._vmd_check_task.cancel()
        self.voice_mail_detector = None
        
        await self._cancel_preemptive_generation()
        
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
        logger.info("Conversation flow cleaned up")