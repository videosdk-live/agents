from __future__ import annotations

from abc import ABC
from typing import Awaitable, Callable, Literal, AsyncIterator, Any, TYPE_CHECKING
import time
import json
import asyncio
from .event_emitter import EventEmitter
from .stt.stt import STT, STTResponse
from .llm.llm import LLM, ResponseChunk
from .llm.chat_context import ChatRole, ImageContent
from .utils import is_function_tool, get_tool_info
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
    from .cascading_pipeline import EOUConfig, InterruptConfig
    
logger = logging.getLogger(__name__)


class ConversationFlow(EventEmitter[Literal["transcription"]], ABC):
    """
    Manages the conversation flow by listening to transcription events.
    """

    def __init__(self, agent: Agent | None = None, stt: STT | None = None, llm: LLM | None = None, tts: TTS | None = None, vad: VAD | None = None, turn_detector: EOU | None = None, denoise: Denoise | None = None, avatar: Any | None = None) -> None:
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

        self._accumulated_transcript = ""
        self._waiting_for_more_speech = False
        self._wait_timer: asyncio.TimerHandle | None = None
        self._transcript_processing_lock = asyncio.Lock()

        self.min_speech_wait_timeout = 0.5
        self.max_speech_wait_timeout = 0.8
        self.mode: Literal["ADAPTIVE", "DEFAULT"] = "DEFAULT"
        self.eou_certainty_threshold = 0.85 

        self.interrupt_mode: Literal["VAD_ONLY", "STT_ONLY", "HYBRID"] = "HYBRID"
        self.interrupt_min_duration = 0.5
        self.interrupt_min_words = 1

        self.false_interrupt_pause_duration = 2.0
        self.resume_on_false_interrupt = False
        self._is_in_false_interrupt_pause = False
        self._false_interrupt_timer: asyncio.TimerHandle | None = None

        self._false_interrupt_paused_speech = False
        self._is_user_speaking = False

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
          
        # Conversational Graph
        self.conversational_graph = None
        
        # Context truncation
        self.max_context_items: int | None = None


    def apply_flow_config(self, eou_config: "EOUConfig", interrupt_config: "InterruptConfig") -> None:
        """Override default timing/interaction parameters using pipeline config."""
        self.mode = eou_config.mode
        self.min_speech_wait_timeout = eou_config.min_max_speech_wait_timeout[0]
        self.max_speech_wait_timeout = eou_config.min_max_speech_wait_timeout[1]
        self.interrupt_mode = interrupt_config.mode
        self.interrupt_min_duration = interrupt_config.interrupt_min_duration
        self.interrupt_min_words = interrupt_config.interrupt_min_words
        self.false_interrupt_pause_duration = interrupt_config.false_interrupt_pause_duration
        self.resume_on_false_interrupt = interrupt_config.resume_on_false_interrupt
        
    def _update_preemptive_generation_flag(self) -> None:
        """Update the preemptive generation flag based on current STT instance"""
        self._enable_preemptive_generation = getattr(self.stt, 'enable_preemptive_generation', False) if self.stt else False
        cascading_metrics_collector.set_preemptive_generation_enabled()

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
        """Handle VAD events with interruption logic"""
        
        if (self.agent and self.agent.session and self.agent.session.agent_state == AgentState.SPEAKING):
            
            if vad_response.event_type == VADEventType.START_OF_SPEECH:
                if not hasattr(self, '_interruption_check_task') or self._interruption_check_task.done():
                    logger.info("User started speaking during agent response, initiating interruption monitoring")
                    self._interruption_check_task = asyncio.create_task(
                        self._monitor_interruption_duration()
                    )
                return
                
            elif vad_response.event_type == VADEventType.END_OF_SPEECH:
                if hasattr(self, '_interruption_check_task') and not self._interruption_check_task.done():
                    logger.info("User stopped speaking, cancelling interruption check")
                    self._interruption_check_task.cancel()
                return
        
        if vad_response.event_type == VADEventType.START_OF_SPEECH:
            self._is_user_speaking = True            
            if self._waiting_for_more_speech:
                logger.debug("User continued speaking, cancelling wait timer")
                await self._handle_continued_speech()
                
            await self.on_speech_started()
            
        elif vad_response.event_type == VADEventType.END_OF_SPEECH:
            self._is_user_speaking = False
            self.on_speech_stopped()


    async def _monitor_interruption_duration(self) -> None:
        """
        Monitor user speech duration during agent response.
        Triggers interruption if speech exceeds the configured threshold.
        """
        logger.debug(f"Interruption monitoring started (mode={self.interrupt_mode}, threshold={self.interrupt_min_duration}s)")
        
        if self.interrupt_mode not in ("VAD_ONLY", "HYBRID"):
            logger.debug(f"Interruption mode is {self.interrupt_mode}, VAD monitoring not active")
            return
        
        try:
            await asyncio.sleep(self.interrupt_min_duration)
            
            if (self.agent.session and self.agent.session.current_utterance and self.agent.session.current_utterance.is_interruptible):
                logger.info(f"User speech duration exceeded {self.interrupt_min_duration}s threshold, triggering interruption")
                await self._trigger_interruption()
            else:
                logger.debug("Interruption threshold reached but utterance is not interruptible")
                
        except asyncio.CancelledError:
            logger.debug("Interruption monitoring cancelled (user stopped speaking)")


    async def _handle_continued_speech(self) -> None:
        """Handle when user continues speaking while we're waiting"""
        if self._wait_timer:
            self._wait_timer.cancel()
            self._wait_timer = None
        
        self._waiting_for_more_speech = False

    async def on_stt_transcript(self, stt_response: STTResponse) -> None:
        """Handle STT transcript events with enhanced EOU logic"""
       
        utterance = self.agent.session.current_utterance if self.agent and self.agent.session else None
        if utterance and not utterance.is_interruptible and self.agent.session.agent_state == AgentState.SPEAKING:
            logger.info(f"Agent is playing non-interruptible message. Ignoring user speech until message completes.")
            return

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
            if cascading_metrics_collector.data.current_turn:
                cascading_metrics_collector.on_stt_preflight_end()
            await self._handle_preflight_transcript(text)
            
        
        elif stt_response.event_type == SpeechEventType.FINAL:
            if cascading_metrics_collector.data.current_turn:
                cascading_metrics_collector.data.current_turn.stt_preemptive_generation_occurred = False
            user_text = stt_response.data.text
            
            if self._enable_preemptive_generation:

                if cascading_metrics_collector.data.current_turn:
                    cascading_metrics_collector.on_stt_complete()
                    cascading_metrics_collector.data.current_turn.stt_preemptive_generation_occurred = True
                await self._authorize_or_process_final_transcript(user_text)
                
            else:
                await self._process_transcript_with_eou(user_text)
                
            
        elif stt_response.event_type == SpeechEventType.INTERIM:
            if cascading_metrics_collector.data.current_turn and self._enable_preemptive_generation: 
                cascading_metrics_collector.on_stt_interim_end()

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
            if self._accumulated_transcript:
                self._accumulated_transcript += " " + new_transcript
            else:
                self._accumulated_transcript = new_transcript

            if self.mode == 'DEFAULT':
                logger.info(f"DEFAULT Mode, using min speech wait timeout seconds {self.min_speech_wait_timeout} and max speech wait timeout seconds {self.max_speech_wait_timeout}")
                delay = self.min_speech_wait_timeout
                if self.turn_detector:
                    logger.info(f"Turn detector is available, getting EOU probability")
                    cascading_metrics_collector.on_eou_start()
                    eou_probability = self.turn_detector.get_eou_probability(self.agent.chat_context)
                    cascading_metrics_collector.on_eou_complete()
                    logger.info(f"EOU probability: {eou_probability}")
                    if eou_probability < self.eou_certainty_threshold:
                        logger.info(f"EOU probability is less than the threshold, using max speech wait timeout")
                        delay = self.max_speech_wait_timeout
                logger.info(f"Using delay: {delay} seconds")
                await self._wait_for_additional_speech(delay)

            elif self.mode == 'ADAPTIVE':
                logger.info(f"ADAPTIVE Mode, using min speech wait timeout seconds {self.min_speech_wait_timeout} and max speech wait timeout seconds {self.max_speech_wait_timeout}")
                delay = self.min_speech_wait_timeout
                if self.turn_detector:
                    logger.info(f"Turn detector is available, getting EOU probability")
                    cascading_metrics_collector.on_eou_start()
                    eou_probability = self.turn_detector.get_eou_probability(self.agent.chat_context)
                    cascading_metrics_collector.on_eou_complete()
                    logger.info(f"EOU probability: {eou_probability}")
                    logger.info(f"Calculating delay using sliding scale {self.min_speech_wait_timeout} to {self.max_speech_wait_timeout}")
                    delay_range = self.max_speech_wait_timeout - self.min_speech_wait_timeout
                    wait_factor = 1.0 - eou_probability  
                    logger.info(f"Wait factor: {wait_factor}")
                    delay = self.min_speech_wait_timeout + (delay_range * wait_factor)
                    logger.info(f"Calculated delay: {delay}")
                await self._wait_for_additional_speech(delay)

        

    async def _check_end_of_utterance(self, transcript: str) -> bool:
        """Check if the current transcript represents end of utterance"""
        if not self.turn_detector:
            return True
        
        temp_context = self.agent.chat_context.copy()
        temp_context.add_message(role=ChatRole.USER, content=transcript)
        
        cascading_metrics_collector.on_eou_start()
        is_eou = self.turn_detector.detect_end_of_utterance(temp_context)
        cascading_metrics_collector.on_eou_complete()
        
        return is_eou

    async def _wait_for_additional_speech(self, delay: float) -> None:
        """Wait for additional speech within the timeout period"""
        logger.info(f"Called _wait_for_additional_speech method, Waiting for additional speech for {delay} seconds")

        if self._waiting_for_more_speech:
            if self._wait_timer:
                self._wait_timer.cancel()
        
        self._waiting_for_more_speech = True
        
        loop = asyncio.get_event_loop()
        self._wait_timer = loop.call_later(
            delay,
            lambda: asyncio.create_task(self._on_speech_timeout())
        )

    async def _on_speech_timeout(self) -> None:
        """Handle timeout when no additional speech is detected"""
        async with self._transcript_processing_lock:
            if not self._waiting_for_more_speech:
                return
            
            self._waiting_for_more_speech = False
            self._wait_timer = None
            
            await self._finalize_transcript_and_respond()

    async def _finalize_transcript_and_respond(self) -> None:
        """Finalize the accumulated transcript and generate response"""
        if not self._accumulated_transcript.strip():
            return
        
        final_transcript = self._accumulated_transcript.strip()
        logger.info(f"Finalizing transcript: '{final_transcript}'")
        
        self._accumulated_transcript = ""
        
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

        if self.conversational_graph:
            final_user_text = self.conversational_graph.handle_input(user_text)

        self.agent.chat_context.add_message(
            role=ChatRole.USER,
            content=final_user_text
        )

        if self.agent and self.agent.session:
            if self.agent.session.current_utterance and not self.agent.session.current_utterance.done():
                if self.agent.session.current_utterance.is_interruptible:
                    self.agent.session.current_utterance.interrupt()
                else:
                    logger.info("Current utterance is not interruptible. Skipping interruption in cascading pipeline.")
            
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
            if self.vad:
                original_vad_handler = self.on_vad_event
                self.on_vad_event = lambda x: None
            
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
                metadata = None
                
                try:
                    async for chunk in llm_stream:
                        if handle.interrupted or (wait_for_authorization and self._preemptive_cancelled):
                            logger.info("LLM collection interrupted")
                            await q.put(None)
                            return "".join(response_parts)

                        content = chunk
                        chunk_metadata = None

                        if hasattr(chunk, "content"):
                            content = chunk.content
                            if hasattr(chunk, "metadata"):
                                chunk_metadata = chunk.metadata

                        if content:
                            response_parts.append(content)
                            await q.put(content)
                        if chunk_metadata:
                            metadata = chunk_metadata
                        
                        self._partial_response = "".join(response_parts)

                    if not handle.interrupted:
                        await q.put(None)
                    
                    if self.conversational_graph and metadata.get("graph_response"):
                        _ = await self.conversational_graph.handle_decision(self.agent, metadata.get("graph_response"))
                    return "".join(response_parts)
                        
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
            
            if self.max_context_items:
                current_items = len(self.agent.chat_context.items)
                if current_items > self.max_context_items:
                    try:
                        logger.info(f"Chat Context Truncating from {current_items} to {self.max_context_items} items (max_context_items={self.max_context_items})")
                        self.agent.chat_context.truncate(self.max_context_items)
                        logger.info(f"Chat Context Truncation complete. Final size: {len(self.agent.chat_context.items)} items")
                    except Exception as e:
                        logger.error(f"Chat Context Error during truncation: {e}", exc_info=True)
                else:
                    logger.debug(f"Context size {current_items} is within limit (max_context_items={self.max_context_items})")

            cascading_metrics_collector.on_llm_start()
            first_chunk_received = False
            
            agent_session = getattr(self.agent, "session", None) if self.agent else None
            if agent_session:
                agent_session._emit_user_state(UserState.IDLE)
                agent_session._emit_agent_state(AgentState.THINKING)

            async for llm_chunk_resp in self.llm.chat(
                self.agent.chat_context,
                tools=self.agent._tools,
                conversational_graph=self.conversational_graph if self.conversational_graph else None
            ):
                if llm_chunk_resp.metadata and "usage" in llm_chunk_resp.metadata:
                    cascading_metrics_collector.set_llm_usage(llm_chunk_resp.metadata["usage"])

                if self._is_interrupted:
                    logger.info("LLM processing interrupted")
                    break

                if not self.agent or not getattr(self.agent, "chat_context", None):
                    logger.info("Agent context unavailable, stopping LLM processing")
                    break

                if not first_chunk_received:
                    first_chunk_received = True
                    cascading_metrics_collector.on_llm_first_token()

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

                            async for new_resp in self.llm.chat(
                                chat_context,
                                tools=self.agent.tools,
                                conversational_graph=self.conversational_graph if self.conversational_graph else None
                            ):
                                if self._is_interrupted:
                                    break
                                if new_resp:
                                    yield ResponseChunk(new_resp.content, new_resp.metadata, new_resp.role)
                        except Exception as e:
                            logger.error(
                                f"Error executing function {func_call['name']}: {e}")
                            continue
                        finally:
                            agent_session = getattr(self.agent, "session", None) if self.agent else None
                            if agent_session:
                                agent_session._is_executing_tool = False
                else:
                    if llm_chunk_resp:
                        yield ResponseChunk(llm_chunk_resp.content, llm_chunk_resp.metadata, llm_chunk_resp.role)

            if not self._is_interrupted:
                cascading_metrics_collector.on_llm_complete()

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
            if response_chunk.content:
                full_response += response_chunk.content

        if full_response:
            cascading_metrics_collector.set_agent_response(full_response)
            cascading_metrics_collector.complete_current_turn()
            global_event_emitter.emit("text_response", {"text": full_response})

    async def run(self, transcript: str) -> AsyncIterator[str]:
        """
        Main conversation loop: handle a user turn.
        Users should implement this method to preprocess transcripts and yield response chunks.
        """
        if not cascading_metrics_collector.data.current_turn:
            cascading_metrics_collector.start_new_interaction(transcript)
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
        
        if self.agent and self.agent.session:
            self.agent.session._emit_user_state(UserState.SPEAKING)

    def on_speech_stopped_stt(self, event_data: Any) -> None:
        pass

    async def handle_stt_event(self, text: str) -> None:
        """Handle STT event"""
        if not text or not text.strip():
            return
        
        word_count = len(text.strip().split())
        logger.info(f"handle_stt_event: Word count: {word_count}")
        
        if self.resume_on_false_interrupt and self._is_in_false_interrupt_pause and word_count >= self.interrupt_min_words:
            logger.info(f"[FALSE_INTERRUPT] STT transcript received while in paused state: '{text}' ({word_count} words). Confirming real interruption.")
            self._cancel_false_interrupt_timer()
            self._is_in_false_interrupt_pause = False
            self._false_interrupt_paused_speech = False
            logger.info("[FALSE_INTERRUPT] Clearing audio buffers and finalizing interruption.")
            await self._interrupt_tts()
            return
        
        if self.interrupt_mode in ("STT_ONLY", "HYBRID"):
            if word_count >= self.interrupt_min_words:
                if self.agent.session and self.agent.session.current_utterance and self.agent.session.current_utterance.is_interruptible:
                    await self._trigger_interruption()
                else:
                    logger.info("Interruption not allowed for the current utterance.")
            
    async def _trigger_interruption(self) -> None:
            """Trigger interruption once, respecting the utterance's interruptible flag."""
            logger.info("Interruption triggered")
            if self._is_interrupted:
                logger.info("Already interrupted, skipping")
                return

            utterance = self.agent.session.current_utterance if self.agent and self.agent.session else None
            if utterance and not utterance.is_interruptible:
                logger.info("Interruption is disabled for the current utterance. Ignoring.")
                return
                
            self._is_interrupted = True 

            can_resume = self.resume_on_false_interrupt and self.tts and self.tts.can_pause
            
            if can_resume:
                logger.info(f"[FALSE_INTERRUPT] Pausing TTS for potential resume. (resume_on_false_interrupt={self.resume_on_false_interrupt}, can_pause={self.tts.can_pause if self.tts else False})")
                self._false_interrupt_paused_speech = True
                self._is_in_false_interrupt_pause = True
                await self.tts.pause()
                self._start_false_interrupt_timer()
            else:
                logger.info("performing full interruption.")
                await self._interrupt_tts()

    def _start_false_interrupt_timer(self):
        """Starts a timer to detect if an interruption was just a brief false interrupt."""
        if self._false_interrupt_timer:
            logger.info("[FALSE_INTERRUPT] Cancelling existing timer before starting a new one.")
            self._false_interrupt_timer.cancel()
            
        if self.false_interrupt_pause_duration is None:
            logger.info("[FALSE_INTERRUPT] Timeout is None, skipping timer.")
            return

        logger.info(f"[FALSE_INTERRUPT] Starting timer for {self.false_interrupt_pause_duration}s. If no STT transcript is received within this time, speech will resume.")
        loop = asyncio.get_event_loop()
        self._false_interrupt_timer = loop.call_later(
            self.false_interrupt_pause_duration,
            lambda: asyncio.create_task(self._on_false_interrupt_timeout())
        )

    def _cancel_false_interrupt_timer(self):
        """Cancels the false-interrupt timer if it's running."""
        if self._false_interrupt_timer:
            logger.info("[FALSE_INTERRUPT] Cancelling false-interrupt timer - real interruption confirmed.")
            self._false_interrupt_timer.cancel()
            self._false_interrupt_timer = None

    async def _on_false_interrupt_timeout(self):
        """Called when the user remains silent after an interruption."""
        logger.info(f"[FALSE_INTERRUPT] Timeout reached after {self.false_interrupt_pause_duration}s. User did not follow up with speech.")
        self._false_interrupt_timer = None

        if self._is_user_speaking:
            logger.info("[FALSE_INTERRUPT] User is still speaking at timeout. Confirming as a real interruption.")
            self._is_in_false_interrupt_pause = False
            self._false_interrupt_paused_speech = False
            await self._interrupt_tts()
            return

        if self._is_in_false_interrupt_pause and self.tts and self.tts.can_pause:
            logger.info("[FALSE_INTERRUPT] Resuming agent speech from paused position - false interruption detected.")
            self._is_interrupted = False
            self._is_in_false_interrupt_pause = False
            self._false_interrupt_paused_speech = False
            await self.tts.resume()
        else:
            if self._is_interrupted or self._false_interrupt_paused_speech:
                logger.info(f"[FALSE_INTERRUPT] Cannot resume (is_in_false_interrupt_pause={self._is_in_false_interrupt_pause}, can_pause={self.tts.can_pause if self.tts else False}). Finalizing interruption.")
                await self._interrupt_tts()
            else:
                logger.info("[FALSE_INTERRUPT] Timeout reached but no paused state found. No action needed.")        
    
    async def on_speech_started(self) -> None:       
            cascading_metrics_collector.on_user_speech_start()

            if self.user_speech_callback:
                self.user_speech_callback()

            if self._stt_started:
                self._stt_started = False
            
            utterance = self.agent.session.current_utterance if self.agent and self.agent.session else None
            if utterance and not utterance.is_interruptible:
                logger.info("Interruption is disabled for the current utterance. Not interrupting.")
                if self.agent and self.agent.session:
                    self.agent.session._emit_user_state(UserState.SPEAKING)
                return

            self._cancel_false_interrupt_timer()
            
            can_resume = self.resume_on_false_interrupt and self.tts and self.tts.can_pause

            if self._false_interrupt_paused_speech:
                logger.info("User continued speaking, confirming interruption of paused speech.")
                self._false_interrupt_paused_speech = False 
                await self._interrupt_tts() 
            elif self.agent and self.agent.session and self.agent.session.agent_state == AgentState.SPEAKING:
                if can_resume:
                    logger.info("Pausing agent speech for potential hesitation (will resume if user stops quickly).")
                    await self._trigger_interruption()
                else:
                    await self._interrupt_tts()
            
            if self.agent and self.agent.session:
                self.agent.session._emit_user_state(UserState.SPEAKING)

    async def _interrupt_tts(self) -> None:
        self._is_interrupted = True        
        self._cancel_false_interrupt_timer()
        self._is_in_false_interrupt_pause = False
        self._false_interrupt_paused_speech = False

        if self.agent and self.agent.session and self.agent.session.current_utterance:
            if self.agent.session.current_utterance.is_interruptible:
                self.agent.session.current_utterance.interrupt()
            else:
                logger.info("Cannot interrupt non-interruptible utterance in _interrupt_tts")

        if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
            await self.agent.session.stop_thinking_audio()

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
            for task in tasks_to_cancel:
                task.cancel()
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            
        self._partial_response = ""
        self._is_interrupted = False

        cascading_metrics_collector.on_interrupted()

    async def _cancel_llm(self) -> None:
        """Cancel LLM generation"""
        try:
            await self.llm.cancel_current_generation()
            cascading_metrics_collector.on_llm_complete()
        except Exception as e:
            logger.error(f"LLM cancellation failed: {e}")

    def on_speech_stopped(self) -> None:
        if not self._stt_started:
            cascading_metrics_collector.on_stt_start()
            self._stt_started = True

        cascading_metrics_collector.on_user_speech_end()
        
        if self.agent and self.agent.session:
            self.agent.session._emit_user_state(UserState.IDLE)

    async def _synthesize_with_tts(self, response_gen: AsyncIterator[str] | str) -> None:
        """
        Stream LLM response directly to TTS.
        """
        if not self.tts:
            return

        if self.agent and self.agent.session:
            self.agent.session._pause_wake_up_timer()

        if not self.audio_track:
            if self.agent and self.agent.session and hasattr(self.agent.session, "pipeline") and hasattr(self.agent.session.pipeline, "audio_track"):
                self.audio_track = self.agent.session.pipeline.audio_track
            else:
                logger.warning("[ConversationFlow] Audio track not found in pipeline — last audio callback will be skipped.")

        if self.audio_track and hasattr(self.audio_track, "enable_audio_input"):
            # Require manual re-enable so old audio never bleeds into the next utterance.
            self.audio_track.enable_audio_input(manual_control=True)

        async def on_first_audio_byte():
            if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
                await self.agent.session.stop_thinking_audio()
            cascading_metrics_collector.on_tts_first_byte()
            cascading_metrics_collector.on_agent_speech_start()

            if self.agent and self.agent.session:
                self.agent.session._emit_agent_state(AgentState.SPEAKING)
                self.agent.session._emit_user_state(UserState.LISTENING)

        async def on_last_audio_byte():
            if self.agent and self.agent.session:
                self.agent.session._emit_agent_state(AgentState.IDLE)
                self.agent.session._emit_user_state(UserState.IDLE)
            logger.info("[TTS] Last audio byte processed — Agent and User set to IDLE")
            cascading_metrics_collector.on_agent_speech_end()
            cascading_metrics_collector.complete_current_turn()

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
            async def counting_wrapper(iterator: AsyncIterator[str]):
                async for chunk in iterator:
                    if chunk:
                        # Count characters and update metrics
                        cascading_metrics_collector.add_tts_characters(len(chunk))
                    yield chunk
            await self.tts.synthesize(counting_wrapper(response_iterator))

        finally:
            if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
                await self.agent.session.stop_thinking_audio()

            if self.agent and self.agent.session:
                self.agent.session._reply_in_progress = False
                self.agent.session._reset_wake_up_timer()

    
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