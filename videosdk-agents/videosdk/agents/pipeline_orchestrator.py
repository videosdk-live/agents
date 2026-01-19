from __future__ import annotations

from typing import Literal, TYPE_CHECKING, Any, AsyncIterator
import asyncio
import uuid
import logging
import av
from .event_emitter import EventEmitter
from .speech_understanding import SpeechUnderstanding
from .content_generation import ContentGeneration
from .speech_generation import SpeechGeneration
from .stt.stt import STT
from .llm.llm import LLM
from .tts.tts import TTS
from .vad import VAD
from .eou import EOU
from .denoise import Denoise
from .llm.chat_context import ChatRole, ImageContent
from .utterance_handle import UtteranceHandle
from .utils import UserState, AgentState
from .voice_mail_detector import VoiceMailDetector

if TYPE_CHECKING:
    from .agent import Agent
    from .knowledge_base.base import KnowledgeBase

logger = logging.getLogger(__name__)


class PipelineOrchestrator(EventEmitter[Literal[
    "transcript_ready", 
    "content_generated", 
    "synthesis_complete",
    "voicemail_result",
    "error"
]]):
    """
    Orchestrates the execution of speech understanding, content generation, and speech generation components.
    
    Supports various component chains:
    1. Full Chain: VAD → STT → TurnD → LLM → TTS
    2. No TTS: VAD → STT → TurnD → LLM (text output)
    3. No STT: LLM → TTS (text input)
    4. Hybrid: VAD → STT → [User Processing] → TTS
    
    Events:
    - transcript_ready: Transcript is ready for processing
    - content_generated: LLM has generated content
    - synthesis_complete: TTS synthesis is complete
    - voicemail_result: Voicemail detection result
    - error: Error occurred in pipeline
    """
    
    def __init__(
        self,
        agent: Agent | None = None,
        stt: STT | None = None,
        llm: LLM | None = None,
        tts: TTS | None = None,
        vad: VAD | None = None,
        turn_detector: EOU | None = None,
        denoise: Denoise | None = None,
        avatar: Any | None = None,
        mode: Literal["ADAPTIVE", "DEFAULT"] = "DEFAULT",
        min_speech_wait_timeout: tuple[float, float] = (0.5, 0.8),
        interrupt_mode: Literal["VAD_ONLY", "STT_ONLY", "HYBRID"] = "HYBRID",
        interrupt_min_duration: float = 0.5,
        interrupt_min_words: int = 2,
        false_interrupt_pause_duration: float = 2.0,
        resume_on_false_interrupt: bool = False,
        conversational_graph: Any | None = None,
        max_context_items: int | None = None,
        voice_mail_detector: VoiceMailDetector | None = None,
    ) -> None:
        super().__init__()
        
        self.agent = agent
        self.avatar = avatar
        self.conversational_graph = conversational_graph
        self.voice_mail_detector = voice_mail_detector
        self.voice_mail_detection_done = False
        self._vmd_buffer = ""
        self._vmd_check_task: asyncio.Task | None = None
        
        # Interruption configuration
        self.interrupt_mode = interrupt_mode
        self.interrupt_min_duration = interrupt_min_duration
        self.interrupt_min_words = interrupt_min_words
        self.false_interrupt_pause_duration = false_interrupt_pause_duration
        self.resume_on_false_interrupt = resume_on_false_interrupt
        
        # Interruption state
        self._is_interrupted = False
        self._is_in_false_interrupt_pause = False
        self._false_interrupt_paused_speech = False
        self._false_interrupt_timer: asyncio.TimerHandle | None = None
        self._is_user_speaking = False
        self._interruption_check_task: asyncio.Task | None = None
        
        # Component modules
        self.speech_understanding: SpeechUnderstanding | None = None
        self.content_generation: ContentGeneration | None = None
        self.speech_generation: SpeechGeneration | None = None
        
        # Initialize components based on what's available
        if stt or vad or turn_detector:
            self.speech_understanding = SpeechUnderstanding(
                agent=agent,
                stt=stt,
                vad=vad,
                turn_detector=turn_detector,
                denoise=denoise,
                mode=mode,
                min_speech_wait_timeout=min_speech_wait_timeout[0],
                max_speech_wait_timeout=min_speech_wait_timeout[1],
            )
            # Setup event listeners with sync wrappers
            self.speech_understanding.on("transcript_final", self._wrap_async(self._on_transcript_final))
            self.speech_understanding.on("transcript_preflight", self._wrap_async(self._on_transcript_preflight))
            self.speech_understanding.on("transcript_interim", self._wrap_async(self._on_transcript_interim))
            self.speech_understanding.on("speech_started", self._wrap_async(self._on_speech_started))
            self.speech_understanding.on("speech_stopped", self._wrap_async(self._on_speech_stopped))
            self.speech_understanding.on("turn_resumed", self._wrap_async(self._on_turn_resumed))
        
        if llm:
            self.content_generation = ContentGeneration(
                agent=agent,
                llm=llm,
                conversational_graph=conversational_graph,
                max_context_items=max_context_items,
            )
            # Setup event listeners
            self.content_generation.on("generation_started", lambda data: logger.info("Content generation started"))
            self.content_generation.on("generation_chunk", lambda data: None)
            self.content_generation.on("generation_complete", lambda data: logger.info("Content generation complete"))
        
        if tts:
            self.speech_generation = SpeechGeneration(
                agent=agent,
                tts=tts,
                avatar=avatar,
            )
            # Setup event listeners
            self.speech_generation.on("synthesis_started", lambda data: logger.info("Speech synthesis started"))
            self.speech_generation.on("first_audio_byte", lambda data: logger.info("First audio byte ready"))
            self.speech_generation.on("last_audio_byte", lambda data: logger.info("Synthesis complete"))
            self.speech_generation.on("synthesis_interrupted", lambda data: logger.info("Synthesis interrupted"))
        
        # Generation tasks
        self._current_generation_task: asyncio.Task | None = None
        self._partial_response = ""
        
        # Preemptive generation
        self._preemptive_generation_task: asyncio.Task | None = None
        self._preemptive_authorized = asyncio.Event()
        self._preemptive_cancelled = False
    
    def _wrap_async(self, async_func):
        """
        Wrap an async function to be compatible with EventEmitter's sync-only handlers.
        
        Args:
            async_func: The async function to wrap
            
        Returns:
            A sync function that schedules the async function as a task
        """
        def sync_wrapper(*args, **kwargs):
            asyncio.create_task(async_func(*args, **kwargs))
        return sync_wrapper
    
    def set_audio_track(self, audio_track: Any) -> None:
        """Set audio track for TTS output"""
        if self.speech_generation:
            self.speech_generation.set_audio_track(audio_track)
    
    def set_voice_mail_detector(self, detector: VoiceMailDetector | None) -> None:
        """Configure voicemail detection"""
        self.voice_mail_detector = detector
        self.voice_mail_detection_done = False
        self._vmd_buffer = ""
    
    async def start(self) -> None:
        """Start all components"""
        if self.speech_understanding:
            await self.speech_understanding.start()
        if self.content_generation:
            await self.content_generation.start()
        if self.speech_generation:
            await self.speech_generation.start()
        
        logger.info("PipelineOrchestrator started")
    
    async def process_audio(self, audio_data: bytes) -> None:
        """
        Process incoming audio through the pipeline.
        
        Args:
            audio_data: Raw audio bytes
        """
        if self.speech_understanding:
            await self.speech_understanding.process_audio(audio_data)
    
    async def process_text(self, text: str) -> None:
        """
        Process text input directly (bypasses STT).
        
        Args:
            text: User text input
        """
        if not self.agent:
            logger.warning("No agent available for text processing")
            return
        
        self.agent.chat_context.add_message(
            role=ChatRole.USER,
            content=text
        )
        
        if self.content_generation:
            full_response = ""
            async for response_chunk in self.content_generation.generate(text, self.agent.knowledge_base):
                if response_chunk.content:
                    full_response += response_chunk.content
            
            if full_response:
                self.agent.chat_context.add_message(
                    role=ChatRole.ASSISTANT,
                    content=full_response
                )
                
                self.emit("content_generated", {"text": full_response})
                
                if self.speech_generation:
                    await self.speech_generation.synthesize(full_response)
                
                self.emit("synthesis_complete", {})
    
    async def inject_text_to_llm(self, text: str) -> None:
        """
        Inject processed text into LLM for generation (hybrid mode).
        
        Args:
            text: Processed text to send to LLM
        """
        await self.process_text(text)
    
    async def inject_text_to_tts(self, text: str) -> None:
        """
        Inject text directly to TTS (bypassing LLM).
        
        Args:
            text: Text to synthesize
        """
        if self.speech_generation:
            await self.speech_generation.synthesize(text)
            self.emit("synthesis_complete", {})
    
    def get_latest_transcript(self) -> str:
        """Get the latest accumulated transcript (for hybrid scenarios)"""
        if self.speech_understanding:
            return self.speech_understanding._accumulated_transcript
        return ""
    
    async def _on_transcript_final(self, data: dict) -> None:
        """Handle final transcript from speech understanding"""
        text = data["text"]
        is_preemptive = data.get("is_preemptive", False)
        
        if self.voice_mail_detector and not self.voice_mail_detection_done and text.strip():
            self._vmd_buffer += f" {text}"
            if not self._vmd_check_task:
                logger.info("Starting Voice Mail Detection Timer")
                self._vmd_check_task = asyncio.create_task(self._run_vmd_check())
        
        self.emit("transcript_ready", {
            "text": text,
            "is_final": True,
            "is_preemptive": is_preemptive,
            "metadata": data.get("metadata", {})
        })
        
        if is_preemptive:
            await self._handle_preemptive_final(text)
        else:
            await self._process_final_transcript(text)
    
    async def _on_transcript_preflight(self, data: dict) -> None:
        """Handle preflight transcript for preemptive generation"""
        preflight_text = data["text"]
        
        if self.agent and self.content_generation:
            user_text = preflight_text
            if self.agent.knowledge_base:
                kb_context = await self.agent.knowledge_base.process_query(preflight_text)
                if kb_context:
                    user_text = f"{kb_context}\n\nUser: {preflight_text}"
            
            self.agent.chat_context.add_message(
                role=ChatRole.USER,
                content=user_text
            )
            
            if self.agent.session:
                if self.agent.session.current_utterance and not self.agent.session.current_utterance.done():
                    self.agent.session.current_utterance.interrupt()
                
                handle = UtteranceHandle(utterance_id=f"utt_{uuid.uuid4().hex[:8]}")
                self.agent.session.current_utterance = handle
            else:
                handle = UtteranceHandle(utterance_id="utt_fallback")
                handle._mark_done()
            
            self._preemptive_authorized.clear()
            self._preemptive_cancelled = False
            self._preemptive_generation_task = asyncio.create_task(
                self._generate_and_synthesize(user_text, handle, wait_for_authorization=True)
            )
    
    async def _handle_preemptive_final(self, final_text: str) -> None:
        """Handle final transcript when preemptive generation is active"""
        if not self.speech_understanding:
            return
        
        if self.speech_understanding.check_preemptive_match(final_text):
            logger.info("Preemptive generation MATCH - authorizing playback")
            
            self._preemptive_authorized.set()
            
            if self._preemptive_generation_task:
                try:
                    await asyncio.wait_for(self._preemptive_generation_task, timeout=30.0)
                    logger.info("Preemptive generation completed successfully")
                except asyncio.TimeoutError:
                    logger.error("Preemptive playback timeout")
                except Exception as e:
                    logger.error(f"Error in preemptive playback: {e}")
        else:
            logger.info("Preemptive generation MISMATCH - cancelling")
            
            await self._cancel_preemptive_generation()
            
            if self.agent and self.agent.chat_context.messages:
                if self.agent.chat_context.messages[-1].role == ChatRole.USER:
                    self.agent.chat_context.messages.pop()
            
            await self._process_final_transcript(final_text)
        
        self.speech_understanding.clear_preemptive_state()
        self._preemptive_generation_task = None
    
    async def _on_transcript_interim(self, data: dict) -> None:
        """Handle interim transcript"""
        pass
    
    async def _on_speech_started(self, data: dict) -> None:
        """Handle speech started event"""
        self._is_user_speaking = True
        
        if self.agent and self.agent.session and self.agent.session.agent_state == AgentState.SPEAKING:
            if self._interruption_check_task is None or self._interruption_check_task.done():
                logger.info("User started speaking during agent response, initiating interruption monitoring")
                self._interruption_check_task = asyncio.create_task(
                    self._monitor_interruption_duration()
                )
    
    async def _on_speech_stopped(self, data: dict) -> None:
        """Handle speech stopped event"""
        self._is_user_speaking = False
        
        if self._interruption_check_task is not None and not self._interruption_check_task.done():
            logger.info("User stopped speaking, cancelling interruption check")
            self._interruption_check_task.cancel()
    
    async def _on_turn_resumed(self, data: dict) -> None:
        """Handle turn resumed event"""
        await self._cancel_preemptive_generation()
    
    async def _process_final_transcript(self, user_text: str) -> None:
        """Process final transcript through the full pipeline"""
        if not self.agent:
            logger.warning("No agent available")
            return
        
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
        
        if self.agent.session:
            if self.agent.session.current_utterance and not self.agent.session.current_utterance.done():
                if self.agent.session.current_utterance.is_interruptible:
                    self.agent.session.current_utterance.interrupt()
                else:
                    logger.info("Current utterance is not interruptible")
            
            handle = UtteranceHandle(utterance_id=f"utt_{uuid.uuid4().hex[:8]}")
            self.agent.session.current_utterance = handle
        else:
            handle = UtteranceHandle(utterance_id="utt_fallback")
            handle._mark_done()
        
        asyncio.create_task(self._generate_and_synthesize(final_user_text, handle))
    
    async def _generate_and_synthesize(
        self, 
        user_text: str, 
        handle: UtteranceHandle,
        wait_for_authorization: bool = False
    ) -> None:
        """Generate LLM response and synthesize with TTS"""
        self._is_interrupted = False
        full_response = ""
        self._partial_response = ""
        
        try:
            if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
                await self.agent.session.start_thinking_audio()
            
            if not self.content_generation:
                logger.warning("No content generation available")
                return
            
            llm_stream = self.content_generation.generate(user_text)
            
            q = asyncio.Queue(maxsize=50)
            
            async def collector():
                """Collect LLM chunks"""
                response_parts = []
                try:
                    async for chunk in llm_stream:
                        if handle.interrupted or (wait_for_authorization and self._preemptive_cancelled):
                            logger.info("LLM collection interrupted")
                            await q.put(None)
                            return "".join(response_parts)
                        
                        content = chunk.content if hasattr(chunk, "content") else chunk
                        
                        if content:
                            response_parts.append(content)
                            await q.put(content)
                        
                        self._partial_response = "".join(response_parts)
                    
                    if not handle.interrupted:
                        await q.put(None)
                    
                    return "".join(response_parts)
                
                except asyncio.CancelledError:
                    logger.info("LLM collection cancelled")
                    await q.put(None)
                    return "".join(response_parts)
            
            async def tts_consumer():
                """Consume LLM chunks and send to TTS"""
                if wait_for_authorization:
                    try:
                        await asyncio.wait_for(
                            self._preemptive_authorized.wait(),
                            timeout=10.0
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
                
                if self.speech_generation:
                    try:
                        await self.speech_generation.synthesize(tts_stream_gen())
                    except asyncio.CancelledError:
                        if self.speech_generation:
                            await self.speech_generation.interrupt()
            
            collector_task = asyncio.create_task(collector())
            tts_task = asyncio.create_task(tts_consumer())
            
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
            
            if full_response and self.agent:
                self.agent.chat_context.add_message(
                    role=ChatRole.ASSISTANT,
                    content=full_response
                )
                
                self.emit("content_generated", {"text": full_response})
        
        finally:
            if not handle.done():
                handle._mark_done()
    
    async def say(self, message: str, handle: UtteranceHandle) -> None:
        """
        Direct TTS synthesis (for initial messages).
        
        Args:
            message: Message to synthesize
            handle: Utterance handle to track
        """
        if self.speech_generation:
            try:
                await self.speech_generation.synthesize(message)
            finally:
                handle._mark_done()
    
    async def reply_with_context(
        self,
        instructions: str,
        wait_for_playback: bool,
        handle: UtteranceHandle,
        frames: list[av.VideoFrame] | None = None
    ) -> None:
        """
        Generate a reply using instructions and current chat context.
        
        Args:
            instructions: Instructions to add to chat context
            wait_for_playback: If True, disable VAD/STT during response
            handle: Utterance handle
            frames: Optional video frames for vision
        """
        if not self.agent:
            handle._mark_done()
            return
        
        original_handlers = {}
        if wait_for_playback and self.speech_understanding:
            if self.speech_understanding.vad:
                original_handlers['vad'] = self.speech_understanding._on_vad_event
                self.speech_understanding._on_vad_event = lambda x: None
            
            if self.speech_understanding.stt:
                original_handlers['stt'] = self.speech_understanding._on_stt_transcript
                self.speech_understanding._on_stt_transcript = lambda x: None
        
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
            
            await self._generate_and_synthesize(final_instructions, handle)
        
        finally:
            if wait_for_playback and self.speech_understanding:
                if 'vad' in original_handlers:
                    self.speech_understanding._on_vad_event = original_handlers['vad']
                if 'stt' in original_handlers:
                    self.speech_understanding._on_stt_transcript = original_handlers['stt']
            
            if not handle.done():
                handle._mark_done()
    
    async def _monitor_interruption_duration(self) -> None:
        """Monitor user speech duration during agent response"""
        if self.interrupt_mode not in ("VAD_ONLY", "HYBRID"):
            return
        
        try:
            await asyncio.sleep(self.interrupt_min_duration)
            
            if self.agent and self.agent.session and self.agent.session.current_utterance:
                if self.agent.session.current_utterance.is_interruptible:
                    logger.info(f"User speech duration exceeded {self.interrupt_min_duration}s threshold, triggering interruption")
                    await self._trigger_interruption()
        
        except asyncio.CancelledError:
            logger.debug("Interruption monitoring cancelled")
    
    async def handle_stt_event(self, text: str) -> None:
        """Handle STT event for interruption (word-based)"""
        if not text or not text.strip():
            return
        
        word_count = len(text.strip().split())
        
        if self.resume_on_false_interrupt and self._is_in_false_interrupt_pause and word_count >= self.interrupt_min_words:
            logger.info(f"STT transcript received while in paused state, confirming real interruption")
            self._cancel_false_interrupt_timer()
            self._is_in_false_interrupt_pause = False
            self._false_interrupt_paused_speech = False
            await self._interrupt_pipeline()
            return
        
        if self.interrupt_mode in ("STT_ONLY", "HYBRID"):
            if word_count >= self.interrupt_min_words:
                if self.agent and self.agent.session and self.agent.session.current_utterance:
                    if self.agent.session.current_utterance.is_interruptible:
                        await self._trigger_interruption()
    
    async def _trigger_interruption(self) -> None:
        """Trigger interruption with optional pause/resume support"""
        if self._is_interrupted:
            return
        
        if self.agent and self.agent.session and self.agent.session.current_utterance:
            if not self.agent.session.current_utterance.is_interruptible:
                logger.info("Interruption disabled for current utterance")
                return
        
        self._is_interrupted = True
        
        can_resume = self.resume_on_false_interrupt and self.speech_generation and self.speech_generation.can_pause()
        
        if can_resume:
            logger.info("Pausing TTS for potential resume")
            self._false_interrupt_paused_speech = True
            self._is_in_false_interrupt_pause = True
            if self.speech_generation:
                await self.speech_generation.pause()
            self._start_false_interrupt_timer()
        else:
            logger.info("Performing full interruption")
            await self._interrupt_pipeline()
    
    def _start_false_interrupt_timer(self):
        """Start timer to detect false interrupts"""
        if self._false_interrupt_timer:
            self._false_interrupt_timer.cancel()
        
        if self.false_interrupt_pause_duration is None:
            return
        
        logger.info(f"Starting false interrupt timer for {self.false_interrupt_pause_duration}s")
        loop = asyncio.get_event_loop()
        self._false_interrupt_timer = loop.call_later(
            self.false_interrupt_pause_duration,
            lambda: asyncio.create_task(self._on_false_interrupt_timeout())
        )
    
    def _cancel_false_interrupt_timer(self):
        """Cancel false interrupt timer"""
        if self._false_interrupt_timer:
            logger.info("Cancelling false interrupt timer")
            self._false_interrupt_timer.cancel()
            self._false_interrupt_timer = None
    
    async def _on_false_interrupt_timeout(self):
        """Handle false interrupt timeout"""
        logger.info(f"False interrupt timeout reached after {self.false_interrupt_pause_duration}s")
        self._false_interrupt_timer = None
        
        if self._is_user_speaking:
            logger.info("User still speaking - confirming real interruption")
            self._is_in_false_interrupt_pause = False
            self._false_interrupt_paused_speech = False
            await self._interrupt_pipeline()
            return
        
        if self._is_in_false_interrupt_pause and self.speech_generation and self.speech_generation.can_pause():
            logger.info("Resuming agent speech - false interruption detected")
            self._is_interrupted = False
            self._is_in_false_interrupt_pause = False
            self._false_interrupt_paused_speech = False
            await self.speech_generation.resume()
    
    async def _interrupt_pipeline(self) -> None:
        """Interrupt all components"""
        self._is_interrupted = True
        self._cancel_false_interrupt_timer()
        self._is_in_false_interrupt_pause = False
        self._false_interrupt_paused_speech = False
        
        if self.agent and self.agent.session and self.agent.session.current_utterance:
            if self.agent.session.current_utterance.is_interruptible:
                self.agent.session.current_utterance.interrupt()
        
        if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
            await self.agent.session.stop_thinking_audio()
        
        if self.speech_generation:
            await self.speech_generation.interrupt()
        
        if self.content_generation:
            await self.content_generation.cancel()
        
        self._partial_response = ""
        self._is_interrupted = False
    
    async def interrupt(self) -> None:
        """Public method to interrupt the pipeline"""
        await self._interrupt_pipeline()
    
    async def _cancel_preemptive_generation(self) -> None:
        """Cancel preemptive generation"""
        logger.info("Cancelling preemptive generation")
        self._preemptive_cancelled = True
        self._preemptive_authorized.set() 
        
        if self._preemptive_generation_task and not self._preemptive_generation_task.done():
            self._preemptive_generation_task.cancel()
            try:
                await self._preemptive_generation_task
            except asyncio.CancelledError:
                logger.info("Preemptive task cancelled successfully")
        
        self._preemptive_generation_task = None
        
        if self.content_generation:
            await self.content_generation.cancel()
        
        if self.speech_generation:
            await self.speech_generation.interrupt()
        
        if self.speech_understanding:
            self.speech_understanding.clear_preemptive_state()
        
        logger.info("Preemptive generation cancelled")
    
    async def _run_vmd_check(self) -> None:
        """Run voicemail detection check"""
        try:
            if not self.voice_mail_detector:
                return
            
            await asyncio.sleep(self.voice_mail_detector.duration)
            
            is_voicemail = await self.voice_mail_detector.detect(self._vmd_buffer.strip())
            self.voice_mail_detection_done = True
            
            if is_voicemail:
                await self._interrupt_pipeline()
            
            self.emit("voicemail_result", {"is_voicemail": is_voicemail})
        
        except Exception as e:
            logger.error(f"Error in VMD check: {e}")
            self.voice_mail_detection_done = True
            self.emit("voicemail_result", {"is_voicemail": False})
        
        finally:
            self._vmd_check_task = None
            self._vmd_buffer = ""
    
    async def cleanup(self) -> None:
        """Cleanup all components"""
        logger.info("Cleaning up pipeline orchestrator")
        
        if self._vmd_check_task and not self._vmd_check_task.done():
            self._vmd_check_task.cancel()
        
        if self._false_interrupt_timer:
            self._false_interrupt_timer.cancel()
        
        if self.speech_understanding:
            await self.speech_understanding.cleanup()
        
        if self.content_generation:
            await self.content_generation.cleanup()
        
        if self.speech_generation:
            await self.speech_generation.cleanup()
        
        self.agent = None
        self.avatar = None
        self.conversational_graph = None
        self.voice_mail_detector = None
        
        logger.info("Pipeline orchestrator cleaned up")
