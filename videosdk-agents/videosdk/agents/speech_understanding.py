from __future__ import annotations

from typing import Callable, Awaitable, Literal, TYPE_CHECKING
import asyncio
import time
import logging
from .event_emitter import EventEmitter
from .stt.stt import STT, STTResponse, SpeechEventType
from .vad import VAD, VADResponse, VADEventType
from .eou import EOU
from .llm.chat_context import ChatRole
from .denoise import Denoise
from .metrics import turn_lifecycle_tracker
if TYPE_CHECKING:
    from .agent import Agent
    from .pipeline_hooks import PipelineHooks

logger = logging.getLogger(__name__)


class SpeechUnderstanding(EventEmitter[Literal["transcript_interim", "transcript_final", "speech_started", "speech_stopped", "eou_detected"]]):
    """
    Handles speech input processing through VAD, STT, and Turn Detection.
    
    Events:
    - transcript_interim: Interim transcription received
    - transcript_final: Final transcription ready for processing
    - speech_started: User started speaking (from VAD)
    - speech_stopped: User stopped speaking (from VAD)
    - eou_detected: End of utterance detected
    """
    
    def __init__(
        self,
        agent: Agent | None = None,
        stt: STT | None = None,
        vad: VAD | None = None,
        turn_detector: EOU | None = None,
        denoise: Denoise | None = None,
        mode: Literal["ADAPTIVE", "DEFAULT"] = "DEFAULT",
        min_speech_wait_timeout: float = 0.5,
        max_speech_wait_timeout: float = 0.8,
        eou_certainty_threshold: float = 0.85,
        hooks: "PipelineHooks | None" = None,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.stt = stt
        self.vad = vad
        self.turn_detector = turn_detector
        self.denoise = denoise
        self.turn_tracker = turn_lifecycle_tracker
        self.hooks = hooks
        
        # EOU configuration
        self.mode = mode
        self.min_speech_wait_timeout = min_speech_wait_timeout
        self.max_speech_wait_timeout = max_speech_wait_timeout
        self.eou_certainty_threshold = eou_certainty_threshold
        
        # State management
        self._accumulated_transcript = ""
        self._waiting_for_more_speech = False
        self._wait_timer: asyncio.TimerHandle | None = None
        self._transcript_processing_lock = asyncio.Lock()
        self._is_user_speaking = False
        self._stt_started = False
        self.stt_lock = asyncio.Lock()
        
        # Preemptive generation state
        self._preemptive_transcript: str | None = None
        self._preemptive_lock = asyncio.Lock()
        self._enable_preemptive_generation = False
        
        # Stream STT state
        self._stt_stream_task: asyncio.Task | None = None
        self._stt_stream_queue: asyncio.Queue | None = None
        self.turn_tracker = turn_lifecycle_tracker

        
        # Setup event handlers
        if self.stt:
            self.stt.on_stt_transcript(self._on_stt_transcript)
        if self.vad:
            self.vad.on_vad_event(self._on_vad_event)
    
    def update_preemptive_generation_flag(self) -> None:
        """Update the preemptive generation flag based on current STT instance"""
        self._enable_preemptive_generation = getattr(self.stt, 'enable_preemptive_generation', False) if self.stt else False
    
    async def start(self) -> None:
        """Start the speech understanding component"""
        self.update_preemptive_generation_flag()
        logger.info("SpeechUnderstanding started")
    
    async def process_audio(self, audio_data: bytes) -> None:
        """
        Process incoming audio data through denoise, STT, and VAD.
        
        Note: speech_in hook is processed at the input stream level before this method.
        
        Args:
            audio_data: Raw audio bytes (already processed through speech_in hook)
        """
        try:
            if self.hooks and self.hooks.has_stt_stream_hook():
                if self._stt_stream_task is None:
                    self._stt_stream_queue = asyncio.Queue()
                    self._stt_stream_task = asyncio.create_task(self._run_stt_stream())
                
                await self._stt_stream_queue.put(audio_data)
                return

            if self.denoise:
                audio_data = await self.denoise.denoise(audio_data)
            
            if self.stt:
                async with self.stt_lock:
                    await self.stt.process_audio(audio_data)
            
            if self.vad:
                await self.vad.process_audio(audio_data)
                
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            self.emit("error", f"Audio processing failed: {str(e)}")

    async def _run_stt_stream(self) -> None:
        """Run the STT stream hook loop"""
        async def audio_generator():
            while True:
                if self._stt_stream_queue:
                    chunk = await self._stt_stream_queue.get()
                    if chunk is None:
                        break
                    yield chunk
                else:
                    break

        try:
            async for event in self.hooks.process_stt_stream(audio_generator()):
                if hasattr(event, "event_type"):
                    await self._on_stt_transcript(event)
        except Exception as e:
            logger.error(f"Error in STT stream task: {e}", exc_info=True)
        finally:
            self._stt_stream_task = None
            self._stt_stream_queue = None
    
    async def _on_vad_event(self, vad_response: VADResponse) -> None:
        """Handle VAD events"""
        if vad_response.event_type == VADEventType.START_OF_SPEECH:
            self._is_user_speaking = True

            if self._waiting_for_more_speech:
                logger.debug("User continued speaking, cancelling wait timer")
                await self._handle_continued_speech()

            # Track STT start time when user starts speaking
            logger.info(f"[VAD DEBUG] START_OF_SPEECH event")
            logger.info(f"[VAD DEBUG] self.turn_tracker = {self.turn_tracker}")
            logger.info(f"[VAD DEBUG] Type: {type(self.turn_tracker).__name__ if self.turn_tracker is not None else 'None'}")
            logger.info(f"[VAD DEBUG] Boolean check (is not None): {self.turn_tracker is not None}")
            logger.info(f"[VAD DEBUG] Truthiness check (if self.turn_tracker): {bool(self.turn_tracker)}")

            if self.turn_tracker:
                logger.info(f"[VAD DEBUG] Calling on_stt_start()")
                self.turn_tracker.on_stt_start(time.perf_counter())
            else:
                logger.warning(f"[VAD DEBUG] turn_tracker is None, cannot track STT start")

            self.emit("speech_started")
            
        elif vad_response.event_type == VADEventType.END_OF_SPEECH:
            self._is_user_speaking = False
            self.emit("speech_stopped")
            
            if not self._stt_started and self.stt:
                self._stt_started = True
    
    async def _on_stt_transcript(self, stt_response: STTResponse) -> None:
        """Handle STT transcript events"""
        if self._waiting_for_more_speech:
            await self._handle_continued_speech()

        text = stt_response.data.text if stt_response.data else ""
        timestamp = time.perf_counter()

        if stt_response.event_type == SpeechEventType.PREFLIGHT:
            # Track preflight transcript timing for preemptive generation
            if self.turn_tracker:
                self.turn_tracker.on_stt_preflight_end(timestamp, text)

            await self._handle_preflight_transcript(text)

        elif stt_response.event_type == SpeechEventType.FINAL:
            # Track final STT completion with rich metadata
            if self.turn_tracker:
                confidence = stt_response.data.confidence if stt_response.data else None
                duration = stt_response.data.duration if stt_response.data else None
                is_preemptive = self._enable_preemptive_generation

                self.turn_tracker.on_stt_complete(
                    timestamp=timestamp,
                    transcript=text,
                    confidence=confidence,
                    duration=duration,
                    metadata=stt_response.metadata,
                    is_preemptive=is_preemptive
                )

            if self._enable_preemptive_generation:
                self.emit("transcript_final", {
                    "text": text,
                    "is_preemptive": True,
                    "metadata": stt_response.metadata
                })
            else:
                await self._process_transcript_with_eou(text)

        elif stt_response.event_type == SpeechEventType.INTERIM:
            # Track interim transcript timing (TTFW)
            if self.turn_tracker:
                self.turn_tracker.on_stt_interim_end(timestamp)

            self.emit("transcript_interim", {
                "text": text,
                "metadata": stt_response.metadata
            })

            if stt_response.metadata and stt_response.metadata.get("turn_resumed"):
                await self._handle_turn_resumed(text)
    
    async def _handle_preflight_transcript(self, preflight_text: str) -> None:
        """Handle preflight transcript for preemptive generation"""
        async with self._preemptive_lock:
            self._preemptive_transcript = preflight_text.strip()
            
            self.emit("transcript_preflight", {
                "text": self._preemptive_transcript
            })
    
    async def _process_transcript_with_eou(self, new_transcript: str) -> None:
        """Process transcript with EOU-based decision making"""
        async with self._transcript_processing_lock:
            if self._accumulated_transcript:
                self._accumulated_transcript += " " + new_transcript
            else:
                self._accumulated_transcript = new_transcript

            delay = self.min_speech_wait_timeout
            eou_probability = None

            if self.mode == 'DEFAULT':
                if self.turn_detector and self.agent:
                    # Track EOU detection timing
                    logger.info(f"[EOU DEBUG] About to call EOU metrics - turn_tracker: {self.turn_tracker is not None}")
                    if self.turn_tracker:
                        logger.info(f"[EOU DEBUG] Calling on_eou_start()")
                        self.turn_tracker.on_eou_start(time.perf_counter())
                    else:
                        logger.warning(f"[EOU DEBUG] turn_tracker is None, cannot track EOU start")

                    eou_probability = self.turn_detector.get_eou_probability(self.agent.chat_context)
                    logger.info(f"EOU probability: {eou_probability}")

                    if self.turn_tracker:
                        logger.info(f"[EOU DEBUG] Calling on_eou_complete() with probability={eou_probability}")
                        self.turn_tracker.on_eou_complete(time.perf_counter(), eou_probability)
                    else:
                        logger.warning(f"[EOU DEBUG] turn_tracker is None, cannot track EOU complete")

                    if eou_probability < self.eou_certainty_threshold:
                        delay = self.max_speech_wait_timeout

            elif self.mode == 'ADAPTIVE':
                if self.turn_detector and self.agent:
                    # Track EOU detection timing
                    logger.info(f"[EOU DEBUG] ADAPTIVE mode - turn_tracker: {self.turn_tracker is not None}")
                    if self.turn_tracker:
                        logger.info(f"[EOU DEBUG] Calling on_eou_start()")
                        self.turn_tracker.on_eou_start(time.perf_counter())
                    else:
                        logger.warning(f"[EOU DEBUG] turn_tracker is None, cannot track EOU start")

                    eou_probability = self.turn_detector.get_eou_probability(self.agent.chat_context)
                    logger.info(f"EOU probability: {eou_probability}")

                    if self.turn_tracker:
                        logger.info(f"[EOU DEBUG] Calling on_eou_complete() with probability={eou_probability}")
                        self.turn_tracker.on_eou_complete(time.perf_counter(), eou_probability)
                    else:
                        logger.warning(f"[EOU DEBUG] turn_tracker is None, cannot track EOU complete")

                    delay_range = self.max_speech_wait_timeout - self.min_speech_wait_timeout
                    wait_factor = 1.0 - eou_probability
                    delay = self.min_speech_wait_timeout + (delay_range * wait_factor)

            # Track wait for additional speech
            if self.turn_tracker and eou_probability is not None:
                self.turn_tracker.on_wait_for_additional_speech(delay, eou_probability)

            logger.info(f"Using delay: {delay} seconds")
            await self._wait_for_additional_speech(delay)
    
    async def _wait_for_additional_speech(self, delay: float) -> None:
        """Wait for additional speech within the timeout period"""
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
            
            await self._finalize_transcript()
    
    async def _finalize_transcript(self) -> None:
        """Finalize the accumulated transcript and emit event"""
        if not self._accumulated_transcript.strip():
            return
        
        final_transcript = self._accumulated_transcript.strip()
        logger.info(f"Finalizing transcript: '{final_transcript}'")
        
        self._accumulated_transcript = ""
        
        self.emit("transcript_final", {
            "text": final_transcript,
            "is_preemptive": False,
            "eou_detected": True
        })
        
        self.emit("eou_detected", {
            "text": final_transcript
        })
    
    async def _handle_continued_speech(self) -> None:
        """Handle when user continues speaking while we're waiting"""
        if self._wait_timer:
            self._wait_timer.cancel()
            self._wait_timer = None
        
        self._waiting_for_more_speech = False
    
    async def _handle_turn_resumed(self, resumed_text: str) -> None:
        """Handle TurnResumed event (user continued speaking)"""
        if self._accumulated_transcript:
            self._accumulated_transcript += " " + resumed_text
        else:
            self._accumulated_transcript = resumed_text
        
        self.emit("turn_resumed", {
            "text": resumed_text
        })
    
    def check_preemptive_match(self, final_text: str) -> bool:
        """
        Check if final transcript matches preflight transcript.
        
        Returns:
            True if match, False if mismatch
        """
        if not self._preemptive_transcript:
            return False
        
        final_normalized = final_text.strip()
        preflight_normalized = self._preemptive_transcript.strip()
        
        return final_normalized == preflight_normalized
    
    def clear_preemptive_state(self) -> None:
        """Clear preemptive generation state"""
        self._preemptive_transcript = None
    
    async def cleanup(self) -> None:
        """Cleanup speech understanding resources"""
        logger.info("Cleaning up speech understanding")
        
        if self._stt_stream_queue:
            await self._stt_stream_queue.put(None)
        
        if self._stt_stream_task:
            try:
                await asyncio.wait_for(self._stt_stream_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            self._stt_stream_task = None
            self._stt_stream_queue = None

        if self._wait_timer:
            self._wait_timer.cancel()
            self._wait_timer = None
        
        self._accumulated_transcript = ""
        self._waiting_for_more_speech = False
        self._preemptive_transcript = None
        
        self.stt = None
        self.vad = None
        self.turn_detector = None
        self.denoise = None
        self.agent = None
        
        logger.info("Speech understanding cleaned up")
