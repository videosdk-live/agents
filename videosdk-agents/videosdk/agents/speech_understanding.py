from __future__ import annotations

from typing import Callable, Awaitable, Literal, TYPE_CHECKING
import asyncio
import logging
import time
from .event_emitter import EventEmitter
from .stt.stt import STT, STTResponse, SpeechEventType
from .vad import VAD, VADResponse, VADEventType
from .eou import EOU
from .llm.chat_context import ChatContext, ChatRole
from .utils import TurnResult, TurnState, is_backchannel_aware
from .denoise import Denoise
from .metrics import metrics_collector
from .utils import UserState, AgentState


if TYPE_CHECKING:
    from .agent import Agent
    from .pipeline_hooks import PipelineHooks

logger = logging.getLogger(__name__)

_STT_FLUSH = object()


class SpeechUnderstanding(EventEmitter[Literal["transcript_interim", "transcript_final", "transcript_speculative", "transcript_speculative_cancel", "speech_started", "speech_stopped", "eou_detected"]]):
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
        eou_certainty_threshold: float = 0.75,
        hooks: "PipelineHooks | None" = None,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.stt = stt
        self.vad = vad
        self.turn_detector = turn_detector
        self.denoise = denoise
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
        self._wait_started_at: float | None = None
        self._transcript_processing_lock = asyncio.Lock()
        self._is_user_speaking = False
        self._stt_started = False
        self.stt_lock = asyncio.Lock()
        self.turn_detector_lock = asyncio.Lock()
        self.denoise_lock = asyncio.Lock()
        
        # Preemptive generation state
        self._preemptive_transcript: str | None = None
        self._preemptive_lock = asyncio.Lock()
        self._enable_preemptive_generation = False
        
        # Stream STT state
        self._stt_stream_task: asyncio.Task | None = None
        self._stt_stream_queue: asyncio.Queue | None = None

        self._vad_queue: asyncio.Queue[bytes | None] | None = None
        self._stt_queue: asyncio.Queue | None = None
        self._vad_consumer_task: asyncio.Task | None = None
        self._stt_consumer_task: asyncio.Task | None = None
        self._vad_queue_max = 50
        self._stt_queue_max = 100
        self._vad_dropped = 0
        self._stt_dropped = 0
        self._consumers_started = False

        # VAD speech context — carries metadata from VAD events for downstream use
        self._last_speech_audio: bytes | None = None
        self._last_speech_confidence: float = 0.0
        self._last_speech_energy: float = 0.0

        # Real-time VAD frame state — updated ~31 times/sec by FRAME_PROCESSED
        self._current_vad_probability: float = 0.0
        self._current_vad_energy: float = 0.0
        self._current_vad_speaking: bool = False
        
        # Turn result from the most recent EOU call this turn.
        self._last_turn_result: TurnResult | None = None

        # Setup event handlers
        if self.stt:
            self.stt.on_stt_transcript(self._on_stt_transcript)
        if self.vad:
            self.vad.on_vad_event(self._on_vad_event)
            # Register for per-frame updates if the VAD supports it.
            if hasattr(self.vad, "on_inference"):
                self.vad.on_inference(self._on_vad_frame)

    def update_preemptive_generation_flag(self) -> None:
        """Update the preemptive generation flag based on current STT instance"""
        self._enable_preemptive_generation = getattr(self.stt, 'enable_preemptive_generation', False) if self.stt else False
        metrics_collector.set_preemptive_generation_enabled(self._enable_preemptive_generation)
    
    async def start(self) -> None:
        """Start the speech understanding component"""
        self.update_preemptive_generation_flag()
        logger.info("SpeechUnderstanding started")
    
    async def process_audio(self, audio_data: bytes) -> None:
        """Denoise the chunk inline, then hand it to the VAD and STT consumers.

        Returns without waiting on inference: the chunk is enqueued onto bounded
        per-consumer queues that background tasks drain at their own pace, so VAD/STT
        processing never blocks the room recv loop. The STT-stream-hook path bypasses
        the queues and feeds its own generator instead.

        Note: the speech_in hook is processed at the input stream level before this method.

        Args:
            audio_data: Raw audio bytes (already processed through the speech_in hook).
        """
        try:
            if self.hooks and self.hooks.has_stt_stream_hook():
                if self._stt_stream_task is None:
                    self._stt_stream_queue = asyncio.Queue()
                    self._stt_stream_task = asyncio.create_task(self._run_stt_stream())

                await self._stt_stream_queue.put(audio_data)
                return

            if self.denoise:
                async with self.denoise_lock:
                    audio_data = await self.denoise.denoise(audio_data)

            if not self._consumers_started:
                self._start_consumers()

            if self.vad and self._vad_queue is not None:
                self._enqueue_with_drop_oldest(
                    self._vad_queue, audio_data, "_vad_dropped", "VAD"
                )

            if self.stt and self._stt_queue is not None:
                self._enqueue_with_drop_oldest(
                    self._stt_queue, audio_data, "_stt_dropped", "STT"
                )

        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            self.emit("error", f"Audio processing failed: {str(e)}")

    def _enqueue_with_drop_oldest(
        self, queue: asyncio.Queue, chunk: bytes, counter_attr: str, label: str
    ) -> None:
        """Non-blocking enqueue. On overflow, drop the oldest chunk to preserve recent audio."""
        try:
            queue.put_nowait(chunk)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(chunk)
            except asyncio.QueueFull:
                return
            dropped = getattr(self, counter_attr) + 1
            setattr(self, counter_attr, dropped)
            if dropped % 25 == 1:
                logger.warning(
                    f"{label} queue overflow: dropped {dropped} chunks so far"
                )

    def _enqueue_stt_flush(self) -> None:
        """Queue a flush request behind the audio already buffered for STT."""
        queue = self._stt_queue
        if queue is None:
            return
        try:
            queue.put_nowait(_STT_FLUSH)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(_STT_FLUSH)
            except asyncio.QueueFull:
                pass

    def _start_consumers(self) -> None:
        """Lazily start VAD/STT consumer tasks on first audio chunk."""
        if self.vad and self._vad_queue is None:
            self._vad_queue = asyncio.Queue(maxsize=self._vad_queue_max)
            self._vad_consumer_task = asyncio.create_task(self._run_vad_consumer())
        if self.stt and self._stt_queue is None:
            self._stt_queue = asyncio.Queue(maxsize=self._stt_queue_max)
            self._stt_consumer_task = asyncio.create_task(self._run_stt_consumer())
        self._consumers_started = True

    async def _run_vad_consumer(self) -> None:
        """Drain the VAD queue, forwarding chunks to the configured VAD instance."""
        queue = self._vad_queue
        if queue is None:
            return
        while True:
            chunk = await queue.get()
            if chunk is None:
                return
            vad = self.vad
            if vad is None:
                continue
            try:
                await vad.process_audio(chunk)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"VAD consumer error: {e}")

    async def _run_stt_consumer(self) -> None:
        """Drain the STT queue, forwarding chunks (and flush requests) to the STT instance."""
        queue = self._stt_queue
        if queue is None:
            return
        while True:
            item = await queue.get()
            if item is None:
                return
            try:
                async with self.stt_lock:
                    stt = self.stt
                    if stt is None:
                        continue
                    if item is _STT_FLUSH:
                        await stt.flush()
                    else:
                        await stt.process_audio(item)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"STT consumer error: {e}")

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
        if self.agent and getattr(self.agent, "session", None) is not None:       
            if not getattr(self.agent.session, "_accept_user_input", True):
                return
       
        """Handle VAD events, forwarding rich metadata to downstream consumers."""
        vad_data = vad_response.data
        logger.info(
            f"[speech_understanding] _on_vad_event: {vad_response.event_type.value} "
            f"| confidence={vad_data.confidence:.4f} "
            f"| energy={vad_data.energy:.4f} "
            f"| raw_prob={vad_data.raw_probability:.4f} "
            f"| _is_user_speaking={self._is_user_speaking}"
        )

        if vad_response.event_type == VADEventType.START_OF_SPEECH:
            logger.info("VAD: EVENT START_OF_SPEECH RECEIVED .............................")
            self._is_user_speaking = True
            self._last_speech_confidence = vad_data.confidence
            self._last_speech_energy = vad_data.energy
            self.agent.session._emit_user_state(UserState.SPEAKING)
            if not (self.agent.session.agent_state == AgentState.SPEAKING or self.agent.session.agent_state == AgentState.THINKING):
                self.agent.session._emit_agent_state(AgentState.LISTENING)
            metrics_collector.on_user_speech_start()
            if self._waiting_for_more_speech:
                logger.debug("User continued speaking, cancelling wait timer")
                await self._handle_continued_speech()

            self.emit("speech_started", {
                "confidence": vad_data.confidence,
                "energy": vad_data.energy,
                "speech_duration": vad_data.speech_duration,
                "timestamp": vad_data.timestamp,
            })

        elif vad_response.event_type == VADEventType.END_OF_SPEECH:
            logger.info("VAD: EVENT END_OF_SPEECH RECEIVED .............................")
            self._is_user_speaking = False
            self._last_speech_audio = vad_data.audio_frames
            self._last_speech_confidence = vad_data.confidence
            self._last_speech_energy = vad_data.energy

            try:
                if self.stt:
                    if self._stt_queue is not None:
                        self._enqueue_stt_flush()
                    else:
                        await self.stt.flush()
            except Exception as e:
                logger.error(f"Error flushing STT: {e}")
            metrics_collector.on_user_speech_end()
            metrics_collector.on_stt_start()

            self.emit("speech_stopped", {
                "confidence": vad_data.confidence,
                "energy": vad_data.energy,
                "speech_duration": vad_data.speech_duration,
                "silence_duration": vad_data.silence_duration,
                "timestamp": vad_data.timestamp,
                "has_audio": vad_data.audio_frames is not None,
            })

            if not self._stt_started and self.stt:
                self._stt_started = True
                
    def _on_vad_frame(self, vad_response: VADResponse) -> None:
        """Handle per-frame FRAME_PROCESSED events (~31/sec).

        This is intentionally synchronous (not async) to avoid creating
        ~31 asyncio tasks per second.  It only sets three scalar values.
        """
        vad_data = vad_response.data
        self._current_vad_probability = vad_data.confidence
        self._current_vad_energy = vad_data.energy
        self._current_vad_speaking = vad_data.is_speech

    @property
    def current_vad_probability(self) -> float:
        """Real-time speech probability from the most recent VAD frame."""
        return self._current_vad_probability

    @property
    def current_vad_energy(self) -> float:
        """Real-time audio energy from the most recent VAD frame."""
        return self._current_vad_energy

    @property
    def current_vad_speaking(self) -> bool:
        """Whether the VAD currently considers the user to be speaking."""
        return self._current_vad_speaking

    async def _on_stt_transcript(self, stt_response: STTResponse) -> None:
        """Handle STT transcript events"""
        if self.agent and getattr(self.agent, "session", None) is not None:
            if not getattr(self.agent.session, "_accept_user_input", True):
                return
        if self._waiting_for_more_speech:
            await self._handle_continued_speech()

        text = stt_response.data.text if stt_response.data else ""


        if not self.vad:
            if not self._is_user_speaking and stt_response.event_type in (SpeechEventType.INTERIM, SpeechEventType.FINAL):
                self._is_user_speaking = True
                metrics_collector.on_user_speech_start()
                self.emit("speech_started")

            if self._is_user_speaking and stt_response.event_type == SpeechEventType.FINAL:
                self._is_user_speaking = False
                metrics_collector.on_user_speech_end()
                self.emit("speech_stopped")

        if stt_response.event_type == SpeechEventType.PREFLIGHT:
            metrics_collector.on_stt_preflight_end()
            await self._handle_preflight_transcript(text)
            
        elif stt_response.event_type == SpeechEventType.FINAL:
            duration = stt_response.data.duration
            confidence = stt_response.data.confidence
            metrics_collector.on_stt_complete(text, duration, confidence)
            if self._enable_preemptive_generation:
                self.emit("transcript_final", {
                    "text": text,
                    "is_preemptive": True,
                    "confidence": confidence,
                    "metadata": stt_response.metadata,
                    "turn_state": None,
                })
            else:
                await self._process_transcript_with_eou(text)

        elif stt_response.event_type == SpeechEventType.INTERIM:
            # If on_stt_start() was never called (no VAD END_OF_SPEECH fired),
            # call it on the first INTERIM so STT latency measures from first
            # transcript activity to FINAL result.
            self.agent.session._emit_user_state(UserState.SPEAKING)
            if not (self.agent.session.agent_state == AgentState.SPEAKING or self.agent.session.agent_state == AgentState.THINKING):
                self.agent.session._emit_agent_state(AgentState.LISTENING)
            if metrics_collector._stt_start_time is None:
                metrics_collector.on_stt_start()
            metrics_collector.on_stt_interim_end()
            if getattr(self.stt, "forward_interim_transcripts", False):
                metrics_collector.emit_user_transcript_transport(text, type="interim")
            self.emit("transcript_interim", {
                "text": text,
                "confidence": stt_response.data.confidence,
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

            self.emit("transcript_speculative", {"text": self._accumulated_transcript})

            delay = self.min_speech_wait_timeout
            self._last_turn_result = None

            if self.turn_detector and self.agent:
                eou_context = ChatContext(items=list(self.agent.chat_context.items))
                eou_context.add_message(
                    role=ChatRole.USER, content=self._accumulated_transcript
                )

                metrics_collector.on_eou_start()
                result = await asyncio.to_thread(
                    self.turn_detector.get_turn_result, eou_context
                )
                metrics_collector.on_eou_complete()
                self._last_turn_result = result
                eou_probability = result.eou_probability
                logger.info(f"EOU probability: {eou_probability} (state={result.state})")

                if (
                    self.hooks
                    and self.hooks.has_turn_state_hooks()
                    and is_backchannel_aware(self.turn_detector)
                ):
                    asyncio.create_task(self.hooks.trigger_turn_state({
                        "text": self._accumulated_transcript,
                        "state": result.state.value if result.state else None,
                        "eou_probability": eou_probability,
                    }))

                if result.state in (TurnState.WAIT, TurnState.BACKCHANNEL):
                    delay = 0.0
                    metrics_collector.on_wait_for_additional_speech(delay, eou_probability)
                elif self.mode == 'DEFAULT':
                    if eou_probability < self.eou_certainty_threshold:
                        delay = self.max_speech_wait_timeout
                    metrics_collector.on_wait_for_additional_speech(delay, eou_probability)
                elif self.mode == 'ADAPTIVE':
                    delay_range = self.max_speech_wait_timeout - self.min_speech_wait_timeout
                    wait_factor = 1.0 - eou_probability
                    delay = self.min_speech_wait_timeout + (delay_range * wait_factor)
                    metrics_collector.on_wait_for_additional_speech(delay, eou_probability)

            logger.info(f"Using delay: {delay} seconds")
            await self._wait_for_additional_speech(delay)
    
    async def _wait_for_additional_speech(self, delay: float) -> None:
        """Wait for additional speech within the timeout period"""
        if self._waiting_for_more_speech:
            if self._wait_timer:
                self._wait_timer.cancel()
            self._record_wait_elapsed()

        self._waiting_for_more_speech = True
        self._wait_started_at = time.perf_counter()

        loop = asyncio.get_event_loop()
        self._wait_timer = loop.call_later(
            delay,
            lambda: asyncio.create_task(self._on_speech_timeout())
        )

    def _record_wait_elapsed(self) -> None:
        """Report the actual elapsed wait time to metrics, then clear the start marker."""
        if self._wait_started_at is None:
            return
        elapsed = time.perf_counter() - self._wait_started_at
        self._wait_started_at = None
        try:
            metrics_collector.on_wait_for_additional_speech_complete(elapsed)
        except Exception as e:
            logger.error(f"Failed to record actual wait duration: {e}")
    
    async def _on_speech_timeout(self) -> None:
        """Handle timeout when no additional speech is detected"""
        async with self._transcript_processing_lock:
            if not self._waiting_for_more_speech:
                return

            self._waiting_for_more_speech = False
            self._wait_timer = None
            self._record_wait_elapsed()

            await self._finalize_transcript()
    
    async def _finalize_transcript(self) -> None:
        """Finalize the accumulated transcript and emit event"""
        if not self._accumulated_transcript.strip():
            return
        
        final_transcript = self._accumulated_transcript.strip()
        logger.info(f"Finalizing transcript: '{final_transcript}'")

        result = self._last_turn_result
        turn_state = result.state.value if result and result.state else None

        self._accumulated_transcript = ""
        self._last_turn_result = None

        self.emit("transcript_final", {
            "text": final_transcript,
            "is_preemptive": False,
            "eou_detected": True,
            "turn_state": turn_state,
        })
        
        self.emit("eou_detected", {
            "text": final_transcript
        })
    
    async def _handle_continued_speech(self) -> None:
        """Handle when user continues speaking while we're waiting"""
        if self._wait_timer:
            self._wait_timer.cancel()
            self._wait_timer = None

        self._record_wait_elapsed()
        self._waiting_for_more_speech = False
        self._last_turn_result = None
        self.emit("transcript_speculative_cancel", {})

    async def _handle_turn_resumed(self, resumed_text: str) -> None:
        """Handle TurnResumed event (user continued speaking)"""
        if self._accumulated_transcript:
            self._accumulated_transcript += " " + resumed_text
        else:
            self._accumulated_transcript = resumed_text
        
        self.emit("turn_resumed", {
            "text": resumed_text
        })

    def get_last_speech_audio(self) -> bytes | None:
        """Return the raw PCM int16 audio from the most recent speech segment.

        Available after an END_OF_SPEECH event when the VAD provides
        audio frames.  Returns ``None`` if no audio has been captured yet
        or if the VAD does not support audio frame capture.
        """
        return self._last_speech_audio

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

        for queue in (self._vad_queue, self._stt_queue):
            if queue is None:
                continue
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

        for task_attr in ("_vad_consumer_task", "_stt_consumer_task"):
            task = getattr(self, task_attr)
            if task and not task.done():
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    task.cancel()
            setattr(self, task_attr, None)

        self._vad_queue = None
        self._stt_queue = None
        self._consumers_started = False

        if self._wait_timer:
            self._wait_timer.cancel()
            self._wait_timer = None

        self._accumulated_transcript = ""
        self._last_turn_result = None
        self._waiting_for_more_speech = False
        self._wait_started_at = None
        self._preemptive_transcript = None
        self._last_speech_audio = None

        if self.vad:
            try:
                await self.vad.flush()
            except Exception as e:
                logger.error(f"Error flushing VAD: {e}")

        self.stt = None
        self.vad = None
        self.turn_detector = None
        self.denoise = None
        self.agent = None
        
        logger.info("Speech understanding cleaned up")
