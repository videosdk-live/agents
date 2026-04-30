from __future__ import annotations

from typing import AsyncIterator, Literal, TYPE_CHECKING, Any
import asyncio
import logging
from .event_emitter import EventEmitter
from .tts.tts import TTS
from .utils import UserState, AgentState
from .metrics import metrics_collector

if TYPE_CHECKING:
    from .agent import Agent
    from .room.output_stream import CustomAudioStreamTrack
    from .pipeline_hooks import PipelineHooks

logger = logging.getLogger(__name__)


class SpeechGeneration(EventEmitter[Literal["synthesis_started", "first_audio_byte", "last_audio_byte", "synthesis_interrupted"]]):
    """
    Handles TTS synthesis and audio playback.
    
    Events:
    - synthesis_started: TTS synthesis begins
    - first_audio_byte: First audio byte ready
    - last_audio_byte: Synthesis complete
    - synthesis_interrupted: Synthesis was interrupted
    """
    
    def __init__(
        self,
        agent: Agent | None = None,
        tts: TTS | None = None,
        avatar: Any | None = None,
        audio_track: CustomAudioStreamTrack | None = None,
        hooks: "PipelineHooks | None" = None,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.tts = tts
        self.avatar = avatar
        self.audio_track = audio_track
        self.hooks = hooks
        self.tts_lock = asyncio.Lock()
        self._is_interrupted = False
        self.full_transcript = ""

        self.spoken_transcript: str = ""
        self._tier2_spoken_transcript: str = ""

        if self.tts and getattr(self.tts, "supports_word_timestamps", False):
            try:
                self.tts.on("word_spoken", self._on_tts_word_spoken)
            except Exception as e:
                logger.debug(f"Failed to subscribe to TTS word_spoken: {e}")
            self.on("last_audio_byte", self._on_final_agent_transcript)

    def _on_tts_word_spoken(self, data: Any) -> None:
        """Handler for TTS ``word_spoken`` events — emits an interim transcript."""
        if not isinstance(data, dict):
            return
        cumulative = data.get("cumulative_text", "")
        if cumulative:
            self._tier2_spoken_transcript = cumulative
            if metrics_collector:
                metrics_collector.emit_agent_transcript_transport(
                    cumulative, type="interim"
                )

    def _on_final_agent_transcript(self, data: Any) -> None:
        """Emit the final agent transcript after playback completes.
        """
        if self.full_transcript and metrics_collector:
            metrics_collector.emit_agent_transcript_transport(
                self.full_transcript, type="final"
            )
    
    async def start(self) -> None:
        """Start the speech generation component"""
        logger.info("SpeechGeneration started")
    
    def set_audio_track(self, audio_track: CustomAudioStreamTrack) -> None:
        """Set the audio track for TTS output"""
        self.audio_track = audio_track
        if self.tts:
            self.tts.audio_track = audio_track
    
    async def synthesize(self, response_gen: AsyncIterator[str] | str) -> None:
        """
        Stream text to TTS and play audio.
        
        Args:
            response_gen: Text generator or string to synthesize
        """
        async with self.tts_lock:
            # Prepare iterator and wrapper shared logic
            response_iterator: AsyncIterator[str]
            if isinstance(response_gen, str):
                async def string_to_iterator(text: str):
                    yield text
                response_iterator = string_to_iterator(response_gen)
            else:
                response_iterator = response_gen

            self.full_transcript = ""
            self.spoken_transcript = ""
            self._tier2_spoken_transcript = ""
            tts_start_recorded = False
            async def character_counting_wrapper(text_iterator: AsyncIterator[str]):
                
                async for text_chunk in text_iterator:
                    nonlocal tts_start_recorded
                    logger.debug(f"[TTS DEBUG] Got text chunk: {len(text_chunk) if text_chunk else 0} chars")
                    if text_chunk and metrics_collector:
                        if not tts_start_recorded:
                            metrics_collector.on_tts_start()
                            tts_start_recorded = True
                        logger.debug(f"[TTS DEBUG] Calling add_tts_characters({len(text_chunk)})")
                        metrics_collector.add_tts_characters(len(text_chunk))
                    if text_chunk:
                        self.full_transcript += text_chunk
                    yield text_chunk
            
            # Wrap the iterator
            response_iterator = character_counting_wrapper(response_iterator)

            if self.hooks and self.hooks.has_tts_stream_hook():
                if self.agent and self.agent.session:
                    self.agent.session._pause_wake_up_timer()
                
                if not self.audio_track:
                    if self.agent and self.agent.session and hasattr(self.agent.session, "pipeline"):
                        if hasattr(self.agent.session.pipeline, "audio_track"):
                            self.audio_track = self.agent.session.pipeline.audio_track

                if self.audio_track and hasattr(self.audio_track, "mark_synthesis_start"):
                    self.audio_track.mark_synthesis_start()

                if self.audio_track and hasattr(self.audio_track, "enable_audio_input"):
                    self.audio_track.enable_audio_input(manual_control=True)

                self.emit("synthesis_started", {})

                if self.hooks and self.hooks.has_agent_turn_start_hooks():
                    await self.hooks.trigger_agent_turn_start()

                try:
                    first_byte_emitted = False

                    async for audio_chunk in self.hooks.process_tts_stream(response_iterator):
                        if not first_byte_emitted:
                            if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
                                await self.agent.session.stop_thinking_audio()

                            metrics_collector.on_tts_first_byte()
                            metrics_collector.on_agent_speech_start()
                            self.emit("first_audio_byte", {})

                            if self.agent and self.agent.session:
                                self.agent.session._emit_agent_state(AgentState.SPEAKING)
                                self.agent.session._emit_user_state(UserState.LISTENING)
                            first_byte_emitted = True

                        if self.audio_track:
                            await self.audio_track.add_new_bytes(audio_chunk)

                    if self.full_transcript and metrics_collector.current_turn:
                        emit = not getattr(self.tts, "supports_word_timestamps", False)
                        metrics_collector.set_agent_response(self.full_transcript, emit_transport=emit)
                    metrics_collector.on_agent_speech_end()
                    metrics_collector.complete_turn()

                    if self.hooks and self.hooks.has_agent_turn_end_hooks():
                        await self.hooks.trigger_agent_turn_end()

                    if self.avatar and hasattr(self.avatar, 'send_segment_end'):
                        await self.avatar.send_segment_end()

                    logger.info("TTS stream synthesis complete")
                    self.emit("last_audio_byte", {})
                    
                except asyncio.CancelledError:
                    logger.info("Synthesis cancelled")
                    self.emit("synthesis_interrupted", {})
                    raise
                except Exception as e:
                    logger.error(f"Error during synthesis: {e}")
                    self.emit("synthesis_error", {"error": str(e)})
                    raise
                finally:
                    if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
                        await self.agent.session.stop_thinking_audio()
                    
                    if self.agent and self.agent.session:
                        self.agent.session._reply_in_progress = False
                        self.agent.session._reset_wake_up_timer()
                
                return

            if not self.tts:
                logger.warning("No TTS available for synthesis")
                return
            
            if self.agent and self.agent.session:
                self.agent.session._pause_wake_up_timer()
            
            if not self.audio_track:
                if self.agent and self.agent.session and hasattr(self.agent.session, "pipeline"):
                    if hasattr(self.agent.session.pipeline, "audio_track"):
                        self.audio_track = self.agent.session.pipeline.audio_track
                    else:
                        logger.warning("Audio track not found in pipeline - last audio callback will be skipped")

            if self.audio_track and hasattr(self.audio_track, "mark_synthesis_start"):
                self.audio_track.mark_synthesis_start()

            if self.audio_track and hasattr(self.audio_track, "enable_audio_input"):
                self.audio_track.enable_audio_input(manual_control=True)

            first_byte_event = asyncio.Event()

            async def on_first_audio_byte():
                """Called when first audio byte is ready"""
                first_byte_event.set()
                if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
                    await self.agent.session.stop_thinking_audio()

                metrics_collector.on_tts_first_byte()
                metrics_collector.on_agent_speech_start()
                self.emit("first_audio_byte", {})

                if self.agent and self.agent.session:
                    self.agent.session._emit_agent_state(AgentState.SPEAKING)
                    self.agent.session._emit_user_state(UserState.LISTENING)

            async def on_last_audio_byte():
                """Called when synthesis is complete"""
                metrics_collector.on_agent_speech_end()
                metrics_collector.complete_turn()

                if self.agent and self.agent.session:
                    self.agent.session._emit_agent_state(AgentState.IDLE)
                    self.agent.session._emit_user_state(UserState.IDLE)
                    self.agent.session._reply_in_progress = False
                    self.agent.session._reset_wake_up_timer()

                if self.hooks and self.hooks.has_agent_turn_end_hooks():
                    await self.hooks.trigger_agent_turn_end()

                logger.info("TTS synthesis complete - Agent and User set to IDLE")
                self.emit("last_audio_byte", {})
            
            self.tts.on_first_audio_byte(on_first_audio_byte)
            
            if self.audio_track:
                if hasattr(self.audio_track, "on_last_audio_byte"):
                    self.audio_track.on_last_audio_byte(on_last_audio_byte)
                else:
                    logger.warning(f"Audio track '{type(self.audio_track).__name__}' does not have 'on_last_audio_byte' method")
            else:
                logger.warning("Audio track not initialized - skipping last audio callback registration")
            
            self.tts.reset_first_audio_tracking()

            self.emit("synthesis_started", {})
            metrics_collector.on_tts_start()

            # Trigger agent_turn_start hook
            if self.hooks and self.hooks.has_agent_turn_start_hooks():
                await self.hooks.trigger_agent_turn_start()
            
            try:
                await self.tts.synthesize(response_iterator)

                # If text was generated but the TTS plugin returned before sending any audio
                # (e.g. non-blocking streaming plugins), wait for the first audio byte
                # to prevent mark_synthesis_complete() from firing immediately on an empty buffer.
                if self.full_transcript and not first_byte_event.is_set():
                    try:
                        await asyncio.wait_for(first_byte_event.wait(), timeout=10.0)
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for first audio byte before marking synthesis complete")

                # Signal that TTS has finished sending all audio data.
                # The audio track will fire on_last_audio_byte only after
                # this flag is set AND the buffer is fully drained.
                if self.audio_track and hasattr(self.audio_track, "mark_synthesis_complete"):
                    self.audio_track.mark_synthesis_complete()

                if self.avatar and hasattr(self.avatar, 'send_segment_end'):
                    await self.avatar.send_segment_end()

            except asyncio.CancelledError:
                logger.info("Synthesis cancelled")
                self.emit("synthesis_interrupted", {})
                raise
            
            except Exception as e:
                logger.error(f"Error during synthesis: {e}")
                self.emit("synthesis_error", {"error": str(e)})
                raise
            
            finally:
                if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
                    await self.agent.session.stop_thinking_audio()

                if self.agent and self.agent.session and self._is_interrupted:
                    self.agent.session._reply_in_progress = False
                    self.agent.session._reset_wake_up_timer()
                elif self.agent and self.agent.session:
                    self.agent.session._reply_in_progress = False

    def compute_spoken_transcript(self) -> str:
        """Best-effort estimate of the portion of full_transcript that was
        actually played out to the listener at the moment of call.
        """
        if not self.full_transcript:
            return ""
        if self._tier2_spoken_transcript:
            return self._tier2_spoken_transcript.strip()
        if not self.audio_track or not hasattr(self.audio_track, "snapshot_playback"):
            return ""
        played, pushed = self.audio_track.snapshot_playback()
        if pushed <= 0:
            return ""
        fraction = max(0.0, min(1.0, played / pushed))
        char_cutoff = int(len(self.full_transcript) * fraction)
        if char_cutoff <= 0:
            return ""
        truncated = self.full_transcript[:char_cutoff]

        last_break = max(truncated.rfind(" "), truncated.rfind("\n"))
        if last_break <= 0:
            return ""
        return truncated[:last_break].strip()

    async def interrupt(self) -> None:
        """Interrupt the current synthesis"""
        self._is_interrupted = True
        self.spoken_transcript = self.compute_spoken_transcript()

        if self.tts:
            await self.tts.interrupt()

        if self.audio_track and hasattr(self.audio_track, 'interrupt'):
            self.audio_track.interrupt()

        if self.avatar and hasattr(self.avatar, 'interrupt'):
            await self.avatar.interrupt()

        self.emit("synthesis_interrupted", {})
    
    async def pause(self) -> None:
        """Pause the current synthesis (if supported)"""
        if self.tts and hasattr(self.tts, 'pause') and self.tts.can_pause:
            await self.tts.pause()
            self.emit("synthesis_paused", {})
    
    async def resume(self) -> None:
        """Resume paused synthesis (if supported)"""
        if self.tts and hasattr(self.tts, 'resume') and self.tts.can_pause:
            await self.tts.resume()
            self.emit("synthesis_resumed", {})
    
    def can_pause(self) -> bool:
        """Check if TTS supports pause/resume"""
        return self.tts and hasattr(self.tts, 'can_pause') and self.tts.can_pause
    
    def reset_interrupt(self) -> None:
        """Reset interrupt flag"""
        self._is_interrupted = False
    
    async def cleanup(self) -> None:
        """Cleanup speech generation resources"""
        logger.info("Cleaning up speech generation")
        
        self.tts = None
        self.agent = None
        self.avatar = None
        self.audio_track = None
        
        logger.info("Speech generation cleaned up")

    @property
    def is_speaking(self) -> bool:
        """Returns True if the agent is currently playing audio"""
        if self.audio_track and hasattr(self.audio_track, "is_speaking"):
            return self.audio_track.is_speaking
        return False
