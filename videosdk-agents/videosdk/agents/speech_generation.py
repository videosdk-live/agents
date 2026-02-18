from __future__ import annotations

from typing import AsyncIterator, Literal, TYPE_CHECKING, Any
import asyncio
import logging
from .event_emitter import EventEmitter
from .tts.tts import TTS
from .utils import UserState, AgentState

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
            if self.hooks and self.hooks.has_tts_stream_hook():
                if self.agent and self.agent.session:
                    self.agent.session._pause_wake_up_timer()
                
                if not self.audio_track:
                    if self.agent and self.agent.session and hasattr(self.agent.session, "pipeline"):
                        if hasattr(self.agent.session.pipeline, "audio_track"):
                            self.audio_track = self.agent.session.pipeline.audio_track
                
                if self.audio_track and hasattr(self.audio_track, "enable_audio_input"):
                    self.audio_track.enable_audio_input(manual_control=True)
                
                self.emit("synthesis_started", {})
                
                if self.hooks and self.hooks.has_agent_turn_start_hooks():
                    await self.hooks.trigger_agent_turn_start()
                
                try:
                    response_iterator: AsyncIterator[str]
                    if isinstance(response_gen, str):
                        async def string_to_iterator(text: str):
                            yield text
                        response_iterator = string_to_iterator(response_gen)
                    else:
                        response_iterator = response_gen
                    
                    first_byte_emitted = False
                    
                    async for audio_chunk in self.hooks.process_tts_stream(response_iterator):
                        if not first_byte_emitted:
                            if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
                                await self.agent.session.stop_thinking_audio()
                            
                            self.emit("first_audio_byte", {})
                            
                            if self.agent and self.agent.session:
                                self.agent.session._emit_agent_state(AgentState.SPEAKING)
                                self.agent.session._emit_user_state(UserState.LISTENING)
                            first_byte_emitted = True
                        
                        if self.audio_track:
                            await self.audio_track.add_new_bytes(audio_chunk)
                    
                    if self.agent and self.agent.session:
                        self.agent.session._emit_agent_state(AgentState.IDLE)
                        self.agent.session._emit_user_state(UserState.IDLE)
                    
                    if self.hooks and self.hooks.has_agent_turn_end_hooks():
                        await self.hooks.trigger_agent_turn_end()
                    
                    logger.info("[synthesize] TTS stream synthesis complete")
                    self.emit("last_audio_byte", {})
                    
                except asyncio.CancelledError:
                    logger.info("[synthesize] Synthesis cancelled")
                    self.emit("synthesis_interrupted", {})
                    raise
                except Exception as e:
                    logger.error(f"[synthesize] Error during synthesis: {e}")
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
                logger.error("[synthesize] No TTS available for synthesis")
                return
            
            if self.agent and self.agent.session:
                self.agent.session._pause_wake_up_timer()
            
            if not self.audio_track:
                if self.agent and self.agent.session and hasattr(self.agent.session, "pipeline"):
                    if hasattr(self.agent.session.pipeline, "audio_track"):
                        self.audio_track = self.agent.session.pipeline.audio_track
                    else:
                        logger.error("[synthesize] Audio track not found in pipeline - last audio callback will be skipped")
            
            if self.audio_track and hasattr(self.audio_track, "enable_audio_input"):
                self.audio_track.enable_audio_input(manual_control=True)
            
            async def on_first_audio_byte():
                """Called when first audio byte is ready"""
                if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
                    await self.agent.session.stop_thinking_audio()
                
                self.emit("first_audio_byte", {})
                
                if self.agent and self.agent.session:
                    self.agent.session._emit_agent_state(AgentState.SPEAKING)
                    self.agent.session._emit_user_state(UserState.LISTENING)
            
            async def on_last_audio_byte():
                """Called when synthesis is complete"""
                if self.agent and self.agent.session:
                    self.agent.session._emit_agent_state(AgentState.IDLE)
                    self.agent.session._emit_user_state(UserState.IDLE)
                
                # Trigger agent_turn_end hook
                if self.hooks and self.hooks.has_agent_turn_end_hooks():
                    await self.hooks.trigger_agent_turn_end()
                
                logger.info("[synthesize] TTS synthesis complete - Agent and User set to IDLE")
                self.emit("last_audio_byte", {})
            
            self.tts.on_first_audio_byte(on_first_audio_byte)
            
            if self.audio_track:
                if hasattr(self.audio_track, "on_last_audio_byte"):
                    self.audio_track.on_last_audio_byte(on_last_audio_byte)
                else:
                    logger.warning(f"[synthesize] Audio track '{type(self.audio_track).__name__}' does not have 'on_last_audio_byte' method")
            else:
                logger.warning("[synthesize] Audio track not initialized - skipping last audio callback registration")
            
            self.tts.reset_first_audio_tracking()
            
            self.emit("synthesis_started", {})
            
            # Trigger agent_turn_start hook
            if self.hooks and self.hooks.has_agent_turn_start_hooks():
                await self.hooks.trigger_agent_turn_start()
            
            try:
                response_iterator: AsyncIterator[str]
                if isinstance(response_gen, str):
                    async def string_to_iterator(text: str):
                        yield text
                    response_iterator = string_to_iterator(response_gen)
                else:
                    response_iterator = response_gen
                
                await self.tts.synthesize(response_iterator)
                
            except asyncio.CancelledError:
                logger.info("[synthesize] Synthesis cancelled")
                self.emit("synthesis_interrupted", {})
                raise
            
            except Exception as e:
                logger.error(f"[synthesize] Error during synthesis: {e}")
                self.emit("synthesis_error", {"error": str(e)})
                raise
            
            finally:
                if self.agent and self.agent.session and self.agent.session.is_background_audio_enabled:
                    await self.agent.session.stop_thinking_audio()
                
                if self.agent and self.agent.session:
                    self.agent.session._reply_in_progress = False
                    self.agent.session._reset_wake_up_timer()
    
    async def interrupt(self) -> None:
        """Interrupt the current synthesis"""
        self._is_interrupted = True
        
        if self.tts:
            await self.tts.interrupt()
        
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
        logger.debug("[cleanup] Cleaning up speech generation")
        
        self.tts = None
        self.agent = None
        self.avatar = None
        self.audio_track = None
        
        logger.info("[cleanup] Speech generation cleaned up")
