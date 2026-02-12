from __future__ import annotations
from typing import Any, Callable, Optional, Literal, Awaitable
import asyncio
import uuid

from .agent import Agent
from .llm.chat_context import ChatRole
from .pipeline import Pipeline
from .utils import get_tool_info, UserState, AgentState
from .utterance_handle import UtteranceHandle 
import time
from .job import get_current_job_context
from .event_emitter import EventEmitter
from .event_bus import global_event_emitter
from .background_audio import BackgroundAudioHandler,BackgroundAudioHandlerConfig
from .dtmf_handler import DTMFHandler
from .voice_mail_detector import VoiceMailDetector
from .playground_manager import PlaygroundManager
from .metrics.unified_metrics_collector import UnifiedMetricsCollector
import logging
import av
logger = logging.getLogger(__name__)

class AgentSession(EventEmitter[Literal["user_state_changed", "agent_state_changed"]]):
    """
    Manages an agent session with its associated conversation flow and pipeline.
    """

    def __init__(
        self,
        agent: Agent,
        pipeline: Pipeline,
        wake_up: Optional[int] = None,
        background_audio: Optional[BackgroundAudioHandlerConfig] = None,
        dtmf_handler: Optional[DTMFHandler] = None,
        voice_mail_detector: Optional[VoiceMailDetector] = None,
    ) -> None:
        """
        Initialize an agent session.

        Args:
            agent: Instance of an Agent class that handles the core logic
            pipeline: Pipeline instance to process the agent's operations
            wake_up: Time in seconds after which to trigger wake-up callback if no speech detected
            background_audio: Configuration for background audio (optional)
            dtmf_handler: DTMF handler for phone number input (optional)
            voice_mail_detector: Voicemail detector (optional)
        """
        super().__init__()
        self.agent = agent
        self.pipeline = pipeline
        self.agent.session = self
        self.wake_up = wake_up
        self.on_wake_up: Optional[Callable[[], None] | Callable[[], Any]] = None
        self._wake_up_task: Optional[asyncio.Task] = None
        self._wake_up_timer_active = False
        self._closed: bool = False
        self._reply_in_progress: bool = False
        self._user_state: UserState = UserState.IDLE
        self._agent_state: AgentState = AgentState.IDLE
        self.current_utterance: Optional[UtteranceHandle] = None
        self._thinking_audio_player: Optional[BackgroundAudioHandler] = None
        self._background_audio_player: Optional[BackgroundAudioHandler] = None
        self._thinking_was_playing = False
        self.background_audio_config = background_audio
        self._is_executing_tool = False
        self._job_context = None
        self.dtmf_handler = dtmf_handler
        self.voice_mail_detector = voice_mail_detector
        self._is_voice_mail_detected = False
        self._playground_manager = None
        self._playground = False
        self._send_analytics_to_pubsub = False

        # Set agent on pipeline (pipeline handles all internal wiring)
        if hasattr(self.pipeline, 'set_agent'):
            self.pipeline.set_agent(self.agent)

        # Initialize analytics client
        from .metrics.analytics import AnalyticsClient
        analytics_client = AnalyticsClient()

        # Generate session ID
        session_id = str(uuid.uuid4())
        analytics_client.set_session_id(session_id)

        # Initialize unified metrics collector
        self.metrics_collector = UnifiedMetricsCollector(
            session_id=session_id,
            agent=self.agent,
            pipeline=self.pipeline,
            analytics_client=analytics_client,
            traces_flow_manager=None,  # Will be set later if needed
            playground_manager=None  # Will be set later if needed
        )

        # Set session-level information
        self.metrics_collector.session_metrics.system_instruction = agent.instructions or ""
        # room_id will be set when transport is initialized (in start() method)

        # Inject metrics collector into pipeline components for direct timing
        logger.info(f"[AGENT_SESSION DEBUG] self.metrics_collector = {self.metrics_collector}")
        logger.info(f"[AGENT_SESSION DEBUG] self.metrics_collector.turn_tracker = {self.metrics_collector.turn_tracker}")
        logger.info(f"[AGENT_SESSION DEBUG] Type: {type(self.metrics_collector.turn_tracker).__name__}")

        if hasattr(self.pipeline, 'orchestrator') and self.pipeline.orchestrator:
            logger.info(f"[AGENT_SESSION DEBUG] Calling orchestrator.set_metrics_collector with turn_tracker")
            self.pipeline.orchestrator.set_metrics_collector(self.metrics_collector.turn_tracker)
        elif hasattr(self.pipeline, 'set_metrics_collector'):
            # Direct pipeline support (for non-orchestrator pipelines)
            logger.info(f"[AGENT_SESSION DEBUG] Calling pipeline.set_metrics_collector with turn_tracker")
            self.pipeline.set_metrics_collector(self.metrics_collector.turn_tracker)

        # Setup voicemail detection
        if self.voice_mail_detector:
            if hasattr(self.pipeline, "set_voice_mail_detector"):
                self.pipeline.set_voice_mail_detector(self.voice_mail_detector)
            
            if hasattr(self.pipeline, "on"):
                self.pipeline.on("voicemail_result", self._handle_voicemail_result)

        # Setup wake-up callback
        if hasattr(self.pipeline, 'set_wake_up_callback'):
            self.pipeline.set_wake_up_callback(self._reset_wake_up_timer)


        # Get job context
        try:
            job_ctx = get_current_job_context()
            if job_ctx:
                self._job_context = job_ctx
                job_ctx.add_shutdown_callback(self.close)
                self._playground = job_ctx.room_options.playground
                self._send_analytics_to_pubsub = job_ctx.room_options.send_analytics_to_pubsub

        except Exception as e:
            logger.error(f"AgentSession: Error in session initialization: {e}")
            self._job_context = None

    @property
    def is_voicemail_detected(self) -> bool:
        """Returns True if voicemail was detected in this session."""
        return self._is_voicemail_detected

    def _handle_voicemail_result(self, data: dict) -> None:
        """
        Handler for the voicemail_result event from ConversationFlow.
        Updates session state and executes callback if needed.
        """
        is_vm = data.get("is_voicemail", False)
        self._is_voicemail_detected = is_vm
        
        if is_vm:
            logger.info("AgentSession: Voicemail confirmed. Executing callback.")
            if self.voice_mail_detector.callback:
                asyncio.create_task(self._safe_execute_vmd_callback())

    async def _safe_execute_vmd_callback(self) -> None:
        try:

            if self.voice_mail_detector.callback:
                await self.voice_mail_detector.callback()
        except Exception as e:
            logger.error(f"Error executing voicemail callback: {e}")


    def _start_wake_up_timer(self) -> None:
        if self.wake_up is not None and self.on_wake_up is not None:
            self._wake_up_timer_active = True
            self._wake_up_task = asyncio.create_task(self._wake_up_timer_loop())
    
    def _reset_wake_up_timer(self) -> None:
        if self.wake_up is not None and self.on_wake_up is not None:
            if self._reply_in_progress:
                return
            if self._wake_up_task and not self._wake_up_task.done():
                self._wake_up_task.cancel()
            self._wake_up_timer_active = True
            self._wake_up_task = asyncio.create_task(self._wake_up_timer_loop())
    
    def _pause_wake_up_timer(self) -> None:
        if self._wake_up_task and not self._wake_up_task.done():
            self._wake_up_task.cancel()
    
    def _cancel_wake_up_timer(self) -> None:
        if self._wake_up_task and not self._wake_up_task.done():
            self._wake_up_task.cancel()
        self._wake_up_timer_active = False
    
    async def _wake_up_timer_loop(self) -> None:
        try:
            await asyncio.sleep(self.wake_up)
            if self._wake_up_timer_active and self.on_wake_up and not self._reply_in_progress:
                if asyncio.iscoroutinefunction(self.on_wake_up):
                    await self.on_wake_up()
                else:
                    self.on_wake_up()
        except asyncio.CancelledError:
            pass

    def _emit_user_state(self, state: UserState, data: dict | None = None) -> None:
        if state != self._user_state:
            self._user_state = state
            payload = {"state": state.value, **(data or {})}
            self.emit("user_state_changed", payload)

    def _emit_agent_state(self, state: AgentState, data: dict | None = None) -> None:
        if state != self._agent_state:
            self._agent_state = state
            payload = {"state": state.value, **(data or {})}
            self.emit("agent_state_changed", payload)

    @property
    def user_state(self) -> UserState:
        return self._user_state

    @property
    def agent_state(self) -> AgentState:
        return self._agent_state

    @property
    def is_background_audio_enabled(self) -> bool:
        """Check if background audio is enabled in the pipeline"""
        audio_track = self._get_audio_track()
        return hasattr(audio_track, 'add_background_bytes')

    async def start(
        self,
        wait_for_participant: bool = False,
        run_until_shutdown: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Start the agent session.
        This will:
        1. Initialize the agent (including MCP tools if configured)
        2. Call the agent's on_enter hook
        3. Start the pipeline processing
        4. Start wake-up timer if configured (but only if callback is set)
        5. Optionally handle full lifecycle management (connect, wait, shutdown)
        
        Args:
            wait_for_participant: If True, wait for a participant to join before starting
            run_until_shutdown: If True, manage the full lifecycle including connection,
                               waiting for shutdown signals, and cleanup. This is a convenience
                               that internally calls ctx.run_until_shutdown() with this session.
            **kwargs: Additional arguments to pass to the pipeline start method
            
        Examples:
            Simple start (manual lifecycle management):
            ```python
            await session.start()
            ```
            
            Full lifecycle management (recommended):
            ```python
            await session.start(wait_for_participant=True, run_until_shutdown=True)
            ```
        """
        if run_until_shutdown:
            try:
                ctx = get_current_job_context()
                if ctx:
                    logger.info("Starting session with full lifecycle management")
                    await ctx.run_until_shutdown(
                        session=self,
                        wait_for_participant=wait_for_participant
                    )
                    return
                else:
                    logger.warning(
                        "run_until_shutdown=True requires a JobContext, "
                        "falling back to normal start()"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to get JobContext for run_until_shutdown: {e}, "
                    "falling back to normal start()"
                )
        
        self._emit_agent_state(AgentState.STARTING)
        await self.agent.initialize_mcp()

        if self.dtmf_handler:
            await self.dtmf_handler.start()

        if self._playground or self._send_analytics_to_pubsub:
            job_ctx = get_current_job_context()
            self.playground_manager = PlaygroundManager(job_ctx)
            # Update metrics collector with playground manager
            self.metrics_collector.playground_manager = self.playground_manager

        if hasattr(self.pipeline, 'set_agent'):
            self.pipeline.set_agent(self.agent)

            # CRITICAL: Re-inject metrics collector after set_agent() recreates orchestrator
            # The set_agent() call above recreates the orchestrator, wiping out metrics_collector
            logger.info(f"[AGENT_SESSION DEBUG] Re-injecting metrics collector after set_agent()")
            if hasattr(self.pipeline, 'orchestrator') and self.pipeline.orchestrator:
                self.pipeline.orchestrator.set_metrics_collector(self.metrics_collector.turn_tracker)
            elif hasattr(self.pipeline, 'set_metrics_collector'):
                self.pipeline.set_metrics_collector(self.metrics_collector.turn_tracker)

        await self.pipeline.start()

        # Start metrics collection
        await self.metrics_collector.start()

        if self._should_delay_for_sip_user():
            logger.info("SIP user detected, waiting for audio stream to be enabled before calling on_enter")
            audio_stream_enabled = asyncio.Event()

            def on_audio_stream_enabled(data):
                stream = data.get("stream")
                participant = data.get("participant")
                if stream and stream.kind == "audio" and participant and participant.meta_data.get("sipUser"):
                    logger.info(f"SIP user audio stream enabled for participant {participant.id}")
                    audio_stream_enabled.set()

            global_event_emitter.on("AUDIO_STREAM_ENABLED", on_audio_stream_enabled)

            async def wait_and_start():
                try:
                    await audio_stream_enabled.wait()
                    logger.info("SIP user audio stream enabled, proceeding with on_enter")
                    await self.agent.on_enter()
                    global_event_emitter.emit("AGENT_STARTED", {"session": self})
                    if self.on_wake_up is not None:
                        self._start_wake_up_timer()
                    self._emit_agent_state(AgentState.IDLE)
                except Exception as e:
                    logger.error(f"Error in wait_and_start: {e}")
                finally:
                    global_event_emitter.off("AUDIO_STREAM_ENABLED", on_audio_stream_enabled)

            asyncio.create_task(wait_and_start())
            return 

        await self.agent.on_enter()
        global_event_emitter.emit("AGENT_STARTED", {"session": self})
        if self.on_wake_up is not None:
            self._start_wake_up_timer()
        self._emit_agent_state(AgentState.IDLE)
        
    async def say(self, message: str, interruptible: bool = True) -> UtteranceHandle:
        """
        Send an initial message to the agent and return a handle to track it.
        """
        if self.current_utterance and not self.current_utterance.done():
            self.current_utterance.interrupt()
        handle = UtteranceHandle(utterance_id=f"utt_{uuid.uuid4().hex[:8]}", interruptible=interruptible)
        self.current_utterance = handle

        self.agent.chat_context.add_message(role=ChatRole.ASSISTANT, content=message)
        
        if hasattr(self.pipeline, 'send_message'):
            await self.pipeline.send_message(message, handle=handle)
        
        return handle
    
    async def play_background_audio(self, config: BackgroundAudioHandlerConfig, override_thinking: bool) -> None:
        """Play background audio on demand"""
        if override_thinking and self._thinking_audio_player and self._thinking_audio_player.is_playing:
            await self.stop_thinking_audio()
            self._thinking_was_playing = True

        audio_track = self._get_audio_track()
        if not hasattr(audio_track, 'add_background_bytes'):
            logger.warning(
                "Cannot play background audio. This feature requires the mixing audio track. "
                "Enable it by setting `background_audio=True` in RoomOptions."
            )
            return

        if audio_track:
            self._background_audio_player = BackgroundAudioHandler(config, audio_track)
            
            await self._background_audio_player.start()


    async def stop_background_audio(self) -> None:
        """Stop background audio on demand"""
        if self._background_audio_player:
            await self._background_audio_player.stop()
            self._background_audio_player = None

        if self._thinking_was_playing:
            await self.start_thinking_audio()
            self._thinking_was_playing = False

    def _get_audio_track(self):
        """Get audio track from pipeline"""
        if hasattr(self.pipeline, 'tts') and self.pipeline.tts and self.pipeline.tts.audio_track: # Cascading
            return self.pipeline.tts.audio_track
        elif hasattr(self.pipeline, 'model') and self.pipeline.model and self.pipeline.model.audio_track: # Realtime
            return self.pipeline.model.audio_track
        return None
    
    async def start_thinking_audio(self):
        """Start thinking audio"""
        if self._background_audio_player and self._background_audio_player.is_playing:
            return

        audio_track = self._get_audio_track()
        if not hasattr(audio_track, 'add_background_bytes'):
            logger.warning(
                "Cannot play 'thinking' audio. This feature requires the mixing audio track. "
                "Enable it by setting `background_audio=True` in RoomOptions."
            )
            return

        if self.agent._thinking_background_config and audio_track:
            self._thinking_audio_player = BackgroundAudioHandler(self.agent._thinking_background_config, audio_track)
            await self._thinking_audio_player.start()

    async def stop_thinking_audio(self):
        """Stop thinking audio"""
        if self._thinking_audio_player:
            await self._thinking_audio_player.stop()
            self._thinking_audio_player = None
    

    async def reply(self, instructions: str, wait_for_playback: bool = True, frames: list[av.VideoFrame] | None = None, interruptible: bool = True) -> UtteranceHandle:
        """
        Generate a response from agent using instructions and current chat context.
        
        This method is safe to call from function tools - it will automatically
        detect re-entrant calls and schedule them as background tasks.
        
        Args:
            instructions: Instructions to add to chat context
            wait_for_playback: If True, wait for playback to complete
            frames: Optional list of VideoFrame objects to include in the reply
            
        Returns:
            UtteranceHandle: A handle to track the utterance lifecycle
        """
        if self._reply_in_progress:
            if self.current_utterance:
                return self.current_utterance
            handle = UtteranceHandle(utterance_id="placeholder", interruptible=interruptible)
            handle._mark_done()
            return handle
        
        handle = UtteranceHandle(utterance_id=f"utt_{uuid.uuid4().hex[:8]}", interruptible=interruptible)
        self.current_utterance = handle

        if self._is_executing_tool:
            asyncio.create_task(
                self._internal_blocking_reply(instructions, wait_for_playback, handle, frames)
            )
            return handle
        else:
            await self._internal_blocking_reply(instructions, wait_for_playback, handle, frames)
            return handle

    async def _internal_blocking_reply(self, instructions: str, wait_for_playback: bool, handle: UtteranceHandle, frames: list[av.VideoFrame] | None = None) -> None:
        """
        The original, blocking logic of the reply method.
        """
        if not instructions:
            handle._mark_done()
            return
        self._reply_in_progress = True
        self._pause_wake_up_timer()
        
        try:
            # Call pipeline's reply_with_context
            if hasattr(self.pipeline, 'reply_with_context'):
                await self.pipeline.reply_with_context(instructions, wait_for_playback, handle=handle, frames=frames)

            if wait_for_playback:
                await handle

        finally:
            self._reply_in_progress = False
            if not handle.done():
                handle._mark_done() 

    def interrupt(self, *, force: bool = False) -> None:
        """
        Interrupt the agent's current speech.
        """
        if self.current_utterance and not self.current_utterance.interrupted:
            try:
                self.current_utterance.interrupt(force=force) 
            except RuntimeError as e:
                logger.warning(f"Could not interrupt utterance: {e}")
                return
        
        if hasattr(self.pipeline, 'interrupt'):
            self.pipeline.interrupt()

    async def close(self) -> None:
        """
        Close the agent session.
        """
        logger.info("Closing agent session")
        if self._closed:
            logger.info("Agent session already closed")
            return
        self._closed = True
        self._emit_agent_state(AgentState.CLOSING)

        # Cleanup metrics collection
        if hasattr(self, 'metrics_collector') and self.metrics_collector:
            try:
                await self.metrics_collector.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up metrics collector: {e}")

        self._cancel_wake_up_timer()
        

        logger.info("Cleaning up agent session")
        try:
            await self.agent.on_exit()
        except Exception as e:
            logger.error(f"Error in agent.on_exit(): {e}")
        
        if self._thinking_audio_player:
            await self._thinking_audio_player.stop()

        if self._background_audio_player:
            await self._background_audio_player.stop()

        try:
            await self.pipeline.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up pipeline: {e}")
        
        try:
            await self.agent.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up agent: {e}")
        
        self.agent = None
        self.pipeline = None
        self.on_wake_up = None
        self._wake_up_task = None
        logger.info("Agent session cleaned up")

    async def leave(self) -> None:
        """
        Leave the agent session.
        """
        self._emit_agent_state(AgentState.CLOSING)
        await self.pipeline.leave()

    async def hangup(self, reason: str = "manual_hangup") -> None:
        """
        Hang up the session, leaving the room immediately if possible.
        """
        job_ctx = self._job_context
        if not job_ctx:
            try:
                job_ctx = get_current_job_context()
            except Exception:
                job_ctx = None

        room = getattr(job_ctx, "room", None) if job_ctx else None
        if room and hasattr(room, "force_end_session"):
            try:
                await room.force_end_session(reason)
                return
            except Exception as exc:
                logger.error(f"Error forcing room to end session: {exc}")

        await self.close()
     
    async def call_transfer(self,token: str, transfer_to: str) -> None:
        """ Transfer the call to a provided Phone number or SIP endpoint.
        Args:
            token: VideoSDK auth token.
            transfer_to: Phone number or SIP endpoint to transfer the call to.
        """
        job_ctx = self._job_context
        if not job_ctx:
            try:
                job_ctx = get_current_job_context()
            except Exception:
                job_ctx = None

        room = getattr(job_ctx, "room", None) if job_ctx else None
        if room and hasattr(room, "call_transfer"):
            try:
                await room.call_transfer(token, transfer_to)
                return
            except Exception as exc:
                logger.error(f"Error calling call_transfer: {exc}")

    def _should_delay_for_sip_user(self) -> bool:
        """Check if there are SIP users in the room that need audio stream initialization"""
        job_ctx = self._job_context
        if not job_ctx:
            try:
                job_ctx = get_current_job_context()
            except Exception:
                job_ctx = None
        room = getattr(job_ctx, "room", None) if job_ctx else None
        if room and hasattr(room, "participants_data"):
            participants = room.participants_data
            for participant_info in participants.values():
                # SIP-specific on_enter logic is currently limited to outbound calls.
                if participant_info.get("sipUser") and participant_info.get("sipCallType") == "outbound":
                    return True
        return False