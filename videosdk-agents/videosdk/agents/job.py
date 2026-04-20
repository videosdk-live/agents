import logging
from typing import Callable, Coroutine, Optional, Any, TYPE_CHECKING, Dict, Union
import os
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum, unique
import logging
import requests
import sys

if TYPE_CHECKING:
    from .pipeline import Pipeline
    from .worker import ExecutorType, WorkerPermissions, _default_executor_type
    from .transports.base import BaseTransportHandler
    from .room.room import VideoSDKHandler
else:
    # Import at runtime to avoid circular imports
    ExecutorType = None
    WorkerPermissions = None
    _default_executor_type = None
    BaseTransportHandler = object # Fallback if not checking types

logger = logging.getLogger(__name__)

_current_job_context: ContextVar[Optional["JobContext"]] = ContextVar(
    "current_job_context", default=None
)


class TransportMode(Enum):
    """Enumeration of supported transport modes for room connections."""

    VIDEOSDK = "videosdk"
    WEBSOCKET = "websocket"
    WEBRTC = "webrtc"


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket transport including port and endpoint path."""

    port: int = 8080
    path: str = "/ws"


@dataclass
class WebRTCConfig:
    """Configuration for WebRTC transport including signaling and ICE server settings."""

    signaling_url: Optional[str] = None
    signaling_type: str = "websocket"
    ice_servers: Optional[list] = None

    def __post_init__(self):
        if self.ice_servers is None:
            self.ice_servers = [{"urls": "stun:stun.l.google.com:19302"}]

@dataclass
class TracesOptions:
    """Configuration for OpenTelemetry trace export settings."""

    enabled: bool = True
    export_url: Optional[str] = None
    export_headers: Optional[Dict[str, str]] = None

@dataclass
class MetricsOptions:
    """Configuration for metrics collection and export settings."""

    enabled: bool = True
    export_url: Optional[str] = None
    export_headers: Optional[Dict[str, str]] = None

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})

@dataclass
class LoggingOptions:
    """Configuration for log collection, level filtering, and export settings.

    ``level`` accepts either:
      * a string (threshold, e.g. ``"INFO"``) — captures that level and above
      * a list of strings (explicit allowlist, e.g. ``["DEBUG", "ERROR"]``) —
        captures only the listed levels
    """

    enabled: bool = False
    level: Union[str, list[str]] = "INFO"
    export_url: Optional[str] = None
    export_headers: Optional[Dict[str, str]] = None
    send_to_dashboard: bool = False

    def __post_init__(self):
        if isinstance(self.level, str):
            upper = self.level.upper()
            if upper not in _VALID_LOG_LEVELS:
                raise ValueError(
                    f"Invalid log level {self.level!r}. Must be one of {sorted(_VALID_LOG_LEVELS)}."
                )
            self.level = upper
        elif isinstance(self.level, list):
            normalized: list[str] = []
            for lvl in self.level:
                if not isinstance(lvl, str):
                    raise ValueError(
                        f"level list entries must be strings, got {type(lvl).__name__}"
                    )
                upper = lvl.upper()
                if upper not in _VALID_LOG_LEVELS:
                    raise ValueError(
                        f"Invalid log level {lvl!r}. Must be one of {sorted(_VALID_LOG_LEVELS)}."
                    )
                normalized.append(upper)
            self.level = normalized
        else:
            raise ValueError(
                f"level must be str or list[str], got {type(self.level).__name__}"
            )


@dataclass
class ObservabilityOptions:
    """Grouped config for recording, traces, metrics, and logs.

    Semantics when used via this wrapper:
      * Field absent (``None``) — framework default
        (recording/logs OFF, traces/metrics ON via VideoSDK backend).
      * Field present — feature ON with the supplied config, unless a
        sub-option explicitly sets ``enabled=False``.

    Usable both on :class:`RoomOptions` and inline on
    ``AgentSession.start(observability=...)``.
    """

    recording: Optional["RecordingOptions"] = None
    traces: Optional[TracesOptions] = None
    metrics: Optional[MetricsOptions] = None
    logs: Optional[LoggingOptions] = None


@dataclass
class RecordingOptions:
    """
    Extra recording when RoomOptions.recording is True.

    Audio is always recorded when recording=True (track API, kind=audio).
    Set video and/or screen_share here only when you need them.
    screen_share=True requires RoomOptions.vision=True.
    """

    video: bool = False
    screen_share: bool = False


def _coerce_recording_options_dict(ro: dict) -> RecordingOptions:
    """Build RecordingOptions from a dict (e.g. backend JSON); ignores unknown keys."""
    return RecordingOptions(
        video=bool(ro.get("video", False)),
        screen_share=bool(ro.get("screen_share", False)),
    )


def validate_room_options_recording(room_options: "RoomOptions") -> None:
    """Raise ValueError if recording-related options are inconsistent.

    Evaluates the resolved observability config so both the flat-field and
    the new :class:`ObservabilityOptions` paths are checked.
    """
    if isinstance(room_options.recording_options, dict):
        room_options.recording_options = _coerce_recording_options_dict(
            room_options.recording_options
        )
    ro = room_options._resolved_observability().recording
    if ro is None:
        return
    if ro.screen_share and not room_options.vision:
        raise ValueError(
            "RoomOptions: recording_options.screen_share=True requires vision=True "
            "(vision subscribes to video/share streams required for screen recording)."
        )


def resolve_video_sdk_recording(
    room_options: "RoomOptions",
) -> tuple[Optional[bool], bool]:
    """
    Map the resolved observability recording config to VideoSDKHandler inputs.

    Returns:
        (record_audio, record_screen_share)
        - record_audio: None → participant recording (audio+video composite API);
          True → track recording, kind=audio only.
        - record_screen_share: whether to start screen_* track recording APIs.
    """
    ro = room_options._resolved_observability().recording
    if ro is None:
        return None, False
    if ro.video:
        return None, False
    if ro.screen_share:
        return True, True
    return True, False


@dataclass
class RoomOptions:
    """Configuration options for connecting to and managing a VideoSDK room, including transport, telemetry, and session settings."""

    room_id: Optional[str] = None
    auth_token: Optional[str] = None
    name: Optional[str] = "Agent"
    agent_participant_id: Optional[str] = None
    playground: bool = True
    vision: bool = False
    recording: bool = False
    # recording=True → always record audio (track API). Optional RecordingOptions.video /
    # RecordingOptions.screen_share for camera video and/or screen share (see validate/resolve).
    recording_options: Optional[RecordingOptions] = None
    avatar: Optional[Any] = None
    join_meeting: Optional[bool] = True
    on_room_error: Optional[Callable[[Any], None]] = None
    # Session management options
    auto_end_session: bool = True
    session_timeout_seconds: Optional[int] = 5
    no_participant_timeout_seconds: Optional[int] = 90
    # VideoSDK connection options
    signaling_base_url: Optional[str] = "api.videosdk.live"
    background_audio: bool = False

    send_logs_to_dashboard: bool = False
    dashboard_log_level: str = "INFO"

    # Telemetry and logging configurations
    traces: Optional[TracesOptions] = None
    metrics: Optional[MetricsOptions] = None
    logs: Optional[LoggingOptions] = None

    # Grouped observability (recording + traces + metrics + logs).
    # When set, takes precedence over the individual fields above at resolution time.
    observability: Optional[ObservabilityOptions] = None

    # New Configuration Fields
    _transport_mode: TransportMode = field(default=TransportMode.VIDEOSDK, init=False, repr=False)

    # Structured configs
    websocket: Optional[WebSocketConfig] = None
    webrtc: Optional[WebRTCConfig] = None

    # Alias properties for easier usage as requested
    @property
    def transport_mode(self) -> TransportMode:
        return self._transport_mode

    @transport_mode.setter
    def transport_mode(self, value):
        if isinstance(value, str):
            try:
                self._transport_mode = TransportMode(value.lower())
            except ValueError:
                # Fallback for compatibility or custom modes
                pass
        elif isinstance(value, TransportMode):
            self._transport_mode = value

    def __init__(
        self,
        transport_mode: Optional[str | TransportMode] = None,
        websocket: Optional[WebSocketConfig] = None,
        webrtc: Optional[WebRTCConfig] = None,
        traces: Optional[TracesOptions] = None,
        metrics: Optional[MetricsOptions] = None,
        logs: Optional[LoggingOptions] = None,
        observability: Optional[ObservabilityOptions] = None,
        **kwargs,
    ):
        # Initialize internal field
        self._transport_mode = TransportMode.VIDEOSDK

        # Handle telemetry options
        self.traces = traces or TracesOptions()
        self.metrics = metrics or MetricsOptions()
        self.logs = logs or LoggingOptions()

        # Grouped observability wrapper (optional; resolved lazily)
        self.observability = observability

        # Handle connection mode
        if transport_mode:
            if isinstance(transport_mode, str):
                try:
                    self._transport_mode = TransportMode(transport_mode.lower())
                except ValueError:
                    pass
            elif isinstance(transport_mode, TransportMode):
                self._transport_mode = transport_mode

        self.websocket = websocket or WebSocketConfig()
        self.webrtc = webrtc or WebRTCConfig()

        # Handle standard fields
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _resolved_observability(self) -> "ObservabilityOptions":
        """Merge observability sources into one effective config.

        Priority: ``self.observability`` > flat fields (``self.recording``,
        ``self.traces``, …) > framework defaults
        (recording/logs OFF, traces/metrics ON).
        """
        obs = self.observability

        if obs is not None and obs.recording is not None:
            recording = obs.recording
        elif self.recording:
            ro = self.recording_options
            if isinstance(ro, dict):
                ro = _coerce_recording_options_dict(ro)
            recording = ro if isinstance(ro, RecordingOptions) else RecordingOptions()
        else:
            recording = None

        if obs is not None and obs.traces is not None:
            traces = obs.traces
        else:
            traces = self.traces or TracesOptions()

        if obs is not None and obs.metrics is not None:
            metrics = obs.metrics
        else:
            metrics = self.metrics or MetricsOptions()

        if obs is not None and obs.logs is not None:
            src = obs.logs
            logs: Optional[LoggingOptions] = LoggingOptions(
                enabled=True,
                level=src.level,
                export_url=src.export_url,
                export_headers=src.export_headers,
                send_to_dashboard=src.send_to_dashboard,
            )
        elif (self.logs and self.logs.enabled) or self.send_logs_to_dashboard:
            base = self.logs or LoggingOptions()
            level_source = base.level if (self.logs and self.logs.enabled) else self.dashboard_log_level
            logs = LoggingOptions(
                enabled=True,
                level=level_source,
                export_url=base.export_url,
                export_headers=base.export_headers,
                send_to_dashboard=base.send_to_dashboard or self.send_logs_to_dashboard,
            )
        else:
            logs = None

        return ObservabilityOptions(
            recording=recording,
            traces=traces,
            metrics=metrics,
            logs=logs,
        )


@dataclass
class Options:
    """Configuration options for WorkerJob execution."""

    executor_type: Any = None  # Will be set in __post_init__
    """Which executor to use to run jobs. Automatically selected based on platform."""

    num_idle_processes: int = 1
    """Number of idle processes/threads to keep warm."""

    initialize_timeout: float = 10.0
    """Maximum amount of time to wait for a process/thread to initialize/prewarm"""

    close_timeout: float = 60.0
    """Maximum amount of time to wait for a job to shut down gracefully"""

    memory_warn_mb: float = 500.0
    """Memory warning threshold in MB."""

    memory_limit_mb: float = 0.0
    """Maximum memory usage for a job in MB. Defaults to 0 (disabled)."""

    ping_interval: float = 30.0
    """Interval between health check pings."""

    max_processes: int = 1
    """Maximum number of processes/threads."""

    agent_id: str = "VideoSDKAgent"
    """ID of the agent."""

    auth_token: Optional[str] = None
    """VideoSDK authentication token. Uses VIDEOSDK_AUTH_TOKEN env var if not provided."""

    permissions: Any = None  # Will be set in __post_init__
    """Permissions for the agent participant."""

    max_retry: int = 16
    """Maximum number of times to retry connecting to VideoSDK."""

    load_threshold: float = 0.75
    """Load threshold above which worker is marked as unavailable."""

    register: bool = False
    """Whether to register with the backend. Defaults to False for local development."""

    signaling_base_url: str = "api.videosdk.live"
    """Signaling base URL for VideoSDK services. Defaults to api.videosdk.live."""

    host: str = "0.0.0.0"
    """Host for the debug HTTP server."""

    port: int = 8081
    """Port for the debug HTTP server."""

    log_level: str = "INFO"
    """Log level for SDK logging. Options: DEBUG, INFO, WARNING, ERROR. Defaults to INFO."""

    def __post_init__(self):
        """Post-initialization setup."""
        # Import here to avoid circular imports
        from .worker import ExecutorType, WorkerPermissions, _default_executor_type

        if self.executor_type is None:
            self.executor_type = _default_executor_type

        if self.permissions is None:
            self.permissions = WorkerPermissions()

        from .utils import resolve_videosdk_auth_token
        self.auth_token = resolve_videosdk_auth_token(self.auth_token)


class WorkerJob:
    """Wraps an async entrypoint function and manages its execution either directly or via a Worker process."""

    def __init__(self, entrypoint, jobctx=None, options: Optional[Options] = None):
        """
        :param entrypoint: An async function accepting one argument: jobctx
        :param jobctx: A static object or a callable that returns a context per job
        :param options: Configuration options for job execution
        """
        if not asyncio.iscoroutinefunction(entrypoint):
            raise TypeError("entrypoint must be a coroutine function")
        self.entrypoint = entrypoint
        self.jobctx = jobctx
        self.options = options or Options()

    def start(self):
        from .worker import Worker, WorkerOptions

        # Convert JobOptions to WorkerOptions for compatibility
        worker_options = WorkerOptions(
            entrypoint_fnc=self.entrypoint,
            agent_id=self.options.agent_id,
            auth_token=self.options.auth_token,
            executor_type=self.options.executor_type,
            num_idle_processes=self.options.num_idle_processes,
            initialize_timeout=self.options.initialize_timeout,
            close_timeout=self.options.close_timeout,
            memory_warn_mb=self.options.memory_warn_mb,
            memory_limit_mb=self.options.memory_limit_mb,
            ping_interval=self.options.ping_interval,
            max_processes=self.options.max_processes,
            permissions=self.options.permissions,
            max_retry=self.options.max_retry,
            load_threshold=self.options.load_threshold,
            register=self.options.register,
            signaling_base_url=self.options.signaling_base_url,
            host=self.options.host,
            port=self.options.port,
            log_level=self.options.log_level,
        )

        # If register=True, run the worker in backend mode (don't execute entrypoint immediately)
        if self.options.register:
            default_room_options = None
            if self.jobctx:
                if callable(self.jobctx):
                    job_context = self.jobctx()
                else:
                    job_context = self.jobctx
                default_room_options = job_context.room_options
            # Run the worker normally (for backend registration mode)
            Worker.run_worker(
                options=worker_options, default_room_options=default_room_options
            )
        else:
            # Direct mode - run entrypoint immediately if we have a job context
            if self.jobctx:
                if callable(self.jobctx):
                    job_context = self.jobctx()
                else:
                    job_context = self.jobctx

                # Set the current job context and run the entrypoint
                token = _set_current_job_context(job_context)
                try:
                    asyncio.run(self.entrypoint(job_context))
                finally:
                    _reset_current_job_context(token)
            else:
                # No job context provided, run worker normally
                Worker.run_worker(worker_options)


class JobContext:
    """Holds the runtime state for a single job, including room connection, pipeline, and shutdown lifecycle management."""

    def __init__(
        self,
        *,
        room_options: RoomOptions,
        metadata: Optional[dict] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self.room_options = room_options
        self.metadata = metadata or {}
        self._loop = loop or asyncio.get_event_loop()
        self._pipeline: Optional["Pipeline"] = None
        from .utils import resolve_videosdk_auth_token
        self.videosdk_auth = resolve_videosdk_auth_token(self.room_options.auth_token)
        self.room: Optional["BaseTransportHandler"] = None
        self._shutdown_callbacks: list[Callable[[], Coroutine[None, None, None]]] = []
        self._is_shutting_down: bool = False
        self._meeting_joined_event: asyncio.Event = asyncio.Event()
        self._wait_for_meeting_join: bool = False
        self.want_console = len(sys.argv) > 1 and sys.argv[1].lower() == "console"
        
        from .metrics import metrics_collector
        self.metrics_collector = metrics_collector
        
        self._log_manager = None
        self._job_logger = None
        
    def _set_pipeline_internal(self, pipeline: Any) -> None:
        """Internal method called by pipeline constructors"""
        self._pipeline = pipeline
        if self.room:
            self.room.pipeline = pipeline
            if hasattr(self.room, 'input_stream_manager'):
                self.room.input_stream_manager.pipeline = pipeline

            # Reset audio track state for the new pipeline — a previous cascade
            # pipeline may have enabled manual_audio_control, which causes
            # interrupt() to set _accepting_audio=False, blocking all output.
            audio_track = getattr(self.room, 'audio_track', None)
            if audio_track:
                audio_track._manual_audio_control = False
                audio_track._accepting_audio = True

            if hasattr(pipeline, "_set_loop_and_audio_track"):
                pipeline._set_loop_and_audio_track(self._loop, self.room.audio_track)

            # Ensure our lambda function fix is preserved after pipeline setup
            # This prevents the pipeline from overriding our event handlers
            if hasattr(self.room, "meeting") and self.room.meeting:
                # Re-apply our lambda function fix to ensure it's not overridden
                self.room.meeting.add_event_listener(
                    self.room._create_meeting_handler()
                )

    async def connect(self) -> None:
        """Connect to the room"""
        if self.room_options:
            custom_camera_video_track = None
            custom_microphone_audio_track = None
            sinks = []

            avatar = self.room_options.avatar
            if not avatar and self._pipeline and hasattr(self._pipeline, "avatar"):
                avatar = self._pipeline.avatar

            if avatar:
                if not self.room_options.room_id:
                    env_room_id = (os.getenv("VIDEOSDK_ROOM_ID") or "").strip()
                    self.room_options.room_id = env_room_id or self.get_room_id()
                room_id = self.room_options.room_id

                from .avatar import AvatarAudioOut, generate_avatar_credentials

                if isinstance(avatar, AvatarAudioOut):
                    avatar.set_room_id(room_id)
                    await avatar.connect()
                    audio_out = avatar
                elif hasattr(avatar, 'participant_id'):
                    _api_key = os.getenv("VIDEOSDK_API_KEY")
                    _secret_key = os.getenv("VIDEOSDK_SECRET_KEY")
                    credentials = generate_avatar_credentials(
                        _api_key, _secret_key, participant_id=avatar.participant_id
                    )
                    await avatar.connect(room_id, credentials.token)
                    audio_out = AvatarAudioOut(credentials=credentials, room_id=room_id)
                    await audio_out.connect()  # no-op (no dispatcher_url)
                else:
                    await avatar.connect()
                    audio_out = avatar

                custom_camera_video_track = getattr(avatar, 'video_track', None)
                custom_microphone_audio_track = getattr(avatar, 'audio_track', None)
                sinks.append(audio_out)
                self._cloud_avatar = avatar if not isinstance(avatar, AvatarAudioOut) else None
                self._avatar_audio_out = audio_out if isinstance(audio_out, AvatarAudioOut) else None
                if self._pipeline:
                    self._pipeline.avatar = audio_out

            if self.want_console:
                from .console_mode import setup_console_voice_for_ctx

                if not self._pipeline:
                    raise RuntimeError(
                        "Pipeline must be constructed before ctx.connect() in console mode"
                    )
                cleanup_callback = await setup_console_voice_for_ctx(self)
                self.add_shutdown_callback(cleanup_callback)
            else:
                resolved_obs = self.room_options._resolved_observability()
                self.metrics_collector.transport_mode = self.room_options.transport_mode
                self.metrics_collector.analytics_client.configure(resolved_obs.metrics)
                if self.room_options.transport_mode == TransportMode.VIDEOSDK:
                    from .room.room import VideoSDKHandler

                    if not self.room_options.room_id:
                        env_room_id = (os.getenv("VIDEOSDK_ROOM_ID") or "").strip()
                        self.room_options.room_id = env_room_id or self.get_room_id()
                    if resolved_obs.logs is not None:
                        from .metrics.logger_handler import LogManager, JobLogger
                        self._log_manager = LogManager()
                        self._log_manager.start(auth_token=self.videosdk_auth or "")
                        self._job_logger = JobLogger(
                            queue=self._log_manager.get_queue(),
                            room_id=self.room_options.room_id or "",
                            peer_id=self.room_options.agent_participant_id or "agent",
                            auth_token=self.videosdk_auth or "",
                            dashboard_log_level=resolved_obs.logs.level,
                            send_logs_to_dashboard=True,
                        )

                    if self.room_options.join_meeting:
                        validate_room_options_recording(self.room_options)
                        record_audio_resolved, record_screen_share = resolve_video_sdk_recording(
                            self.room_options
                        )
                        agent_id = self._pipeline.agent.id if self._pipeline and hasattr(self._pipeline, 'agent') else None
                        self.room = VideoSDKHandler(
                            meeting_id=self.room_options.room_id,
                            auth_token=self.videosdk_auth,
                            name=self.room_options.name,
                            agent_participant_id=self.room_options.agent_participant_id,
                            agent_id=agent_id,
                            pipeline=self._pipeline,
                            loop=self._loop,
                            vision=self.room_options.vision,
                            recording=resolved_obs.recording is not None,
                            record_audio=record_audio_resolved,
                            record_screen_share=record_screen_share,
                            custom_camera_video_track=custom_camera_video_track,
                            custom_microphone_audio_track=custom_microphone_audio_track,
                            audio_sinks=sinks,
                            background_audio=self.room_options.background_audio,
                            on_room_error=self.room_options.on_room_error,
                            auto_end_session=self.room_options.auto_end_session,
                            session_timeout_seconds=self.room_options.session_timeout_seconds,
                            no_participant_timeout_seconds=self.room_options.no_participant_timeout_seconds,
                            signaling_base_url=self.room_options.signaling_base_url,
                            job_logger=self._job_logger,
                            traces_options=resolved_obs.traces,
                            metrics_options=resolved_obs.metrics,
                            logs_options=resolved_obs.logs,
                            avatar_participant_id=avatar.participant_id if avatar and hasattr(avatar, 'participant_id') else None,
                        )
                    if self._pipeline and hasattr(
                        self._pipeline, "_set_loop_and_audio_track"
                    ):
                        self._pipeline._set_loop_and_audio_track(
                            self._loop, self.room.audio_track
                        )

                elif self.room_options.transport_mode == TransportMode.WEBSOCKET:
                    if not self.room_options.websocket:
                        raise ValueError("WebSocket configuration (websocket) is required when mode is WEBSOCKET")
                    
                    if self.room_options.webrtc and (self.room_options.webrtc.signaling_url or self.room_options.webrtc.ice_servers != [{"urls": "stun:stun.l.google.com:19302"}]):
                        logger.warning("WebRTC configuration provided but transport mode is set to WEBSOCKET. WebRTC config will be ignored.")

                    from .transports.websocket_handler import WebSocketTransportHandler
                    self.room = WebSocketTransportHandler(
                        loop=self._loop,
                        pipeline=self._pipeline,
                        port=self.room_options.websocket.port,
                        path=self.room_options.websocket.path
                    )
                elif self.room_options.transport_mode == TransportMode.WEBRTC:
                    if not self.room_options.webrtc:
                        raise ValueError("WebRTC configuration (webrtc) is required when mode is WEBRTC")
                    
                    if not self.room_options.webrtc.signaling_url:
                        raise ValueError("WebRTC signaling_url is required when mode is WEBRTC")

                    if self.room_options.websocket and (self.room_options.websocket.port != 8080 or self.room_options.websocket.path != "/ws"):
                        logger.warning("WebSocket configuration provided but connection mode is set to WEBRTC. WebSocket config will be ignored.")

                    from .transports.webrtc_handler import WebRTCTransportHandler
                    self.room = WebRTCTransportHandler(
                        loop=self._loop,
                        pipeline=self._pipeline,
                        signaling_url=self.room_options.webrtc.signaling_url,
                        ice_servers=self.room_options.webrtc.ice_servers
                    )
                
                elif self.room_options.transport_mode == TransportMode.VIDEOSDK:
                    if self.room_options.websocket and (self.room_options.websocket.port != 8080 or self.room_options.websocket.path != "/ws"):
                         logger.warning("WebSocket configuration provided but transport mode is VIDEOSDK. WebSocket config will be ignored.")
                    if self.room_options.webrtc and (self.room_options.webrtc.signaling_url or self.room_options.webrtc.ice_servers != [{"urls": "stun:stun.l.google.com:19302"}]):
                         logger.warning("WebRTC configuration provided but transport mode is VIDEOSDK. WebRTC config will be ignored.")

        if self.room:
            await self.room.connect()

            # For Non-VideoSDK modes, we still need to ensure audio track is linked if not done inside constructor
            if (
                self.room_options.transport_mode != TransportMode.VIDEOSDK
                and self._pipeline
                and hasattr(self._pipeline, "_set_loop_and_audio_track")
            ):
                # BaseTransportHandler subclasses now initialize self.audio_track
                if self.room.audio_track:
                    self._pipeline._set_loop_and_audio_track(self._loop, self.room.audio_track)

        if (
            self.room_options.playground
            and self.room_options.join_meeting
            and not self.want_console
            and self.room_options.transport_mode == TransportMode.VIDEOSDK
        ):
            if self.videosdk_auth:
                playground_url = f"https://playground.videosdk.live?token={self.videosdk_auth}&meetingId={self.room_options.room_id}"
                print(f"\033[1;36m" + "Agent started in playground mode" + "\033[0m")
                print("\033[1;75m" + "Interact with agent here at:" + "\033[0m")
                print("\033[1;4;94m" + playground_url + "\033[0m")
            else:
                raise ValueError(
                    "No VideoSDK auth available. Provide auth_token in RoomOptions, "
                    "set VIDEOSDK_AUTH_TOKEN, or set VIDEOSDK_API_KEY + VIDEOSDK_SECRET_KEY."
                )

    async def shutdown(self) -> None:
        """Called by Worker during graceful shutdown"""
        if self._is_shutting_down:
            logger.info("JobContext already shutting down")
            return
        self._is_shutting_down = True
        logger.info("JobContext shutting down")
        for callback in self._shutdown_callbacks:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Error in shutdown callback: {e}")

        if self._pipeline:
            try:
                await self._pipeline.cleanup()
            except Exception as e:
                logger.error(f"Error during pipeline cleanup: {e}")
            self._pipeline = None

        cloud_avatar = getattr(self, '_cloud_avatar', None)
        if cloud_avatar and hasattr(cloud_avatar, 'aclose'):
            try:
                await cloud_avatar.aclose()
            except Exception as e:
                logger.error(f"Error during cloud avatar aclose: {e}")
        audio_out = getattr(self, '_avatar_audio_out', None)
        if audio_out:
            try:
                await audio_out.aclose()
            except Exception as e:
                logger.error(f"Error during avatar audio_out aclose: {e}")

        if self._job_logger:
            try:
                self._job_logger.cleanup()
            except Exception as e:
                logger.error(f"Error during job logger cleanup: {e}")
            self._job_logger = None
        if self._log_manager:
            try:
                self._log_manager.stop()
            except Exception as e:
                logger.error(f"Error during log manager stop: {e}")
            self._log_manager = None

        if self.room:
            try:
                if not getattr(self.room, "_left", False):
                    await self.room.leave()
                else:
                    logger.info("Room already left, skipping room.leave()")
            except Exception as e:
                logger.error(f"Error during room leave: {e}")
            try:
                if hasattr(self.room, "cleanup"):
                    await self.room.cleanup()
            except Exception as e:
                logger.error(f"Error during room cleanup: {e}")
            self.room = None

        self.room_options = None
        self._loop = None
        self.videosdk_auth = None
        self._shutdown_callbacks.clear()
        logger.info("JobContext cleaned up")

    def add_shutdown_callback(
        self, callback: Callable[[], Coroutine[None, None, None]]
    ) -> None:
        """Add a callback to be called during shutdown"""
        self._shutdown_callbacks.append(callback)

    def notify_meeting_joined(self) -> None:
        """Called when the agent successfully joins the meeting."""
        self._meeting_joined_event.set()
        audio_out = getattr(self, '_avatar_audio_out', None)
        if audio_out and self.room and self.room.meeting:
            audio_out._set_meeting(self.room.meeting)

    async def wait_for_meeting_joined(self, timeout: float = 30.0) -> bool:
        """Wait until the meeting is joined or timeout. Returns True if joined."""
        try:
            await asyncio.wait_for(self._meeting_joined_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for meeting join after {timeout}s")
            return False

    async def wait_for_participant(self, participant_id: str | None = None) -> str:
        if self.room:
            return await self.room.wait_for_participant(participant_id)
        else:
            raise ValueError("Room not initialized")

    async def run_until_shutdown(
        self,
        session: Any = None,
        wait_for_participant: bool = False,
    ) -> None:
        """
        Simplified helper that handles all cleanup boilerplate.

        This method:
        1. Connects to the room
        2. Sets up session end callbacks
        3. Waits for participant (optional)
        4. Starts the session
        5. Waits for shutdown signal
        6. Cleans up gracefully

        Args:
            session: AgentSession to manage (will call session.start() and session.close())
            wait_for_participant: Whether to wait for a participant before starting

        Example:
            ```python
            async def entrypoint(ctx: JobContext):
                session = AgentSession(agent=agent, pipeline=pipeline)
                await ctx.run_until_shutdown(session=session, wait_for_participant=True)
            ```
        """
        shutdown_event = asyncio.Event()

        if session:

            async def cleanup_session():
                logger.info("Cleaning up session...")
                try:
                    await session.close()
                except Exception as e:
                    logger.error(f"Error closing session in cleanup: {e}")
                shutdown_event.set()

            self.add_shutdown_callback(cleanup_session)
        else:

            async def cleanup_no_session():
                logger.info("Shutdown called, no session to clean up")
                shutdown_event.set()

            self.add_shutdown_callback(cleanup_no_session)

        def on_session_end(reason: str):
            logger.info(f"Session ended: {reason}")
            asyncio.create_task(self.shutdown())

        try:
            try:
                await self.connect()
            except Exception as e:
                logger.error(f"Error connecting to room: {e}")
                raise

            if self.room:
                try:
                    self.room.setup_session_end_callback(on_session_end)
                    logger.info("Session end callback configured")
                except Exception as e:
                    logger.warning(f"Error setting up session end callback: {e}")
            else:
                logger.warning(
                    "Room not available, session end callback not configured"
                )

            if wait_for_participant and self.room:
                try:
                    logger.info("Waiting for participant...")
                    participant_id = await self.room.wait_for_participant()
                    if participant_id is None:
                        logger.info("Session ended before any participant joined, shutting down")
                        return
                    logger.info("Participant joined")
                except Exception as e:
                    logger.error(f"Error waiting for participant: {e}")
                    raise

            if session:
                try:
                    await session.start()
                    logger.info("Agent session started")
                except Exception as e:
                    logger.error(f"Error starting session: {e}")
                    raise

            logger.info(
                "Agent is running... (will exit when session ends or on interrupt)"
            )
            await shutdown_event.wait()
            logger.info("Shutdown event received, exiting gracefully...")

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error in run_until_shutdown: {e}")
            raise
        finally:
            if session:
                try:
                    await session.close()
                except Exception as e:
                    logger.error(f"Error closing session in finally: {e}")

            try:
                await self.shutdown()
            except Exception as e:
                logger.error(f"Error in ctx.shutdown: {e}")

    def get_room_id(self) -> str:
        """
        Creates a new room using the VideoSDK API and returns the room ID.

        Raises:
            ValueError: If the VIDEOSDK_AUTH_TOKEN is missing.
            RuntimeError: If the API request fails or the response is invalid.
        """
        if self.want_console:
            return None

        if self.videosdk_auth:
            base_url = self.room_options.signaling_base_url
            url = f"https://{base_url}/v2/rooms"
            headers = {"Authorization": self.videosdk_auth}

            try:
                response = requests.post(url, headers=headers)
                response.raise_for_status()
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to create room: {e}") from e

            data = response.json()
            room_id = data.get("roomId")
            if not room_id:
                raise RuntimeError(f"Unexpected API response, missing roomId: {data}")

            return room_id
        else:
            raise ValueError(
                "No VideoSDK auth available. Provide auth_token in RoomOptions, "
                "set VIDEOSDK_AUTH_TOKEN, or set VIDEOSDK_API_KEY + VIDEOSDK_SECRET_KEY."
            )


def get_current_job_context() -> Optional["JobContext"]:
    """Get the current job context (used by pipeline constructors)"""
    return _current_job_context.get()


def _set_current_job_context(ctx: "JobContext") -> Any:
    """Set the current job context (used by Worker)"""
    return _current_job_context.set(ctx)


def _reset_current_job_context(token: Any) -> None:
    """Reset the current job context (used by Worker)"""
    _current_job_context.reset(token)


@unique
class JobExecutorType(Enum):
    """Enumeration of executor types for running jobs in separate processes or threads."""

    PROCESS = "process"
    THREAD = "thread"


@dataclass
class JobAcceptArguments:
    """Holds identity, name, and metadata used when accepting a job from the worker pool."""

    identity: str
    name: str
    metadata: str = ""


@dataclass
class RunningJobInfo:
    """Tracks a running job's context, connection details, and associated worker identity."""

    accept_arguments: JobAcceptArguments
    job: JobContext
    url: str
    token: str
    worker_id: str

    async def _run(self):
        # Placeholder for job execution logic if needed in the future
        pass
