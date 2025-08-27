import logging
from .room.room import VideoSDKHandler
from .pipeline import Pipeline
from typing import Callable, Coroutine, Optional, Any
import os
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum, unique
from typing import TYPE_CHECKING
import logging
import requests
import sys

if TYPE_CHECKING:
    from .worker import ExecutorType, WorkerPermissions, _default_executor_type
else:
    # Import at runtime to avoid circular imports
    ExecutorType = None
    WorkerPermissions = None
    _default_executor_type = None

logger = logging.getLogger(__name__)

_current_job_context: ContextVar[Optional["JobContext"]] = ContextVar(
    "current_job_context", default=None
)


@dataclass
class RoomOptions:
    room_id: Optional[str] = None
    auth_token: Optional[str] = None
    name: Optional[str] = "Agent"
    playground: bool = True
    vision: bool = False
    recording: bool = False
    avatar: Optional[Any] = None
    join_meeting: Optional[bool] = True
    on_room_error: Optional[Callable[[Any], None]] = None
    # Session management options
    auto_end_session: bool = True
    session_timeout_seconds: Optional[int] = 30
    # VideoSDK connection options
    signaling_base_url: Optional[str] = None


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

        if not self.auth_token:
            self.auth_token = os.getenv("VIDEOSDK_AUTH_TOKEN")


class WorkerJob:
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

        # Create worker and run with job context
        worker = Worker(worker_options)

        # If register=True, run the worker in backend mode (don't execute entrypoint immediately)
        if self.options.register:
            # Run the worker normally (for backend registration mode)
            Worker.run_worker(worker_options)
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
    def __init__(
        self,
        *,
        room_options: RoomOptions,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self.room_options = room_options
        self._loop = loop or asyncio.get_event_loop()
        self._pipeline: Optional[Pipeline] = None
        self.videosdk_auth = self.room_options.auth_token or os.getenv(
            "VIDEOSDK_AUTH_TOKEN"
        )
        self.room: Optional[VideoSDKHandler] = None
        self._shutdown_callbacks: list[Callable[[], Coroutine[None, None, None]]] = []
        self._is_shutting_down: bool = False
        self.want_console = (len(sys.argv) > 1 and sys.argv[1].lower() == "console")

    def _set_pipeline_internal(self, pipeline: Any) -> None:
        """Internal method called by pipeline constructors"""
        self._pipeline = pipeline
        if self.room:
            self.room.pipeline = pipeline
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
            if not self.room_options.room_id:
                self.room_options.room_id = self.get_room_id()
            custom_camera_video_track = None
            custom_microphone_audio_track = None
            sinks = []

            avatar = self.room_options.avatar
            if not avatar and self._pipeline and hasattr(self._pipeline, "avatar"):
                avatar = self._pipeline.avatar

            if avatar:
                await avatar.connect()
                custom_camera_video_track = avatar.video_track
                custom_microphone_audio_track = avatar.audio_track
                sinks.append(avatar)


            if self.want_console:
                from .console_mode import setup_console_voice_for_ctx
                if not self._pipeline:
                    raise RuntimeError("Pipeline must be constructed before ctx.connect() in console mode")
                cleanup_callback = await setup_console_voice_for_ctx(self)
                self.add_shutdown_callback(cleanup_callback)
            else:
                if self.room_options.join_meeting:
                    self.room = VideoSDKHandler(
                        meeting_id=self.room_options.room_id,
                        auth_token=self.videosdk_auth,
                        name=self.room_options.name,
                        pipeline=self._pipeline,
                        loop=self._loop,
                        vision=self.room_options.vision,
                        recording=self.room_options.recording,
                        custom_camera_video_track=custom_camera_video_track,
                        custom_microphone_audio_track=custom_microphone_audio_track,
                        audio_sinks=sinks,
                        on_room_error= self.room_options.on_room_error,
                        auto_end_session=self.room_options.auto_end_session,
                        session_timeout_seconds=self.room_options.session_timeout_seconds,
                        signaling_base_url=self.room_options.signaling_base_url,
                    )
                if self._pipeline and hasattr(self._pipeline, '_set_loop_and_audio_track'):
                    self._pipeline._set_loop_and_audio_track(self._loop, self.room.audio_track)

        if self.room and self.room_options.join_meeting:
            self.room.init_meeting()
            await self.room.join()

        if self.room_options.playground and self.room_options.join_meeting and not self.want_console:
            if self.videosdk_auth:
                playground_url = f"https://playground.videosdk.live?token={self.videosdk_auth}&meetingId={self.room_options.room_id}"
                print(f"\033[1;36m" + "Agent started in playground mode" + "\033[0m")
                print("\033[1;75m" + "Interact with agent here at:" + "\033[0m")
                print("\033[1;4;94m" + playground_url + "\033[0m")
            else:
                raise ValueError("VIDEOSDK_AUTH_TOKEN environment variable not found")

    async def shutdown(self) -> None:
        """Called by Worker during graceful shutdown"""
        if self._is_shutting_down:
            logger.info("JobContext already shutting down")
            return
        self._is_shutting_down = True
        for callback in self._shutdown_callbacks:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Error in shutdown callback: {e}")        
        if self.room:
            try:
                await self.room.leave()
            except Exception as e:
                logger.error(f"Error during room leave: {e}")
            try:
                if hasattr(self.room, "cleanup"):
                    await self.room.cleanup()
            except Exception as e:
                logger.error(f"Error during room cleanup: {e}")
            self.room = None

    def add_shutdown_callback(
        self, callback: Callable[[], Coroutine[None, None, None]]
    ) -> None:
        """Add a callback to be called during shutdown"""
        self._shutdown_callbacks.append(callback)

    async def wait_for_participant(self, participant_id: str | None = None) -> str:
        if self.room:
            return await self.room.wait_for_participant(participant_id)
        else:
            raise ValueError("Room not initialized")        

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
            url = "https://api.videosdk.live/v2/rooms"
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
            "VIDEOSDK_AUTH_TOKEN not found. "
            "Set it as an environment variable or provide it in room options via auth_token."
            )

def get_current_job_context() -> Optional['JobContext']:
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
    PROCESS = "process"
    THREAD = "thread"


@dataclass
class JobAcceptArguments:
    """Arguments for accepting a job."""

    identity: str
    name: str
    metadata: str = ""


@dataclass
class RunningJobInfo:
    """Information about a running job."""

    accept_arguments: JobAcceptArguments
    job: JobContext
    url: str
    token: str
    worker_id: str

    async def _run(self):
        # Placeholder for job execution logic if needed in the future
        pass
