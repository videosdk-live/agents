import logging
from .room.room import VideoSDKHandler
from .pipeline import Pipeline
from typing import Callable, Coroutine, Optional, Any
import os
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum, unique

logger = logging.getLogger(__name__)

_current_job_context: ContextVar[Optional["JobContext"]] = ContextVar(
    "current_job_context", default=None
)


@dataclass
class RoomOptions:
    room_id: str
    auth_token: Optional[str] = None
    name: Optional[str] = "Agent"
    playground: bool = True
    vision: bool = False
    avatar: Optional[Any] = None
    join_meeting: Optional[bool] = True
    on_room_error: Optional[Callable[[Any], None]] = None
    # Session management options
    auto_end_session: bool = True
    session_timeout_seconds: Optional[int] = 30


class WorkerJob:
    def __init__(self, entrypoint, jobctx=None):
        """
        :param entrypoint: An async function accepting one argument: jobctx
        :param jobctx: A static object or a callable that returns a context per job
        """
        if not asyncio.iscoroutinefunction(entrypoint):
            raise TypeError("entrypoint must be a coroutine function")
        self.entrypoint = entrypoint
        self.jobctx = jobctx

    def start(self):
        from .worker import Worker

        worker = Worker(self)
        worker.run()


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

            if self.room_options.join_meeting:
                self.room = VideoSDKHandler(
                    meeting_id=self.room_options.room_id,
                    auth_token=self.videosdk_auth,
                    name=self.room_options.name,
                    pipeline=self._pipeline,
                    loop=self._loop,
                    vision=self.room_options.vision,
                    custom_camera_video_track=custom_camera_video_track,
                    custom_microphone_audio_track=custom_microphone_audio_track,
                    audio_sinks=sinks,
                    on_room_error=self.room_options.on_room_error,
                    auto_end_session=self.room_options.auto_end_session,
                    session_timeout_seconds=self.room_options.session_timeout_seconds,
                )
            if self._pipeline and hasattr(self._pipeline, "_set_loop_and_audio_track"):
                self._pipeline._set_loop_and_audio_track(
                    self._loop, self.room.audio_track
                )

        if self.room and self.room_options.join_meeting:
            self.room.init_meeting()
            await self.room.join()

        if self.room_options.playground and self.room_options.join_meeting:
            if self.videosdk_auth:
                playground_url = f"https://playground.videosdk.live?token={self.videosdk_auth}&meetingId={self.room_options.room_id}"
                logger.info("Agent started in playground mode")
                logger.info(f"Interact with agent here at: {playground_url}")
            else:
                raise ValueError("VIDEOSDK_AUTH_TOKEN environment variable not found")

    async def shutdown(self) -> None:
        """Called by Worker during graceful shutdown"""
        for callback in self._shutdown_callbacks:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Error in shutdown callback: {e}")

        if self.room:
            self.room.leave()
            await self.room.cleanup()
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
    job: Any
    url: str
    token: str
    worker_id: str
