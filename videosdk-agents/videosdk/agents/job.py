from .room.room import VideoSDKHandler
from .pipeline import Pipeline
from typing import Callable, Coroutine, Optional, Any
import os
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from .metrics import metrics_collector
from .metrics.integration import auto_initialize_telemetry_and_logs

_current_job_context: ContextVar[Optional['JobContext']] = ContextVar('current_job_context', default=None)

@dataclass
class RoomOptions:
    room_id: str
    auth_token: Optional[str] = None
    name: Optional[str] = "Agent"
    playground: bool = True
    vision: bool = False
    avatar: Optional[Any] = None
    join_meeting: Optional[bool] = True

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
        self.videosdk_auth = self.room_options.auth_token or os.getenv("VIDEOSDK_AUTH_TOKEN")
        self.room: Optional[VideoSDKHandler] = None
        self._shutdown_callbacks: list[Callable[[], Coroutine[None, None, None]]] = []
        self._session_id_collected = False
        self._session_id: Optional[str] = None
    
    def _set_pipeline_internal(self, pipeline: Any) -> None:
        """Internal method called by pipeline constructors"""
        self._pipeline = pipeline
        if self.room:
            self.room.pipeline = pipeline
            if hasattr(pipeline, '_set_loop_and_audio_track'):
                pipeline._set_loop_and_audio_track(self._loop, self.room.audio_track)

    async def connect(self) -> None:
        """Connect to the room"""
        if self.room_options:
            custom_camera_video_track = None
            custom_microphone_audio_track = None
            sinks = []
            
            avatar = self.room_options.avatar
            if not avatar and self._pipeline and hasattr(self._pipeline, 'avatar'):
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
                )
            if self._pipeline and hasattr(self._pipeline, '_set_loop_and_audio_track'):
                self._pipeline._set_loop_and_audio_track(self._loop, self.room.audio_track)

        if self.room and self.room_options.join_meeting:
            self.room.init_meeting()
            await self.room.join()
            await asyncio.sleep(2)
            await self._collect_session_id()
            await self._collect_meeting_attributes()
        
        if self.room_options.playground and self.room_options.join_meeting:
            if self.videosdk_auth:
                playground_url = f"https://playground.videosdk.live?token={self.videosdk_auth}&meetingId={self.room_options.room_id}"
                print(f"\033[1;36m" + "Agent started in playground mode" + "\033[0m")
                print("\033[1;75m" + "Interact with agent here at:" + "\033[0m")
                print("\033[1;4;94m" + playground_url + "\033[0m")
            else:
                raise ValueError("VIDEOSDK_AUTH_TOKEN environment variable not found")

    async def _collect_session_id(self) -> None:
        """Collect session ID from room and set it in metrics collector"""
        if self.room and self.room.meeting and not self._session_id_collected:
            try:
                session_id = getattr(self.room.meeting, 'session_id', None)
                if session_id:
                    self._session_id = session_id
                    metrics_collector.set_session_id(session_id)
                    self._session_id_collected = True
            except Exception as e:
                print(f"Error collecting session ID: {e}")

    async def _collect_meeting_attributes(self) -> None:
        """Collect meeting attributes from room and initialize telemetry and logs"""
        if self.room and self.room.meeting:
            try:
                if hasattr(self.room.meeting, 'get_attributes'):
                    attributes = self.room.meeting.get_attributes()
                    if attributes:
                        peer_id = getattr(self.room.meeting, 'participant_id', 'agent')
                        auto_initialize_telemetry_and_logs(
                            room_id=self.room_options.room_id,
                            peer_id=peer_id,
                            room_attributes=attributes,
                            session_id=self._session_id
                        )
                    else:
                        print("No meeting attributes found")
                else:
                    print("Meeting object does not have get_attributes method")
            except Exception as e:
                print(f"Error collecting meeting attributes: {e}")

    async def shutdown(self) -> None:
        """Called by Worker during graceful shutdown"""
        for callback in self._shutdown_callbacks:
            try:
                await callback()
            except Exception as e:
                print(f"Error in shutdown callback: {e}")
        
        if self.room:
            self.room.leave()
            self.room.cleanup()
            self.room = None

    def add_shutdown_callback(self, callback: Callable[[], Coroutine[None, None, None]]) -> None:
        """Add a callback to be called during shutdown"""
        self._shutdown_callbacks.append(callback)
    
    async def wait_for_participant(self, participant_id: str | None = None) -> str:
        if self.room:
            return await self.room.wait_for_participant(participant_id)
        else:
            raise ValueError("Room not initialized")

def get_current_job_context() -> Optional['JobContext']:
    """Get the current job context (used by pipeline constructors)"""
    return _current_job_context.get()

def _set_current_job_context(ctx: 'JobContext') -> Any:
    """Set the current job context (used by Worker)"""
    return _current_job_context.set(ctx)

def _reset_current_job_context(token: Any) -> None:
    """Reset the current job context (used by Worker)"""
    _current_job_context.reset(token)

