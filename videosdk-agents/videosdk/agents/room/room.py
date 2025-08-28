import logging
from videosdk import (
    VideoSDK,
    Participant,
    Stream,
    PubSubPublishConfig,
    PubSubSubscribeConfig,
)
from .meeting_event_handler import MeetingHandler
from .participant_event_handler import ParticipantHandler
from .audio_stream import TeeCustomAudioStreamTrack
from videosdk.agents.pipeline import Pipeline
from dotenv import load_dotenv
import numpy as np
import asyncio
import os
from asyncio import AbstractEventLoop
from ..metrics.traces_flow import TracesFlowManager
from ..metrics import cascading_metrics_collector
from ..metrics.integration import auto_initialize_telemetry_and_logs
from typing import Callable, Optional, Any
from ..metrics.realtime_metrics_collector import realtime_metrics_collector
import requests
import time
import logging
from ..event_bus import global_event_emitter
logger = logging.getLogger(__name__)

START_RECORDING_URL = "https://api.videosdk.live/v2/recordings/participant/start"
STOP_RECORDING_URL = "https://api.videosdk.live/v2/recordings/participant/stop"
MERGE_RECORDINGS_URL = "https://api.videosdk.live/v2/recordings/participant/merge"

load_dotenv()


class VideoSDKHandler:
    def __init__(
        self,
        *,
        meeting_id: str,
        auth_token: str | None = None,
        name: str,
        pipeline: Pipeline,
        loop: AbstractEventLoop,
        vision: bool = False,
        recording: bool = False,
        custom_camera_video_track=None,
        custom_microphone_audio_track=None,
        audio_sinks=None,
        on_room_error: Optional[Callable[[Any], None]] = None,
        # Session management options
        auto_end_session: bool = True,
        session_timeout_seconds: Optional[int] = None,
        on_session_end: Optional[Callable[[str], None]] = None,
        # VideoSDK connection options
        signaling_base_url: Optional[str] = None,
    ):
        self.meeting_id = meeting_id
        self.auth_token = auth_token
        self.name = name
        self.pipeline = pipeline
        self.loop = loop
        self.vision = vision
        self.custom_camera_video_track = custom_camera_video_track
        self.custom_microphone_audio_track = custom_microphone_audio_track
        self.audio_sinks = audio_sinks or []

        # Session management
        self.auto_end_session = auto_end_session
        self.session_timeout_seconds = session_timeout_seconds
        self.on_session_end = on_session_end
        self._session_ended = False
        self._session_end_task = None

        # VideoSDK connection
        self.signaling_base_url = signaling_base_url

        # Participant tracking
        self._non_agent_participant_count = 0
        self._first_participant_event = asyncio.Event()
        self._participant_joined_events = {}

        # Meeting and event handling
        self.meeting = None
        self.participants_data = {}
        self.audio_listener_tasks = {}
        self.video_listener_tasks = {}

        self._meeting_joined_data = None
        self.agent_meeting = None
        self._session_id: Optional[str] = None
        self._session_id_collected = False
        self.recording = recording

        self.traces_flow_manager = TracesFlowManager(room_id=self.meeting_id)
        cascading_metrics_collector.set_traces_flow_manager(
            self.traces_flow_manager)

        if custom_microphone_audio_track:
            self.audio_track = custom_microphone_audio_track
            if audio_sinks:
                self.agent_audio_track = TeeCustomAudioStreamTrack(
                    loop=self.loop, sinks=audio_sinks, pipeline=pipeline
                )
            else:
                self.agent_audio_track = None
        else:
            self.audio_track = TeeCustomAudioStreamTrack(
                loop=self.loop, sinks=audio_sinks, pipeline=pipeline
            )
            self.agent_audio_track = None

        self.auth_token = auth_token or os.getenv("VIDEOSDK_AUTH_TOKEN")
        if not self.auth_token:
            raise ValueError("VIDEOSDK_AUTH_TOKEN is not set")

        # Create meeting config as a dictionary instead of using MeetingConfig
        self.meeting_config = {
            "name": self.name,
            "meeting_id": self.meeting_id,
            "token": self.auth_token,
            "mic_enabled": True,
            "webcam_enabled": custom_camera_video_track is not None,
            "custom_microphone_audio_track": self.audio_track,
            "custom_camera_video_track": custom_camera_video_track,
        }
        if self.signaling_base_url is not None:
            self.meeting_config["signaling_base_url"] = self.signaling_base_url

        self.attributes = {}
        self.on_room_error = on_room_error
        self._participant_joined_events: dict[str, asyncio.Event] = {}
        self._left: bool = False
        # Session management
        self.auto_end_session = auto_end_session

    def init_meeting(self):
        self._left: bool = False
        self.sdk_metadata = {
            "sdk": "agents",
            "sdk_version": "0.0.28"
        }

        self.meeting = VideoSDK.init_meeting(
            **self.meeting_config, sdk_metadata=self.sdk_metadata)
        self.meeting.add_event_listener(
            MeetingHandler(
                on_meeting_joined=self.on_meeting_joined,
                on_meeting_left=self.on_meeting_left,
                on_participant_joined=self.on_participant_joined,
                on_participant_left=self.on_participant_left,
                on_error=self.on_error,
            )
        )

    async def join(self):
        await self.meeting.async_join()

    async def leave(self):
        if self._left:
            logger.info("Meeting already left")
            return
        self._left = True
        for audio_task in list(self.audio_listener_tasks.values()):
            try:
                audio_task.cancel()
            except Exception:
                pass
        for video_task in list(self.video_listener_tasks.values()):
            try:
                video_task.cancel()
            except Exception:
                pass
        if self.traces_flow_manager:
            try:
                self.traces_flow_manager.agent_meeting_end()
            except Exception as e:
                logger.error(f"Error while ending agent_meeting_end span: {e}")
        if self.recording:
            try:
                await self.stop_and_merge_recordings()
            except Exception as e:
                logger.error(f"Error stopping/merging recordings: {e}")
        try:
            self.meeting.leave()
        except Exception as e:
            logger.error(f"Error leaving meeting: {e}")

    def on_error(self, data):
        if self.on_room_error:
            self.on_room_error(data)

    def on_meeting_joined(self, data):
        logger.info(f"Agent joined the meeting")
        self._meeting_joined_data = data
        asyncio.create_task(self._collect_session_id())
        asyncio.create_task(self._collect_meeting_attributes())
        if self.recording:
            asyncio.create_task(
                self.start_participant_recording(
                    self.meeting.local_participant.id)
            )

    def on_meeting_left(self, data):
        logger.info(f"Meeting Left: {data}")
        self._cancel_session_end_task()

        # Handle job cleanup if we have a session end callback
        if hasattr(self, "on_session_end") and self.on_session_end:
            try:
                self.on_session_end("meeting_left")
            except Exception as e:
                logger.error(
                    f"Error in session end callback during meeting left: {e}")

    def _is_agent_participant(self, participant: Participant) -> bool:
        """Check if a participant is an agent (based on name or other criteria)."""
        # Consider participants with names containing 'agent' or matching our agent name as agents
        participant_name = participant.display_name.lower()
        return (
            "agent" in participant_name
            or participant_name == self.name.lower()
            or participant.id == self.meeting.local_participant.id
            if self.meeting and self.meeting.local_participant
            else False
        )

    def _update_non_agent_participant_count(self):
        """Update the count of non-agent participants."""
        if not self.meeting:
            return

        count = 0
        for participant in self.meeting.participants.values():
            if not self._is_agent_participant(participant):
                count += 1

        self._non_agent_participant_count = count
        logger.debug(f"Non-agent participant count: {count}")

    def _cancel_session_end_task(self):
        """Cancel the session end task if it exists."""
        if self._session_end_task and not self._session_end_task.done():
            self._session_end_task.cancel()
            self._session_end_task = None

    async def _end_session(self, reason: str = "session_ended"):
        """End the current session."""
        if self._session_ended:
            return

        self._cancel_session_end_task()

        logger.info(f"Ending session: {reason}")

        # Call the session end callback if provided
        if self.on_session_end:
            try:
                self.on_session_end(reason)
            except Exception as e:
                logger.error(f"Error in session end callback: {e}")

        # Leave the meeting FIRST, then mark session as ended
        await self.leave()

        # Mark session as ended AFTER leaving
        self._session_ended = True

    def setup_session_end_callback(self, callback):
        """Set up the session end callback."""
        self.on_session_end = callback
        logger.debug("Session end callback set up")

    def _schedule_session_end(self, timeout_seconds: int):
        """Schedule session end after timeout."""
        if self._session_end_task and not self._session_end_task.done():
            self._session_end_task.cancel()

        self._session_end_task = asyncio.create_task(
            self._delayed_session_end(timeout_seconds)
        )
        logger.info(f"Session end scheduled in {timeout_seconds} seconds")

    async def _delayed_session_end(self, timeout_seconds: int):
        """Delayed session end after timeout."""
        await asyncio.sleep(timeout_seconds)
        await self._end_session("no_participants")

    def on_participant_joined(self, participant: Participant):
        peer_name = participant.display_name
        self.participants_data[participant.id] = {"name": peer_name}
        logger.info(f"Participant joined: {peer_name}")

        if self.recording and len(self.participants_data) == 1:
            asyncio.create_task(
                self.start_participant_recording(participant.id))

        if participant.id in self._participant_joined_events:
            self._participant_joined_events[participant.id].set()

        if not self._first_participant_event.is_set():
            self._first_participant_event.set()

        # Update participant count and cancel session end if participants are present
        self._update_non_agent_participant_count()
        if self._non_agent_participant_count > 0:
            self._cancel_session_end_task()

        def on_stream_enabled(stream: Stream):
            if stream.kind == "audio":
                global_event_emitter.emit("AUDIO_STREAM_ENABLED", {
                                          "stream": stream, "participant": participant})
                logger.info(
                    f"Audio stream enabled for participant: {peer_name}")
                try:
                    task = asyncio.create_task(self.add_audio_listener(stream))
                    self.audio_listener_tasks[stream.id] = task
                except Exception as e:
                    logger.error(f"Error creating audio listener task: {e}")
            if stream.kind == "video" and self.vision:
                self.video_listener_tasks[stream.id] = asyncio.create_task(
                    self.add_video_listener(stream)
                )

        def on_stream_disabled(stream: Stream):
            if stream.kind == "audio":
                audio_task = self.audio_listener_tasks[stream.id]
                if audio_task is not None:
                    audio_task.cancel()
                    del self.audio_listener_tasks[stream.id]
            if stream.kind == "video":
                video_task = self.video_listener_tasks[stream.id]
                if video_task is not None:
                    video_task.cancel()
                    del self.video_listener_tasks[stream.id]

        if participant.id != self.meeting.local_participant.id:
            participant.add_event_listener(
                ParticipantHandler(
                    participant_id=participant.id,
                    on_stream_enabled=on_stream_enabled,
                    on_stream_disabled=on_stream_disabled,
                )
            )

    def on_participant_left(self, participant: Participant):
        logger.info(f"Participant left: {participant.display_name}")
        if participant.id in self.audio_listener_tasks:
            self.audio_listener_tasks[participant.id].cancel()
            del self.audio_listener_tasks[participant.id]
        if participant.id in self.video_listener_tasks:
            self.video_listener_tasks[participant.id].cancel()
            del self.video_listener_tasks[participant.id]
        if participant.id in self.participants_data:
            del self.participants_data[participant.id]
        global_event_emitter.emit(
            "PARTICIPANT_LEFT", {"participant": participant})

        # Update participant count and check if session should end
        self._update_non_agent_participant_count()
        if (
            self._non_agent_participant_count == 0
            and self.auto_end_session
            and self.session_timeout_seconds is not None
        ):
            logger.info(
                "All non-agent participants have left, scheduling session end")
            self._schedule_session_end(self.session_timeout_seconds)

    async def add_audio_listener(self, stream: Stream):
        while True:
            try:
                await asyncio.sleep(0.01)
                frame = await stream.track.recv()
                audio_data = frame.to_ndarray()[0]
                pcm_frame = audio_data.flatten().astype(np.int16).tobytes()
                if self.pipeline:
                    await self.pipeline.on_audio_delta(pcm_frame)
                else:
                    logger.warning(
                        "No pipeline available for audio processing")

            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                break

    async def add_video_listener(self, stream: Stream):
        while True:
            try:
                await asyncio.sleep(0.01)

                frame = await stream.track.recv()
                if self.pipeline:
                    await self.pipeline.on_video_delta(frame)

            except Exception as e:
                logger.error("Video processing error:", e)
                break

    async def wait_for_participant(self, participant_id: str | None = None) -> str:
        """
        Wait for a specific participant to join, or wait for the first participant if none specified.

        Args:
            participant_id: Optional participant ID to wait for. If None, waits for first participant.

        Returns:
            str: The participant ID that joined
        """
        if participant_id:
            if participant_id in self.participants_data:
                return participant_id

            if participant_id not in self._participant_joined_events:
                self._participant_joined_events[participant_id] = asyncio.Event(
                )

            await self._participant_joined_events[participant_id].wait()
            return participant_id
        else:
            if self.participants_data:
                return next(iter(self.participants_data.keys()))

            await self._first_participant_event.wait()
            return next(iter(self.participants_data.keys()))

    async def subscribe_to_pubsub(self, pubsub_config: PubSubSubscribeConfig):
        old_messages = await self.meeting.pubsub.subscribe(pubsub_config)
        return old_messages

    async def publish_to_pubsub(self, pubsub_config: PubSubPublishConfig):
        await self.meeting.pubsub.publish(pubsub_config)

    async def upload_file(self, base64_data, file_name):
        return self.meeting.upload_base64(base64_data, self.auth_token, file_name)

    async def fetch_file(self, url):
        return self.meeting.fetch_base64(url, self.auth_token)

    async def cleanup(self):
        """Add cleanup method"""

        if hasattr(self, "audio_track"):
            await self.audio_track.cleanup()

    async def _collect_session_id(self) -> None:
        """Collect session ID from room and set it in metrics cascading_metrics_collector/realtime_metrics_collector"""
        if self.meeting and not self._session_id_collected:
            try:
                session_id = getattr(self.meeting, "session_id", None)
                if session_id:
                    self._session_id = session_id
                    cascading_metrics_collector.set_session_id(session_id)
                    realtime_metrics_collector.set_session_id(session_id)
                    self._session_id_collected = True
                    if self.traces_flow_manager:
                        self.traces_flow_manager.set_session_id(session_id)
            except Exception as e:
                logger.error(f"Error collecting session ID: {e}")

    async def _collect_meeting_attributes(self) -> None:
        """
        Collect meeting attributes from room and initialize telemetry and logs.
        Also creates parent-child spans and logs after meeting is joined.
        """
        if not self.meeting:
            logger.error("Meeting not initialized")
            return

        try:
            if hasattr(self.meeting, "get_attributes"):
                attributes = self.meeting.get_attributes()

                if attributes:
                    peer_id = getattr(self.meeting, "participant_id", "agent")
                    auto_initialize_telemetry_and_logs(
                        room_id=self.meeting_id,
                        peer_id=peer_id,
                        room_attributes=attributes,
                        session_id=self._session_id,
                        sdk_metadata=self.sdk_metadata,
                    )
                else:
                    logger.error("No meeting attributes found")
            else:
                logger.error(
                    "Meeting object does not have 'get_attributes' method")

            if self._meeting_joined_data and self.traces_flow_manager:
                start_time = time.perf_counter()
                agent_joined_attributes = {
                    "roomId": self.meeting_id,
                    "sessionId": self._session_id,
                    "agent_name": self.name,
                    "peerId": self.meeting.local_participant.id,
                    "sdk_metadata": self.sdk_metadata,
                    "start_time": start_time,
                }
                self.traces_flow_manager.start_agent_joined_meeting(
                    agent_joined_attributes
                )
        except Exception as e:
            logger.error(
                f"Error collecting meeting attributes and creating spans: {e}")

    async def stop_participants_recording(self):
        await self.stop_participant_recording(self.meeting.local_participant.id)
        for participant_id in self.participants_data.keys():
            logger.info("stopping participant recording for id",
                        participant_id)
            await self.stop_participant_recording(participant_id)

    async def start_participant_recording(self, id: str):
        headers = {"Authorization": self.auth_token,
                   "Content-Type": "application/json"}
        response = requests.request(
            "POST",
            START_RECORDING_URL,
            json={"roomId": self.meeting_id, "participantId": id},
            headers=headers,
        )
        logger.info("response for id", id, response.text)

    async def stop_participant_recording(self, id: str):
        headers = {"Authorization": self.auth_token,
                   "Content-Type": "application/json"}
        response = requests.request(
            "POST",
            STOP_RECORDING_URL,
            json={"roomId": self.meeting_id, "participantId": id},
            headers=headers,
        )
        logger.info("response for id", id, response.text)

    async def merge_participant_recordings(self):
        headers = {"Authorization": self.auth_token,
                   "Content-Type": "application/json"}
        response = requests.request(
            "POST",
            MERGE_RECORDINGS_URL,
            json={
                "sessionId": self.meeting.session_id,
                "channel1": [{"participantId": self.meeting.local_participant.id}],
                "channel2": [
                    {"participantId": participant_id}
                    for participant_id in self.participants_data.keys()
                ],
            },
            headers=headers,
        )
        logger.info(response.text)

    async def stop_and_merge_recordings(self):
        await self.stop_participants_recording()
        await self.merge_participant_recordings()
        logger.info("stopped and merged recordings")
