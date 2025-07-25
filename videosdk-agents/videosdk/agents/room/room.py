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
from .audio_stream import CustomAudioStreamTrack, TeeCustomAudioStreamTrack
from videosdk.agents.pipeline import Pipeline
from dotenv import load_dotenv
import numpy as np
import asyncio
import os
from asyncio import AbstractEventLoop
from .audio_stream import TeeCustomAudioStreamTrack
from typing import Optional, Any, Callable

logger = logging.getLogger(__name__)


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
        self._has_left_meeting = False  # Track if we've already left the meeting

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
        self.pipeline = pipeline
        self.audio_listener_tasks = {}
        self.meeting = None
        self.attributes = {}
        self.participants_data = {}
        self.video_listener_tasks = {}
        self.vision = vision
        self.on_room_error = on_room_error
        self._participant_joined_events: dict[str, asyncio.Event] = {}
        self._first_participant_event = asyncio.Event()

        # Session management
        self.auto_end_session = auto_end_session
        self.session_timeout_seconds = session_timeout_seconds
        self.on_session_end = on_session_end
        self._session_end_task: Optional[asyncio.Task] = None
        self._session_ended = False
        self._non_agent_participant_count = 0

    def init_meeting(self):
        sdk_metadata = {"sdk": "agents", "sdk_version": "0.0.21"}

        # Add signaling base URL to meeting config if provided
        meeting_config = self.meeting_config.copy()
        if self.signaling_base_url:
            meeting_config["signaling_base_url"] = self.signaling_base_url

        self.meeting = VideoSDK.init_meeting(
            **meeting_config, sdk_metadata=sdk_metadata
        )
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

    def leave(self):
        # Don't try to leave if we've already left or if session has ended
        if self._has_left_meeting or self._session_ended:
            logger.debug(
                f"Already left meeting or session ended, skipping leave operation (has_left: {self._has_left_meeting}, session_ended: {self._session_ended})"
            )
            return

        logger.info("Attempting to leave the meeting...")

        # Mark that we're leaving
        self._has_left_meeting = True

        for audio_task in self.audio_listener_tasks.values():
            audio_task.cancel()
        for video_task in self.video_listener_tasks.values():
            video_task.cancel()

    def on_error(self, data):
        if self.on_room_error:
            self.on_room_error(data)

    def on_meeting_joined(self, data):
        logger.info(f"Agent joined the meeting")

    def on_meeting_left(self, data=None):
        logger.info(f"Meeting Left: {data}")
        self._cancel_session_end_task()

        # Handle job cleanup if we have a session end callback
        if hasattr(self, "on_session_end") and self.on_session_end:
            try:
                self.on_session_end("meeting_left")
            except Exception as e:
                logger.error(f"Error in session end callback during meeting left: {e}")

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
        self.leave()

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
        self.participants_data[participant.id] = {
            "name": peer_name,
        }
        logger.info(f"Participant joined: {peer_name}")

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
                logger.info(f"Audio stream enabled for participant: {peer_name}")
                try:
                    task = asyncio.create_task(self.add_audio_listener(stream))
                    self.audio_listener_tasks[stream.id] = task
                except Exception as e:
                    logger.error(f"Error creating audio listener task: {e}")
            if stream.kind == "video" and self.vision:
                self.video_listener_tasks[stream.id] = self.loop.create_task(
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

        # Update participant count and check if session should end
        self._update_non_agent_participant_count()

        if (
            self._non_agent_participant_count == 0
            and self.auto_end_session
            and self.session_timeout_seconds is not None
        ):
            logger.info("All non-agent participants have left, scheduling session end")
            self._schedule_session_end(self.session_timeout_seconds)

    async def add_audio_listener(self, stream: Stream):
        try:
            # Get the participant ID for this stream to check if it's the agent's own stream
            stream_participant_id = None
            for participant in self.meeting.participants.values():
                for participant_stream in participant.streams.values():
                    if participant_stream.id == stream.id:
                        stream_participant_id = participant.id
                        break
                if stream_participant_id:
                    break

            # Check if this is the agent's own audio stream
            is_agent_stream = False
            if stream_participant_id:
                if self.meeting and self.meeting.local_participant:
                    is_agent_stream = (
                        stream_participant_id == self.meeting.local_participant.id
                    )

            # Skip processing if this is the agent's own stream
            if is_agent_stream:
                return

            while True:
                try:
                    await asyncio.sleep(0.01)
                    frame = await stream.track.recv()
                    audio_data = frame.to_ndarray()[0]
                    pcm_frame = audio_data.flatten().astype(np.int16).tobytes()
                    if self.pipeline:
                        await self.pipeline.on_audio_delta(pcm_frame)
                    else:
                        logger.warning("No pipeline available for audio processing")

                except Exception as e:
                    logger.error(f"Audio processing error: {e}")
                    break

        except Exception as e:
            logger.error(f"Error in add_audio_listener: {e}")

    async def add_video_listener(self, stream: Stream):
        while True:
            try:
                await asyncio.sleep(0.01)

                frame = await stream.track.recv()
                if self.pipeline:
                    await self.pipeline.on_video_delta(frame)

            except Exception as e:
                logger.error(f"Video processing error: {e}")
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
                self._participant_joined_events[participant_id] = asyncio.Event()

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
