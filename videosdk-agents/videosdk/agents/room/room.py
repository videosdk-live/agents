from videosdk import MeetingConfig, VideoSDK, Participant, Stream, PubSubPublishConfig, PubSubSubscribeConfig
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


load_dotenv()

class VideoSDKHandler:
    def __init__(self, *, meeting_id: str, auth_token: str | None = None, name: str, pipeline: Pipeline, loop: AbstractEventLoop, vision: bool = False,custom_camera_video_track=None, 
        custom_microphone_audio_track=None,audio_sinks=None, on_room_error: Optional[Callable[[Any], None]] = None):
        self.loop = loop
        self.meeting_id = meeting_id
        self.name = name

        if custom_microphone_audio_track:
            self.audio_track = custom_microphone_audio_track
            if audio_sinks:
                self.agent_audio_track = TeeCustomAudioStreamTrack(
                    loop=self.loop,
                    sinks=audio_sinks
                )
            else:
                self.agent_audio_track = None
        else:
            self.audio_track = TeeCustomAudioStreamTrack(
                loop=self.loop,
                sinks=audio_sinks
            )
            self.agent_audio_track = None

        self.auth_token = auth_token or os.getenv("VIDEOSDK_AUTH_TOKEN")
        if not self.auth_token:
            raise ValueError("VIDEOSDK_AUTH_TOKEN is not set")
        self.meeting_config = MeetingConfig(
            name=self.name,
            meeting_id=self.meeting_id,
            token=self.auth_token,
            mic_enabled=True,
            webcam_enabled=custom_camera_video_track is not None,
            custom_microphone_audio_track=self.audio_track,
            custom_camera_video_track=custom_camera_video_track
        )
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
        
    def init_meeting(self):
        sdk_metadata = {
            "sdk" : "agents",
            "sdk_version" : "0.0.21" 
        }
        self.meeting = VideoSDK.init_meeting(**self.meeting_config, sdk_metadata=sdk_metadata)
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
        for audio_task in self.audio_listener_tasks.values():
            audio_task.cancel()
        for video_task in self.video_listener_tasks.values():
            video_task.cancel()
        self.meeting.leave()

    def on_error(self, data):
        if self.on_room_error:
            self.on_room_error(data)

    def on_meeting_joined(self, data):
        print(f"Agent joined the meeting")

    def on_meeting_left(self, data):
        print(f"Meeting Left", data)
        
    def on_participant_joined(self, participant: Participant):
        peer_name = participant.display_name
        self.participants_data[participant.id] = {
            "name": peer_name,
        }
        print("Participant joined:", peer_name)
        
        if participant.id in self._participant_joined_events:
            self._participant_joined_events[participant.id].set()
        
        if not self._first_participant_event.is_set():
            self._first_participant_event.set()

        def on_stream_enabled(stream: Stream):
            if stream.kind == "audio":
                self.audio_listener_tasks[stream.id] = self.loop.create_task(
                    self.add_audio_listener(stream)
                )
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

        participant.add_event_listener(
            ParticipantHandler(
                participant_id=participant.id,
                on_stream_enabled=on_stream_enabled,
                on_stream_disabled=on_stream_disabled,
            )
        )

    def on_participant_left(self, participant: Participant):
        print("Participant left:", participant.display_name)
        if participant.id in self.audio_listener_tasks:
            self.audio_listener_tasks[participant.id].cancel()
            del self.audio_listener_tasks[participant.id]
        if participant.id in self.video_listener_tasks:
            self.video_listener_tasks[participant.id].cancel()
            del self.video_listener_tasks[participant.id]
        if participant.id in self.participants_data:
            del self.participants_data[participant.id]

    async def add_audio_listener(self, stream: Stream):    
        while True:
            try:
                await asyncio.sleep(0.01)
                frame = await stream.track.recv()
                audio_data = frame.to_ndarray()[0]
                pcm_frame = audio_data.flatten().astype(np.int16).tobytes()
                if self.pipeline:
                    await self.pipeline.on_audio_delta(pcm_frame)

            except Exception as e:
                print("Audio processing error:", e)
                break    

    async def add_video_listener(self, stream: Stream):          
        while True:
            try:
                await asyncio.sleep(0.01)

                frame = await stream.track.recv()
                if self.pipeline:
                    await self.pipeline.on_video_delta(frame)
               
            except Exception as e:
                print("Audio processing error:", e)
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