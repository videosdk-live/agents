from videosdk import MeetingConfig, VideoSDK, Participant, Stream
from .meeting_event_handler import MeetingHandler
from .participant_event_handler import ParticipantHandler
from .audio_stream import CustomAudioStreamTrack
from videosdk.agents.pipeline import Pipeline
from dotenv import load_dotenv
import numpy as np
import librosa
import asyncio
import os
from asyncio import AbstractEventLoop
import sys

load_dotenv()

class VideoSDKHandler:
    def __init__(self, *, meeting_id: str, auth_token: str | None = None, name: str, pipeline: Pipeline, loop: AbstractEventLoop):
        self.loop = loop
        self.audio_track = CustomAudioStreamTrack(
            loop=self.loop
        )
        auth_token = auth_token or os.getenv("VIDEOSDK_AUTH_TOKEN")
        if not auth_token:
            raise ValueError("VIDEOSDK_AUTH_TOKEN is not set")
        self.meeting_config = MeetingConfig(
            name=name,
            meeting_id=meeting_id,
            token=auth_token,
            mic_enabled=True,
            webcam_enabled=False,
            custom_microphone_audio_track=self.audio_track,
        )
        self.pipeline = pipeline
        self.audio_listener_tasks = {}
        self.meeting = None

        self.participants_data = {}

    def init_meeting(self):
        self.meeting = VideoSDK.init_meeting(**self.meeting_config)
        self.meeting.add_event_listener(
            MeetingHandler(
                on_meeting_joined=self.on_meeting_joined,
                on_meeting_left=self.on_meeting_left,
                on_participant_joined=self.on_participant_joined,
                on_participant_left=self.on_participant_left,
            )
        )

    async def join(self):
        await self.meeting.async_join()

    def leave(self):
        for audio_task in self.audio_listener_tasks.values():
            audio_task.cancel()
        self.meeting.leave()

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

        def on_stream_enabled(stream: Stream):
            print("Participant stream enabled")
            if stream.kind == "audio":
                self.audio_listener_tasks[stream.id] = self.loop.create_task(
                    self.add_audio_listener(stream)
                )

        def on_stream_disabled(stream: Stream):
            print("Participant stream disabled")
            if stream.kind == "audio":
                audio_task = self.audio_listener_tasks[stream.id]
                if audio_task is not None:
                    audio_task.cancel()

        participant.add_event_listener(
            ParticipantHandler(
                participant_id=participant.id,
                on_stream_enabled=on_stream_enabled,
                on_stream_disabled=on_stream_disabled,
            )
        )

    def on_participant_left(self, participant: Participant):
        print("Participant left:", participant.display_name)
        for audio_task in self.audio_listener_tasks.values():
            audio_task.cancel()
        self.leave()
        # sys.exit(0)
        

    async def add_audio_listener(self, stream: Stream):
        while True:
            try:
                await asyncio.sleep(0.01)

                frame = await stream.track.recv()
                audio_data = frame.to_ndarray()[0]
                audio_data_float = (
                    audio_data.astype(np.float32) / np.iinfo(np.int16).max
                )
                audio_mono = librosa.to_mono(audio_data_float.T)
                audio_resampled = librosa.resample(
                    audio_mono, orig_sr=48000, target_sr=16000
                )
                pcm_frame = (
                    (audio_resampled * np.iinfo(np.int16).max)
                    .astype(np.int16)
                    .tobytes()
                )
                await self.pipeline.on_audio_delta(pcm_frame)

            except Exception as e:
                print("Audio processing error:", e)
                break
            
    async def cleanup(self):
        """Add cleanup method"""
        
        if hasattr(self, "audio_track"):
            await self.audio_track.cleanup()