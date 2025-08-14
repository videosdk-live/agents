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
from opentelemetry.trace import StatusCode, Span
from ..metrics.integration import create_span, complete_span, create_log
from ..metrics.traces_flow import TracesFlowManager
from ..metrics import cascading_metrics_collector
from ..metrics.integration import auto_initialize_telemetry_and_logs
from typing import Callable, Optional, Any
from ..metrics.realtime_metrics_collector import realtime_metrics_collector
import requests
import time
import logging

logger = logging.getLogger(__name__)


START_RECORDING_URL = "https://api.videosdk.live/v2/recordings/participant/start"
STOP_RECORDING_URL = "https://api.videosdk.live/v2/recordings/participant/stop"
MERGE_RECORDINGS_URL = "https://api.videosdk.live/v2/recordings/participant/merge"
 
load_dotenv()

class VideoSDKHandler:
    def __init__(self, *, meeting_id: str, auth_token: str | None = None, name: str, pipeline: Pipeline, loop: AbstractEventLoop, vision: bool = False, recording: bool = False, custom_camera_video_track=None, 
        custom_microphone_audio_track=None,audio_sinks=None, on_room_error: Optional[Callable[[Any], None]] = None):
        self.loop = loop
        self.meeting_id = meeting_id
        self.name = name
        self._meeting_joined_data = None
        self.agent_meeting = None
        self._session_id: Optional[str] = None
        self._session_id_collected = False
        self.recording = recording
        
        self.traces_flow_manager = TracesFlowManager(room_id=self.meeting_id)
        cascading_metrics_collector.set_traces_flow_manager(self.traces_flow_manager)

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
        self._left: bool = False
        
    def init_meeting(self):
        self.sdk_metadata = {
            "sdk" : "agents",
            "sdk_version" : "0.0.24" 
        }
        
        self.meeting = VideoSDK.init_meeting(**self.meeting_config, sdk_metadata=self.sdk_metadata)
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
        self._meeting_joined_data = data
        self.loop.create_task(self._collect_session_id())
        self.loop.create_task(self._collect_meeting_attributes())
        if self.recording:
            self.loop.create_task(self.start_participants_recording())

    def on_meeting_left(self, data):
        logger.info(f"Meeting Left", data)
        
    def on_participant_joined(self, participant: Participant):
        peer_name = participant.display_name
        self.participants_data[participant.id] = {
            "name": peer_name,
        }
        logger.info("Participant joined:", peer_name)
        
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
                logger.error("Audio processing error:", e)
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

    async def _collect_session_id(self) -> None:
        """Collect session ID from room and set it in metrics cascading_metrics_collector/realtime_metrics_collector"""
        if self.meeting and not self._session_id_collected:
            try:
                session_id = getattr(self.meeting, 'session_id', None)
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
            if hasattr(self.meeting, 'get_attributes'):
                attributes = self.meeting.get_attributes()

                if attributes:
                    peer_id = getattr(self.meeting, 'participant_id', 'agent')
                    auto_initialize_telemetry_and_logs(
                        room_id=self.meeting_id,
                        peer_id=peer_id,
                        room_attributes=attributes,
                        session_id=self._session_id,
                        sdk_metadata=self.sdk_metadata
                    )
                else:
                    logger.error("No meeting attributes found")
            else:
                logger.error("Meeting object does not have 'get_attributes' method")

            if self._meeting_joined_data and self.traces_flow_manager:
                start_time = time.perf_counter() 
                agent_joined_attributes = {
                    "roomId": self.meeting_id,
                    "sessionId": self._session_id,
                    "agent_name": self.name,
                    "peerId": self.meeting.local_participant.id,
                    "sdk_metadata": self.sdk_metadata,
                    "start_time": start_time
                }   
                self.traces_flow_manager.start_agent_joined_meeting(agent_joined_attributes)
        except Exception as e:
            logger.error(f"Error collecting meeting attributes and creating spans: {e}")

    async def start_participants_recording(self) :
        await self.start_participant_recording(self.meeting.local_participant.id)
        for participant in self.meeting.participants.values():
            await self.start_participant_recording(participant.id)

    async def stop_participants_recording(self):
        await self.stop_participant_recording(self.meeting.local_participant.id)
        for participant_id in self.participants_data.keys():
            logger.info("stopping participant recording for id", participant_id)
            await self.stop_participant_recording(participant_id)
             
    async def start_participant_recording(self, id: str):
        headers = {'Authorization' : self.auth_token,'Content-Type' : 'application/json'}
        response = requests.request("POST", START_RECORDING_URL,json = {
		"roomId" : self.meeting_id,
		"participantId" : id
	    },headers = headers)
        logger.info("response for id", id, response.text)
    
    async def stop_participant_recording(self, id: str):
        headers = {'Authorization' : self.auth_token,'Content-Type' : 'application/json'}
        response = requests.request("POST", STOP_RECORDING_URL,json = {
		"roomId" : self.meeting_id,
		"participantId" : id
	    },headers = headers)
        logger.info("response for id", id, response.text)
        
    async def merge_participant_recordings(self):
        headers = {'Authorization' : self.auth_token,'Content-Type' : 'application/json'}
        response = requests.request("POST", MERGE_RECORDINGS_URL,json = {
		"sessionId" : self.meeting.session_id,
		"channel1" : [{"participantId":self.meeting.local_participant.id}],
		"channel2" : [{"participantId":participant_id} for participant_id in self.participants_data.keys()],
	},headers = headers)
        logger.info(response.text)      

    async def stop_and_merge_recordings(self):
        await self.stop_participants_recording()
        await self.merge_participant_recordings() 
        logger.info("stopped and merged recordings")
                 