import asyncio
import time
from typing import Union

from av import AudioFrame, VideoFrame
from videosdk.custom_audio_track import CustomAudioTrack
from videosdk.custom_video_track import CustomVideoTrack
from .avatar_schema import AudioSegmentEnd


class AvatarSync:
    def __init__(
        self,
        audio_track: CustomAudioTrack,
        video_track: CustomVideoTrack,
        video_fps: float,
    ):
        self._audio_track = audio_track
        self._video_track = video_track
        self._video_fps = video_fps
        self._frame_interval = 1.0 / video_fps
        self._last_frame_time = 0
        self._start_time = 0

    async def push(self, frame: Union[AudioFrame, VideoFrame, AudioSegmentEnd]):
        if not self._start_time:
            self._start_time = time.monotonic()
        
        if isinstance(frame, AudioFrame):
            await self._audio_track.put_frame(frame)
        elif isinstance(frame, VideoFrame):
            now = time.monotonic()
            elapsed = now - self._last_frame_time
            if elapsed < self._frame_interval:
                await asyncio.sleep(self._frame_interval - elapsed)
            
            await self._video_track.put_frame(frame)
            self._last_frame_time = time.monotonic()

    async def aclose(self):
        pass
