from __future__ import annotations
import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any
import time
import fractions
import numpy as np

from videosdk.meeting import Meeting
from .avatar_schema import AvatarInput, AudioSegmentEnd, AvatarRenderer
from .avatar_sync import AvatarSync
from videosdk.custom_audio_track import CustomAudioTrack
from videosdk.custom_video_track import CustomVideoTrack
from av import AudioFrame, VideoFrame
from vsaiortc.mediastreams import MediaStreamError

logger = logging.getLogger(__name__)


@dataclass
class AvatarSettings:
    video_width: int
    video_height: int
    video_fps: float
    audio_sample_rate: int
    audio_channels: int


class AvatarVoiceTrack(CustomAudioTrack):
    """Custom audio track that rebuilds steady 20 ms PCM frames from incoming audio."""

    AUDIO_PTIME = 0.02 
    MAX_BUFFER_DURATION = 2.0

    def __init__(self, sample_rate: int, num_channels: int):
        super().__init__()
        self.kind = "audio"
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._start: float | None = None
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, self._sample_rate)
        self._default_samples = int(self._sample_rate * self.AUDIO_PTIME)
        self._sample_width = 2  # s16
        self._chunk_size = self._default_samples * self._num_channels * self._sample_width
        self._audio_buffer = bytearray()
        max_chunks = int(self.MAX_BUFFER_DURATION / self.AUDIO_PTIME)
        self._max_buffer_bytes = max(self._chunk_size * max_chunks, self._chunk_size * 5)
        self._stopped = False
        self._is_speaking = False

    async def put_frame(self, frame: AudioFrame) -> None:
        """Convert incoming frame to bytes and append to jitter buffer."""
        if self._stopped:
            return

        try:
            pcm_bytes = bytes(frame.planes[0])
        except Exception:
            pcm_bytes = frame.to_ndarray().tobytes()

        if not pcm_bytes:
            return

        self._audio_buffer.extend(pcm_bytes)
        if len(self._audio_buffer) > self._max_buffer_bytes:
            overflow = len(self._audio_buffer) - self._max_buffer_bytes
            del self._audio_buffer[:overflow]

    def _build_audio_frame(self, chunk: bytes) -> AudioFrame:
        if len(chunk) < self._chunk_size:
            chunk = chunk + bytes(self._chunk_size - len(chunk))

        data = np.frombuffer(chunk, dtype=np.int16).reshape(
            -1, self._num_channels
        )
        layout = "mono" if self._num_channels == 1 else "stereo"
        audio_frame = AudioFrame.from_ndarray(data.T, format="s16", layout=layout)
        audio_frame.sample_rate = self._sample_rate
        return audio_frame

    def _build_silence_frame(self) -> AudioFrame:
        frame = AudioFrame(
            format="s16",
            layout="mono" if self._num_channels == 1 else "stereo",
            samples=self._default_samples,
        )
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        frame.sample_rate = self._sample_rate
        return frame

    async def recv(self) -> AudioFrame:
        """Receive the next audio frame with tight pacing."""
        if self.readyState != "live":
            raise MediaStreamError

        if self._start is None:
            self._start = time.time()

        if len(self._audio_buffer) >= self._chunk_size:
            chunk = self._audio_buffer[: self._chunk_size]
            del self._audio_buffer[: self._chunk_size]
            frame = self._build_audio_frame(chunk)
            self._is_speaking = True
        else:
            if self._stopped:
                raise MediaStreamError("Track ended")
            frame = self._build_silence_frame()
            if self._is_speaking:
                self._is_speaking = False

        samples = frame.samples or self._default_samples
        pts = self._timestamp
        wait = self._start + (pts / self._sample_rate) - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        frame.pts = pts
        frame.time_base = self._time_base
        self._timestamp += samples
        return frame

    async def stop(self):
        """Stop the track"""
        self._stopped = True
        self._audio_buffer.clear()


class AvatarVisualTrack(CustomVideoTrack):
    """Custom video track for avatar video output"""
    
    def __init__(self, width: int, height: int, fps: float):
        super().__init__()
        self.kind = "video"
        self._width = width
        self._height = height
        self._fps = fps
        self._queue: asyncio.Queue[VideoFrame | None] = asyncio.Queue()
        self._start = None
        self._timestamp = 0
    
    async def put_frame(self, frame: VideoFrame) -> None:
        """Put a video frame into the track"""
        await self._queue.put(frame)
    
    async def recv(self) -> VideoFrame:
        """Receive the next video frame"""
        if self.readyState != "live":
            raise MediaStreamError
        
        if self._start is None:
            self._start = time.time()
        
        frame = await self._queue.get()
        if frame is None:
            raise MediaStreamError("Track ended")
        
        VIDEO_CLOCK_RATE = 90000
        VIDEO_PTIME = 1 / self._fps
        self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, VIDEO_CLOCK_RATE)
        
        return frame
    
    async def stop(self):
        """Stop the track"""
        await self._queue.put(None)


class AvatarController:
    def __init__(
        self,
        meeting: Meeting | None, 
        *,
        audio_recv: AvatarInput,
        video_gen: AvatarRenderer,
        options: AvatarSettings,
    ) -> None:
        self._meeting = meeting
        self._video_gen = video_gen
        self._options = options

        self._audio_recv = audio_recv
        self._playback_position = 0.0
        self._audio_playing = False
        self._tasks: set[asyncio.Task[Any]] = set()

        self._audio_track = AvatarVoiceTrack(
            sample_rate=options.audio_sample_rate,
            num_channels=options.audio_channels,
        )
        self._video_track = AvatarVisualTrack(
            width=options.video_width,
            height=options.video_height,
            fps=options.video_fps,
        )
        self._av_sync = AvatarSync(
            audio_track=self._audio_track,
            video_track=self._video_track,
            video_fps=options.video_fps,
        )
        self._read_audio_atask: asyncio.Task[None] | None = None
        self._forward_video_atask: asyncio.Task[None] | None = None

    @property
    def av_sync(self) -> AvatarSync:
        return self._av_sync

    async def avatar_run(self) -> None:
        await self._audio_recv.start_stream()
        self._audio_recv.on("reset_stream", self._on_reset_stream)

        self._read_audio_atask = asyncio.create_task(self._read_audio())
        self._forward_video_atask = asyncio.create_task(self._forward_video())

    async def wait_for_complete(self) -> None:
        if not self._read_audio_atask or not self._forward_video_atask:
            raise RuntimeError("AvatarController not started")

        await asyncio.gather(
            self._read_audio_atask,
            self._forward_video_atask,
        )

    async def _read_audio(self) -> None:
        async for frame in self._audio_recv:
            if not self._audio_playing and isinstance(frame, AudioFrame):
                self._audio_playing = True
            await self._video_gen.push_stream_chunk(frame)

    async def _forward_video(self) -> None:
        async for frame in self._video_gen:
            if isinstance(frame, AudioSegmentEnd):
                if self._audio_playing:
                    notify_task = self._audio_recv.notify_stream_ended(
                        playback_position=self._playback_position,
                        interrupted=False,
                    )
                    self._audio_playing = False
                    self._playback_position = 0.0
                    if asyncio.iscoroutine(notify_task):
                        task = asyncio.create_task(notify_task)
                        self._tasks.add(task)
                        task.add_done_callback(self._tasks.discard)
                continue

            await self._av_sync.push(frame)
            if isinstance(frame, AudioFrame):
                self._playback_position += frame.samples / frame.sample_rate

    def _on_reset_stream(self) -> None:
        async def _handle_reset_stream(audio_playing: bool) -> None:
            clear_task = self._video_gen.reset_stream()
            if asyncio.iscoroutine(clear_task):
                await clear_task

            if audio_playing:
                notify_task = self._audio_recv.notify_stream_ended(
                    playback_position=self._playback_position,
                    interrupted=True,
                )
                self._playback_position = 0.0
                if asyncio.iscoroutine(notify_task):
                    await notify_task

        task = asyncio.create_task(_handle_reset_stream(self._audio_playing))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self._audio_playing = False

    async def aclose(self) -> None:
        await self._audio_recv.aclose()
        if self._forward_video_atask:
            self._forward_video_atask.cancel()
        if self._read_audio_atask:
            self._read_audio_atask.cancel()
        
        for task in self._tasks:
            task.cancel()

        await self._av_sync.aclose()
