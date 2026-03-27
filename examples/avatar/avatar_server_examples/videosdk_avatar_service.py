"""
VideoSDK Avatar Service

This process is launched by the avatar dispatcher (videosdk_avatar_launcher.py)
for each meeting room.  It:

  1. Joins the VideoSDK meeting using a pre-signed token.
  2. Receives raw PCM audio from the agent over the data channel.
  3. Renders each frame through a waterfall spectrogram visualizer.
  4. Publishes the rendered audio + video via custom WebRTC tracks.

Usage (normally called by the launcher, not directly):
    python videosdk_avatar_service.py \
        --token  <jwt>          \
        --room-id <meeting_id>  \
        --name   "AI Avatar"    \
        --participant-id avatar_xxxx
"""

import argparse
import asyncio
import logging
import sys
import time
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import Union

import numpy as np
from av import AudioFrame, VideoFrame
from videosdk import MeetingEventHandler, MeetingConfig, VideoSDK
from videosdk.participant import Participant

from videosdk.agents import utils
from videosdk.agents.avatar import (
    AudioSegmentEnd,
    AvatarServer,
    AvatarAudioIn,
    AvatarRenderer,
    AvatarSettings,
)

# waterfall_viz.py lives in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from waterfall_viz import WaterfallVisualizer  # noqa: E402

logger = logging.getLogger("videosdk-avatar-service")

class AvatarVisualizer(AvatarRenderer):
    """
    Implements AvatarRenderer using the WaterfallVisualizer.

    For each sync-group of audio frames (one per video frame at `video_fps`),
    it draws the current FFT spectrum as a scrolling waterfall and yields both
    the video frame and the audio frame so AvatarSynchronizer can pace them correctly.
    """

    def __init__(self, options: AvatarSettings) -> None:
        self._options = options
        self._audio_queue: asyncio.Queue[Union[AudioFrame, AudioSegmentEnd]] = asyncio.Queue()

        self._canvas = np.zeros(
            (options.video_height, options.video_width, 4), dtype=np.uint8
        )
        self._canvas.fill(255)
        self._glow = WaterfallVisualizer(sample_rate=options.audio_sample_rate)

        self._audio_bstream = utils.audio.AudioByteStream(
            sample_rate=options.audio_sample_rate,
            num_channels=options.audio_channels,
            samples_per_channel=int(options.audio_sample_rate / options.video_fps),
        )
        self._frame_ts: deque[float] = deque(maxlen=int(options.video_fps))

    async def push_stream_chunk(self, frame: AudioFrame | AudioSegmentEnd) -> None:
        await self._audio_queue.put(frame)

    def reset_stream(self) -> None:
        """Drain the audio queue on interrupt (called synchronously by AvatarServer)."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._audio_bstream.flush()

    def __aiter__(self) -> AsyncIterator[VideoFrame | AudioFrame | AudioSegmentEnd]:
        return self._generate()

    async def _generate(
        self,
    ) -> AsyncGenerator[VideoFrame | AudioFrame | AudioSegmentEnd, None]:
        timeout = 0.5 / self._options.video_fps
        while True:
            try:
                frame = await asyncio.wait_for(self._audio_queue.get(), timeout=timeout)
                self._audio_queue.task_done()
            except asyncio.TimeoutError:
                yield self._render(None)
                self._frame_ts.append(time.time())
                continue

            if isinstance(frame, AudioFrame):
                audio_frames = self._audio_bstream.push(bytes(frame.planes[0]))
            else:
                audio_frames = self._audio_bstream.flush()

            for af in audio_frames:
                yield self._render(af)
                yield af
                self._frame_ts.append(time.time())

            if isinstance(frame, AudioSegmentEnd):
                yield AudioSegmentEnd()

    def _render(self, audio_frame: AudioFrame | None) -> VideoFrame:
        canvas = self._canvas.copy()

        if audio_frame is None:
            samples = np.zeros((1, self._options.audio_channels), dtype=np.int16)
        else:
            samples = np.frombuffer(
                bytes(audio_frame.planes[0]), dtype=np.int16
            ).reshape(-1, audio_frame.layout.nb_channels)

        fps = self._current_fps()
        self._glow.draw(canvas, audio_samples=samples, fps=fps)
        return VideoFrame.from_ndarray(canvas, format="rgba")

    def _current_fps(self) -> float | None:
        if len(self._frame_ts) < 2:
            return None
        return (len(self._frame_ts) - 1) / (self._frame_ts[-1] - self._frame_ts[0])

async def run_avatar_worker(
    token: str,
    room_id: str,
    name: str,
    participant_id: str,
) -> None:
    options = AvatarSettings(
        video_width=1280,
        video_height=720,
        video_fps=30,
        audio_sample_rate=24000,
        audio_channels=1,
    )

    renderer = AvatarVisualizer(options)
    audio_recv = AvatarAudioIn(
        meeting=None,
        channels=options.audio_channels,
        sample_rate=options.audio_sample_rate,
    )
    server = AvatarServer(
        meeting=None,
        audio_recv=audio_recv,
        video_gen=renderer,
        options=options,
    )

    meeting_config = MeetingConfig(
        meeting_id=room_id,
        token=token,
        name=name,
        mic_enabled=True,
        webcam_enabled=True,
        custom_camera_video_track=server._video_track,
        custom_microphone_audio_track=server._audio_track,
        participant_id=participant_id,
    )

    meeting = VideoSDK.init_meeting(**meeting_config)

    server._meeting = meeting
    audio_recv.set_meeting(meeting)

    should_stop = asyncio.Event()

    class _Handler(MeetingEventHandler):
        def on_meeting_left(self, data):  # noqa: N802
            logger.info("Meeting left — stopping worker")
            should_stop.set()

        def on_participant_left(self, participant: Participant):  # noqa: N802
            pass

    meeting.add_event_listener(_Handler())

    logger.info("Joining meeting %s as '%s' (id=%s)…", room_id, name, participant_id)
    await meeting.async_join()
    logger.info("Joined successfully.")

    tasks: list[asyncio.Task] = []
    try:
        await server.start()
        tasks = [
            asyncio.create_task(server.wait_for_complete()),
            asyncio.create_task(should_stop.wait()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except Exception as exc:
        logger.error("Runtime error: %s", exc)
    finally:
        for t in tasks:
            await utils.cancel_and_wait(t)
        await server.aclose()
        meeting.leave()
        logger.info("Avatar Server stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoSDK Avatar Server")
    parser.add_argument("--token", required=True, help="Pre-signed VideoSDK JWT")
    parser.add_argument("--room-id", required=True, help="Meeting ID")
    parser.add_argument("--name", default="AI Avatar", help="Display name")
    parser.add_argument("--participant-id", default="avatar_worker", help="Participant ID")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    asyncio.run(
        run_avatar_worker(
            token=args.token,
            room_id=args.room_id,
            name=args.name,
            participant_id=args.participant_id,
        )
    )
