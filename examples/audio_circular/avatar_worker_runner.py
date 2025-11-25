import asyncio
import logging
import sys
import time
import json
import argparse
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import Optional, Union
import numpy as np

from videosdk import VideoSDK, MeetingConfig, MeetingEventHandler
from videosdk.participant import Participant

from videosdk.agents import utils
from videosdk.agents.avatar import (
    AudioSegmentEnd,
    AvatarSettings,
    AvatarController,
    AvatarDataChannelReceiver,
    AvatarRenderer,
)

from av import AudioFrame, VideoFrame

sys.path.insert(0, str(Path(__file__).parent))
from circular_glow_viz import CircularGlowVisualizer

logger = logging.getLogger("avatar-example")

class AvatarVisualizer(AvatarRenderer):
    def __init__(self, options: AvatarSettings):
        self._options = options
        self._audio_queue = asyncio.Queue[Union[AudioFrame, AudioSegmentEnd]]()

        self._canvas = np.zeros((options.video_height, options.video_width, 4), dtype=np.uint8)
        self._canvas.fill(255)
        self._circular_glow_visualizer = CircularGlowVisualizer(sample_rate=options.audio_sample_rate)

        self._audio_bstream = utils.audio.AudioByteStream(
            sample_rate=options.audio_sample_rate,
            num_channels=options.audio_channels,
            samples_per_channel=options.audio_sample_rate // options.video_fps,
        )
        self._frame_ts: deque[float] = deque(maxlen=options.video_fps)

    async def push_stream_chunk(self, frame: AudioFrame | AudioSegmentEnd) -> None:
        """Called by the runner to push audio frames to the generator."""
        await self._audio_queue.put(frame)

    def reset_stream(self) -> None:
        """Called by the runner to clear the audio buffer (interruptions)"""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._audio_bstream.flush()

    def __aiter__(
        self,
    ) -> AsyncIterator[VideoFrame | AudioFrame | AudioSegmentEnd]:
        return self._video_generation_impl()

    async def _video_generation_impl(
        self,
    ) -> AsyncGenerator[VideoFrame | AudioFrame | AudioSegmentEnd, None]:
        while True:
            try:
                frame = await asyncio.wait_for(
                    self._audio_queue.get(), timeout=0.5 / self._options.video_fps
                )
                self._audio_queue.task_done()
            except asyncio.TimeoutError:
                yield self._generate_frame(None)
                self._frame_ts.append(time.time())
                continue

            audio_frames: list[AudioFrame] = []
            
            if isinstance(frame, AudioFrame):
                audio_frames += self._audio_bstream.push(bytes(frame.planes[0]))
            else:
                audio_frames += self._audio_bstream.flush()

            for audio_frame in audio_frames:
                video_frame = self._generate_frame(audio_frame)
                yield video_frame
                yield audio_frame
                self._frame_ts.append(time.time())

            if isinstance(frame, AudioSegmentEnd):
                yield AudioSegmentEnd()

    def _generate_frame(self, audio_frame: AudioFrame | None) -> VideoFrame:
        canvas = self._canvas.copy()

        if audio_frame is None:
            audio_data = np.zeros((1, self._options.audio_channels))
        else:
            audio_data = np.frombuffer(bytes(audio_frame.planes[0]), dtype=np.int16).reshape(
                -1, audio_frame.layout.nb_channels
            )

        self._circular_glow_visualizer.draw(canvas, audio_samples=audio_data, fps=self._get_fps())
        
        video_frame = VideoFrame.from_ndarray(canvas, format="rgba")
        return video_frame

    def _get_fps(self) -> float | None:
        if len(self._frame_ts) < 2:
            return None
        return (len(self._frame_ts) - 1) / (self._frame_ts[-1] - self._frame_ts[0])

async def run_avatar_worker(token: str, room_id: str, name: str, participant_id: str):
    """
    Main function to run the Avatar Worker.
    """
    
    avatar_options = AvatarSettings(
        video_width=1280,
        video_height=720,
        video_fps=30,
        audio_sample_rate=24000,
        audio_channels=1,
    )
    
    video_gen = AvatarVisualizer(avatar_options)
    audio_recv = AvatarDataChannelReceiver(
        meeting=None, 
        channels=avatar_options.audio_channels
    )

    runner = AvatarController(
        meeting=None, 
        audio_recv=audio_recv, 
        video_gen=video_gen, 
        options=avatar_options
    )
    
    meeting_config = MeetingConfig(
        meeting_id=room_id,
        token=token,
        name=name,
        mic_enabled=True,
        webcam_enabled=True,
        custom_camera_video_track=runner._video_track,
        custom_microphone_audio_track=runner._audio_track,
        participant_id=participant_id,
    )
    
    meeting = VideoSDK.init_meeting(**meeting_config)
    
    runner._meeting = meeting
    audio_recv.set_meeting(meeting)
    
    should_stop = asyncio.Event()

    class AvatarMeetingEventHandler(MeetingEventHandler):
        def __init__(self):
            super().__init__()
        
        def on_participant_left(self, participant: Participant):
            pass
        
        def on_meeting_left(self, data):
            logging.info("Room disconnected, stopping worker")
            should_stop.set()
    
    meeting.add_event_listener(AvatarMeetingEventHandler())
    
    logger.info(f"Joining meeting {room_id} as {name}...")
    await meeting.async_join()
    logger.info("Joined successfully.")

    tasks = []
    try:
        await runner.avatar_run()        
        tasks = [
            asyncio.create_task(runner.wait_for_complete()),
            asyncio.create_task(should_stop.wait()),
        ]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except Exception as e:
        logger.error(f"Runtime error: {e}")
    finally:
        for task in tasks:
            await utils.cancel_and_wait(task)
        
        await runner.aclose()
        meeting.leave() 
        logger.info("Avatar runner stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoSDK Avatar Agent")
    
    parser.add_argument("--token", type=str, required=True, help="Participant Token")
    parser.add_argument("--room-id", type=str, required=True, help="Meeting ID")
    parser.add_argument("--name", type=str, default="AI Avatar", help="Display Name")
    parser.add_argument("--participant-id", type=str, default="avatar_worker", help="Participant ID")
    
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    asyncio.run(run_avatar_worker(
        args.token, 
        args.room_id, 
        args.name, 
        args.participant_id,
    ))