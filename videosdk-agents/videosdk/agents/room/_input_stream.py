import asyncio
import logging
import numpy as np
from videosdk import Stream
from ..event_bus import global_event_emitter
from typing import Optional

logger = logging.getLogger(__name__)

class InputStreamManager:
    """
    Manages incoming audio and video streams from participants.
    """
    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        self.audio_listener_tasks = {}
        self.video_listener_tasks = {}

    async def add_audio_listener(self, stream: Stream):
        """
        Add audio listener for a participant stream.
        """
        while True:
            try:
                frame = await stream.track.recv()
                global_event_emitter.emit("ON_SPEECH_IN", {"frame": frame, "stream": stream})
                
                audio_data = frame.to_ndarray()[0]
                pcm_frame = audio_data.flatten().astype(np.int16).tobytes()
                
                if self.pipeline:
                    await self.pipeline.on_audio_delta(pcm_frame)
                else:
                    logger.warning("No pipeline available for audio processing")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                break

    async def add_video_listener(self, stream: Stream):
        """
        Add video listener for a participant stream.
        """
        while True:
            try:
                await asyncio.sleep(0.01)
                frame = await stream.track.recv()
                
                if self.pipeline:
                    await self.pipeline.on_video_delta(frame)
                else:
                    logger.warning("No pipeline available for video processing")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Video processing error: {e}")
                break

    def cancel_tasks(self):
        """
        Cancel all active listener tasks.
        """
        for task_id, task in self.audio_listener_tasks.items():
            if not task.done():
                task.cancel()
        self.audio_listener_tasks.clear()

        for task_id, task in self.video_listener_tasks.items():
            if not task.done():
                task.cancel()
        self.video_listener_tasks.clear()