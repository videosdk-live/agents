import logging
import asyncio
from fractions import Fraction
from time import time
import traceback
from av import AudioFrame
import numpy as np
from videosdk import CustomAudioTrack
logger = logging.getLogger(__name__)


AUDIO_PTIME = 0.02


class MediaStreamError(Exception):
    pass


class CustomAudioStreamTrack(CustomAudioTrack):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self._start = None
        self._timestamp = 0
        self.frame_buffer = []
        self.audio_data_buffer = bytearray()
        self.frame_time = 0
        self.sample_rate = 24000
        self.channels = 1
        self.sample_width = 2
        self.time_base_fraction = Fraction(1, self.sample_rate)
        self.samples = int(AUDIO_PTIME * self.sample_rate)
        self.chunk_size = int(self.samples * self.channels * self.sample_width)

    def interrupt(self):
        self.frame_buffer.clear()
        self.audio_data_buffer.clear()
            
    async def add_new_bytes(self, audio_data: bytes):
        self.audio_data_buffer += audio_data

        while len(self.audio_data_buffer) >= self.chunk_size:
            chunk = self.audio_data_buffer[: self.chunk_size]
            self.audio_data_buffer = self.audio_data_buffer[self.chunk_size :]
            try:
                audio_frame = self.buildAudioFrames(chunk)
                self.frame_buffer.append(audio_frame)
                logger.debug(
                    f"Added audio frame to buffer, total frames: {len(self.frame_buffer)}"
                )
            except Exception as e:
                logger.error(f"Error building audio frame: {e}")
                break

    def buildAudioFrames(self, chunk: bytes) -> AudioFrame:
        if len(chunk) != self.chunk_size:
            logger.warning(
                f"Incorrect chunk size received {len(chunk)}, expected {self.chunk_size}"
            )

        data = np.frombuffer(chunk, dtype=np.int16)
        expected_samples = self.samples * self.channels
        if len(data) != expected_samples:
            logger.warning(
                f"Incorrect number of samples in chunk {len(data)}, expected {expected_samples}"
            )

        data = data.reshape(-1, self.channels)
        layout = "mono" if self.channels == 1 else "stereo"

        audio_frame = AudioFrame.from_ndarray(data.T, format="s16", layout=layout)
        return audio_frame

    def next_timestamp(self):
        pts = int(self.frame_time)
        time_base = self.time_base_fraction
        self.frame_time += self.samples
        return pts, time_base

    async def recv(self) -> AudioFrame:
        try:
            if self.readyState != "live":
                raise MediaStreamError

            if self._start is None:
                self._start = time()
                self._timestamp = 0
            else:
                self._timestamp += self.samples

            wait = self._start + (self._timestamp / self.sample_rate) - time()

            if wait > 0:
                await asyncio.sleep(wait)

            pts, time_base = self.next_timestamp()

            if len(self.frame_buffer) > 0:
                frame = self.frame_buffer.pop(0)
            else:
                frame = AudioFrame(format="s16", layout="mono", samples=self.samples)
                for p in frame.planes:
                    p.update(bytes(p.buffer_size))

            frame.pts = pts
            frame.time_base = time_base
            frame.sample_rate = self.sample_rate
            return frame
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error while creating tts->rtc frame: {e}")

    async def cleanup(self):
        self.interrupt()
        self.stop()


class TeeCustomAudioStreamTrack(CustomAudioStreamTrack):
    def __init__(self, loop, sinks=None, pipeline=None):
        super().__init__(loop)
        self.sinks = sinks if sinks is not None else []
        self.pipeline = pipeline

    async def add_new_bytes(self, audio_data: bytes):
        await super().add_new_bytes(audio_data)

        # Route audio to sinks (avatars, etc.)
        for sink in self.sinks:
            if hasattr(sink, "handle_audio_input"):
                await sink.handle_audio_input(audio_data)

        # DO NOT route agent's own TTS audio back to pipeline
        # The pipeline should only receive audio from other participants
        # This prevents the agent from hearing itself speak
