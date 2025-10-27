import logging
import asyncio
from fractions import Fraction
from time import time
import traceback
from av import AudioFrame
import numpy as np
from videosdk import CustomAudioTrack
from ..event_bus import global_event_emitter

logger = logging.getLogger(__name__)


AUDIO_PTIME = 0.02


class MediaStreamError(Exception):
    pass


class CustomAudioStreamTrack(CustomAudioTrack):
    """
    Base audio track implementation using a frame buffer.
    Audio frames are created as soon as audio data is received.
    """
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
        global_event_emitter.emit("ON_SPEECH_OUT", {"audio_data": audio_data})
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

class MixingCustomAudioStreamTrack(CustomAudioStreamTrack):
    """
    Audio track implementation with mixing capabilities.
    Inherits from CustomAudioStreamTrack and overrides methods to handle mixing.
    Frames are created just-in-time in the recv method.
    """
    def __init__(self, loop):
        super().__init__(loop)
        self.background_audio_buffer = bytearray()

    def interrupt(self):
        super().interrupt()
        self.background_audio_buffer.clear()

    async def add_new_bytes(self, audio_data: bytes):
        """Overrides base method to buffer bytes instead of creating frames."""
        global_event_emitter.emit("ON_SPEECH_OUT", {"audio_data": audio_data})
        self.audio_data_buffer += audio_data

    async def add_background_bytes(self, audio_data: bytes):
        self.background_audio_buffer += audio_data

    def mix_audio(self, primary_chunk, background_chunk):
        if not background_chunk:
            return primary_chunk

        primary_arr = np.frombuffer(primary_chunk, dtype=np.int16)
        background_arr = np.frombuffer(background_chunk, dtype=np.int16)

        if len(background_arr) < len(primary_arr):
            background_arr = np.pad(background_arr, (0, len(primary_arr) - len(background_arr)), 'constant')
        elif len(background_arr) > len(primary_arr):
            background_arr = background_arr[:len(primary_arr)]

        mixed_arr = np.add(primary_arr, background_arr, dtype=np.int16)
        return mixed_arr.tobytes()

    async def recv(self) -> AudioFrame:
        """
        Overrides base method to perform mixing and just-in-time frame creation.
        """
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

            primary_chunk = b''
            has_primary = len(self.audio_data_buffer) >= self.chunk_size
            if has_primary:
                primary_chunk = self.audio_data_buffer[: self.chunk_size]
                self.audio_data_buffer = self.audio_data_buffer[self.chunk_size :]

            background_chunk = b''
            has_background = len(self.background_audio_buffer) >= self.chunk_size
            if has_background:
                background_chunk = self.background_audio_buffer[: self.chunk_size]
                self.background_audio_buffer = self.background_audio_buffer[self.chunk_size :]
            
            final_chunk = None
            if has_primary:
                final_chunk = self.mix_audio(primary_chunk, background_chunk)
            elif has_background:
                final_chunk = background_chunk
            
            if final_chunk:
                frame = self.buildAudioFrames(final_chunk)
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

class TeeCustomAudioStreamTrack(MixingCustomAudioStreamTrack):
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
