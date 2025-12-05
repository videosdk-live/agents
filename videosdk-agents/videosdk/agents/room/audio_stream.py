import logging
import asyncio
from fractions import Fraction
from time import time
import traceback
from av import AudioFrame
import numpy as np
from videosdk import CustomAudioTrack
from ..event_bus import global_event_emitter
from typing import Callable, Awaitable

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
        self.frame_buffer = asyncio.Queue(maxsize=100)
        self.audio_data_buffer = bytearray()
        self.frame_time = 0
        self.sample_rate = 24000
        self.channels = 1
        self.sample_width = 2
        self.time_base_fraction = Fraction(1, self.sample_rate)
        self.samples = int(AUDIO_PTIME * self.sample_rate)
        self.chunk_size = int(self.samples * self.channels * self.sample_width)
        self._is_speaking = False
        self._last_audio_callback = None
        self._accepting_audio = True
        self._manual_audio_control = False
        
        self._playback_enabled = asyncio.Event()
        self._playback_enabled.set()

    @property
    def can_pause(self) -> bool:
        return True

    async def pause(self) -> None:
        logger.info("Audio track paused.")
        self._playback_enabled.clear()

    async def resume(self) -> None:
        logger.info("Audio track resumed.")
        self._playback_enabled.set()

    def interrupt(self):
        logger.info("Audio track interrupted, clearing buffers.")
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.audio_data_buffer = bytearray()
        if self._manual_audio_control:
            self._accepting_audio = False
        else:
            self._accepting_audio = True
        self._playback_enabled.set()

    def enable_audio_input(self, manual_control: bool = False):
        """
        Allow fresh audio data to be buffered. When manual_control is True,
        future interrupts will pause intake until this method is called again.
        """
        self._manual_audio_control = manual_control
        self._accepting_audio = True

    def on_last_audio_byte(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Set callback for when the final audio byte of synthesis is produced"""
        logger.info("on last audio callback")
        self._last_audio_callback = callback
            
    async def add_new_bytes(self, audio_data: bytes):
        if not self._accepting_audio:
            return
        global_event_emitter.emit("ON_SPEECH_OUT", {"audio_data": audio_data})
        self.audio_data_buffer += audio_data

        while len(self.audio_data_buffer) >= self.chunk_size:
            chunk = self.audio_data_buffer[: self.chunk_size]
            self.audio_data_buffer = self.audio_data_buffer[self.chunk_size :]
            try:
                await self.frame_buffer.put(self.buildAudioFrames(chunk))
            except Exception as e:
                logger.error(f"Error building audio frame: {e}")
                break

    def buildAudioFrames(self, chunk: bytes) -> AudioFrame:
        data = np.frombuffer(chunk, dtype=np.int16).reshape(-1, self.channels)
        return AudioFrame.from_ndarray(data.T, format="s16", layout="mono")

    def next_timestamp(self):
        pts = int(self.frame_time)
        self.frame_time += self.samples
        return pts, Fraction(1, self.sample_rate)

    async def recv(self) -> AudioFrame:
        try:
            if self.readyState != "live":
                raise MediaStreamError

            frame = None
            try:
                await asyncio.wait_for(self._playback_enabled.wait(), timeout=AUDIO_PTIME)
            except asyncio.TimeoutError:
                frame = AudioFrame(format="s16", layout="mono", samples=self.samples)
                for p in frame.planes:
                    p.update(bytes(p.buffer_size))

            if frame is None:
                try:
                    frame = self.frame_buffer.get_nowait()
                    if not self._is_speaking:
                        self._is_speaking = True
                except asyncio.QueueEmpty:
                    if self._is_speaking:
                        self._is_speaking = False
                        if self._last_audio_callback:
                            asyncio.create_task(self._last_audio_callback())
                    
                    frame = AudioFrame(format="s16", layout="mono", samples=self.samples)
                    for p in frame.planes:
                        p.update(bytes(p.buffer_size))

            if self._start is None:
                self._start = time()

            wait = (self._start + (self.frame_time / self.sample_rate)) - time()
            if wait > 0:
                await asyncio.sleep(wait)

            pts, time_base = self.next_timestamp()
            frame.pts = pts
            frame.time_base = time_base
            frame.sample_rate = self.sample_rate
            return frame
        except Exception as e:
            logger.error(f"Error in recv: {e}", exc_info=True)
            return AudioFrame(format="s16", layout="mono", samples=self.samples)
            
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

class TeeMixingCustomAudioStreamTrack(MixingCustomAudioStreamTrack):
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
        
