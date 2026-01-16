import logging
import asyncio
from fractions import Fraction
from time import time
import traceback
from av import AudioFrame
import numpy as np
from videosdk import CustomAudioTrack
from ..event_bus import global_event_emitter
from typing import Callable, Awaitable, List, Union

logger = logging.getLogger(__name__)

AUDIO_PTIME = 0.02


class MediaStreamError(Exception):
    pass


class AdaptiveAudioStreamTrack(CustomAudioTrack):
    """
    A single audio track implementation that adapts based on context:
    - Buffers primary audio (TTS).
    - Mixes background audio if present.
    - Routes audio to sinks (avatars, etc.) if sinks are present.
    - Supports pause/resume.
    
    Uses Just-In-Time (JIT) frame creation in recv() to facilitate mixing and smooth timing.
    """
    def __init__(self, loop, sinks: List[Union[Callable, object]] = None, background_audio: bool = False):
        super().__init__()
        self.loop = loop
        
        # Audio configuration
        self.sample_rate = 24000
        self.channels = 1
        self.sample_width = 2
        self.time_base_fraction = Fraction(1, self.sample_rate)
        self.samples = int(AUDIO_PTIME * self.sample_rate)
        self.chunk_size = int(self.samples * self.channels * self.sample_width)

        # Buffers (Byte-level for JIT mixing)
        self.audio_data_buffer = bytearray()
        self.background_audio_buffer = bytearray()
        
        # Sinks (avatars, recorders, etc.)
        self.sinks = sinks if sinks is not None else []

        # State management
        self._start = None
        self._timestamp = 0
        self._is_speaking = False
        self._is_paused = False
        self._accepting_audio = True
        self._manual_audio_control = False
        self._last_audio_callback = None
        self.background_audio = background_audio
        # Metrics/Status
        self.frame_time = 0 
        print("New object of AdaptiveAudioStreamTrack created background is set to ", background_audio)

    @property
    def background_audio(self):
        return self._background_audio

    @background_audio.setter
    def background_audio(self, value):
        self._background_audio = value

    @property
    def can_pause(self) -> bool:
        """Returns True if this track supports pause/resume operations"""
        return True

    def add_sink(self, sink):
        """Add a new sink (callback or object)"""
        if sink not in self.sinks:
            self.sinks.append(sink)

    def remove_sink(self, sink):
        if sink in self.sinks:
            self.sinks.remove(sink)

    def enable_audio_input(self, manual_control: bool = False):
        """
        Allow fresh audio data to be buffered. When manual_control is True,
        future interrupts will pause intake until this method is called again.
        """
        self._manual_audio_control = manual_control
        self._accepting_audio = True
        logger.debug(f"Audio input enabled (manual_control={manual_control})")

    def interrupt(self):
        """Clear all buffers and reset state"""
        logger.info("AdaptiveAudioTrack interrupted, clearing buffers.")
        self.audio_data_buffer.clear()
        # self.background_audio_buffer.clear()
        self._is_paused = False
        
        # Handle manual audio control mode
        if self._manual_audio_control:
            self._accepting_audio = False
        else:
            self._accepting_audio = True

    async def pause(self) -> None:
        """
        Pause audio playback. 
        In JIT mode, we simply set the flag. recv() will produce silence 
        and stop consuming the byte buffers.
        """
        if self._is_paused:
            logger.warning("Audio track already paused")
            return
            
        logger.info("Audio track paused.")
        self._is_paused = True

    async def resume(self) -> None:
        """
        Resume audio playback from paused position.
        """
        if not self._is_paused:
            logger.warning("Audio track not paused, nothing to resume")
            return
            
        logger.info("Audio track resumed.")
        self._is_paused = False

    def on_last_audio_byte(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Set callback for when the final audio byte of synthesis is produced"""
        logger.info("on last audio callback set")
        self._last_audio_callback = callback

    async def add_new_bytes(self, audio_data: bytes):
        """
        Add new audio bytes to the buffer (primary audio).
        Also routes to sinks.
        """
        if not self._accepting_audio:
            logger.debug("Audio input currently disabled, dropping audio data")
            return
            
        global_event_emitter.emit("ON_SPEECH_OUT", {"audio_data": audio_data})
        self.audio_data_buffer += audio_data

        # Route audio to sinks (avatars, etc.)
        if self.sinks:
            for sink in self.sinks:
                try:
                    if hasattr(sink, "handle_audio_input"):
                        await sink.handle_audio_input(audio_data)
                    elif callable(sink):
                        if asyncio.iscoroutinefunction(sink):
                            await sink(audio_data)
                        else:
                            sink(audio_data)
                except Exception as e:
                    logger.error(f"Error dispatching audio to sink: {e}")

    async def add_background_bytes(self, audio_data: bytes):
        """Add background audio bytes (e.g. music, ambience) to be mixed."""
        self.background_audio_buffer += audio_data

    def _mix_audio(self, primary_chunk, background_chunk):
        """Optimized mixing with reduced operations"""
        if not background_chunk and self.background_audio:
            return primary_chunk

        # Single conversion, avoid intermediate copies
        primary_arr = np.frombuffer(primary_chunk, dtype=np.int16)
        
        # Handle background length efficiently
        bg_len = len(background_chunk) // 2  # int16 = 2 bytes
        prim_len = len(primary_arr)
        
        if bg_len == prim_len:
            # Fast path: same length
            background_arr = np.frombuffer(background_chunk, dtype=np.int16)
        elif bg_len < prim_len:
            # Pad efficiently
            background_arr = np.frombuffer(background_chunk, dtype=np.int16)
            background_arr = np.pad(background_arr, (0, prim_len - bg_len), 'constant')
        else:
            # Trim efficiently - use view instead of copy
            background_arr = np.frombuffer(background_chunk, dtype=np.int16)[:prim_len]
        
        # Use in-place operations where possible
        # Mix at int32 level to avoid overflow, then clip
        mixed = primary_arr.astype(np.int32)
        mixed += background_arr.astype(np.int32)
        np.clip(mixed, -32768, 32767, out=mixed)
        
        return mixed.astype(np.int16).tobytes()

    # def _mix_audio(self, primary_chunk, background_chunk):
    #     """Mix primary and background audio chunks."""
    #     if not background_chunk:
    #         return primary_chunk

    #     primary_arr = np.frombuffer(primary_chunk, dtype=np.int16)
    #     background_arr = np.frombuffer(background_chunk, dtype=np.int16)

    #     # Pad or trim background to match primary length
    #     if len(background_arr) < len(primary_arr):
    #         background_arr = np.pad(background_arr, (0, len(primary_arr) - len(background_arr)), 'constant')
    #     elif len(background_arr) > len(primary_arr):
    #         background_arr = background_arr[:len(primary_arr)]

    #     # Add with overflow protection
    #     mixed_arr = np.clip(primary_arr.astype(np.int32) + background_arr.astype(np.int32), -32768, 32767).astype(np.int16)
    #     return mixed_arr.tobytes()

    def _build_audio_frame(self, chunk: bytes) -> AudioFrame:
        """Convert bytes to AudioFrame."""
        if len(chunk) != self.chunk_size:
            logger.warning(
                f"Incorrect chunk size received {len(chunk)}, expected {self.chunk_size}"
            )

        data = np.frombuffer(chunk, dtype=np.int16)
        expected_samples = self.samples * self.channels
        if len(data) != expected_samples:
            # Re-verify length in case of padding issues
            if len(data) < expected_samples:
                data = np.pad(data, (0, expected_samples - len(data)), 'constant')
            else:
                 data = data[:expected_samples]

        data = data.reshape(-1, self.channels)
        layout = "mono" if self.channels == 1 else "stereo"

        audio_frame = AudioFrame.from_ndarray(data.T, format="s16", layout=layout)
        return audio_frame

    def next_timestamp(self):
        pts = int(self.frame_time)
        time_base = self.time_base_fraction
        self.frame_time += self.samples
        return pts, time_base

    async def cleanup(self):
        self.interrupt()
        self.stop()

    async def recv(self) -> AudioFrame:
        """
        Receive next audio frame.
        Handles mixing and pausing just-in-time.
        """
        try:
            if self.readyState != "live":
                raise MediaStreamError

            # 1. Timing control
            if self._start is None:
                self._start = time()
                self._timestamp = 0
            else:
                self._timestamp += self.samples

            wait = self._start + (self._timestamp / self.sample_rate) - time()
            if wait > 0:
                await asyncio.sleep(wait)

            pts, time_base = self.next_timestamp()

            # 2. Check for Pause
            # When paused, produce silence but keep timing
            if self._is_paused:
                frame = AudioFrame(format="s16", layout="mono", samples=self.samples)
                for p in frame.planes:
                    p.update(bytes(p.buffer_size))
                
                frame.pts = pts
                frame.time_base = time_base
                frame.sample_rate = self.sample_rate
                return frame

            # 3. Prepare Primary Chunk
            primary_chunk = b''
            has_primary = len(self.audio_data_buffer) >= self.chunk_size

            if has_primary:
                primary_chunk = self.audio_data_buffer[: self.chunk_size]
                self.audio_data_buffer = self.audio_data_buffer[self.chunk_size :]
                # We are technically speaking if we are consuming buffer
                self._is_speaking = True
            elif getattr(self, "_is_speaking", False):
                # We were speaking, but ran out of data -> Finished speaking
                logger.info("[AudioTrack] Agent finished speaking — triggering last_audio_callback.")
                self._is_speaking = False

                if self._last_audio_callback:
                    # Execute callback
                    if asyncio.iscoroutinefunction(self._last_audio_callback):
                         asyncio.create_task(self._last_audio_callback())
                    else:
                        asyncio.create_task(self._last_audio_callback())
            # 4. Prepare Background Chunk
            background_chunk = b''
            has_background = len(self.background_audio_buffer) >= self.chunk_size
            
            if has_background:
                background_chunk = self.background_audio_buffer[: self.chunk_size]
                self.background_audio_buffer = self.background_audio_buffer[self.chunk_size :]
            
            # 5. Mix or Passthrough
            final_chunk = None
            if self.background_audio:
                if has_primary:
                    final_chunk = self._mix_audio(primary_chunk, background_chunk)
                elif has_background:
                    final_chunk = background_chunk
                else:
                    final_chunk = primary_chunk
            else:
                final_chunk = primary_chunk
            
            # 6. Build Frame
            if final_chunk:
                frame = self._build_audio_frame(final_chunk)
            else:
                # Silence
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
            # Return silence on error
            frame = AudioFrame(format="s16", layout="mono", samples=self.samples)
            for p in frame.planes:
                p.update(bytes(p.buffer_size))
            return frame