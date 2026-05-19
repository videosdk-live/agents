import asyncio
import logging
import time as _time
import traceback
from dataclasses import dataclass
from typing import Any

import av
import numpy as np
from av import AudioFrame
from av.audio.resampler import AudioResampler

@dataclass
class BackgroundAudioHandlerConfig:
    """Configuration for background audio playback including file path, volume, mode, and looping settings."""

    file_path: str
    enabled: bool = True
    mode: str = 'playback' # 'playback' or 'mixing'
    volume: float = 1.0
    looping: bool = False

logger = logging.getLogger(__name__)

class BackgroundAudioHandler:
    """Handles decoding any libav-decodable audio file and streaming it to an audio track in playback or mixing mode."""

    def __init__(self, config: BackgroundAudioHandlerConfig, audio_track: Any, chunk_size: int = 320):
        self.config = config
        self.audio_track = audio_track
        self.chunk_size = chunk_size  # accepted for backward compat; unused by the PyAV path
        self._task: asyncio.Task | None = None
        self.is_playing = False
        self._container: av.container.InputContainer | None = None
        self._resampler: AudioResampler | None = None
        self._cache: bytes | None = None

    async def start(self):
        """Start background audio playback if enabled and not already playing."""
        if not self.is_playing and self.config.enabled:
            self.is_playing = True
            self._task = asyncio.create_task(self._loop_sound())

    async def stop(self):
        """Stop background audio playback, cancel the task, and release decoder resources."""
        if self.is_playing:
            self.is_playing = False
            if self._task is not None:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                self._task = None
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None
        self._resampler = None
        self._cache = None

    async def _loop_sound(self):
        """Decode the source via PyAV and stream resampled audio to the track.

        looping=False -> one decode pass, EOF ends playback.
        looping=True  -> first pass streams to track and fills self._cache; the
                         container and resampler are released at EOF; subsequent
                         loops slice from the cache and pace against the same
                         play-out clock.
        """
        target_rate = getattr(self.audio_track, "sample_rate", 24000)
        target_channels = getattr(self.audio_track, "channels", 1)
        target_layout = "mono" if target_channels == 1 else "stereo"

        try:
            collect = bytearray() if self.config.looping else None

            await self._decode_and_push_once(
                target_rate, target_layout, collect=collect,
            )

            if self.config.looping and self.is_playing and collect is not None and len(collect) > 0:
                self._cache = bytes(collect)
                await self._replay_cache(target_rate, target_channels)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"Error playing background audio: {exc}")
            traceback.print_exc()
        finally:
            self.is_playing = False
            if self._container is not None:
                try:
                    self._container.close()
                except Exception:
                    pass
                self._container = None
            self._resampler = None

    async def _decode_and_push_once(
        self,
        target_rate: int,
        target_layout: str,
        *,
        collect: bytearray | None = None,
    ) -> None:
        """Open the file, decode every frame, resample, push to the track. Closes container on exit."""
        try:
            self._container = av.open(self.config.file_path)
        except Exception as exc:
            logger.error(f"Failed to open background audio '{self.config.file_path}': {exc}")
            self.is_playing = False
            return

        stream = next((s for s in self._container.streams if s.type == "audio"), None)
        if stream is None:
            logger.error(f"No audio stream in '{self.config.file_path}'")
            self._container.close()
            self._container = None
            self.is_playing = False
            return

        self._resampler = AudioResampler(format="s16", layout=target_layout, rate=target_rate)

        start_wall = _time.monotonic()
        produced_seconds = 0.0
        lead = 0.04

        async def push_and_pace(out_bytes: bytes, frame_seconds: float) -> None:
            nonlocal produced_seconds
            if collect is not None:
                collect.extend(out_bytes)
            if self.config.mode == "mixing":
                if hasattr(self.audio_track, "add_background_bytes"):
                    await self.audio_track.add_background_bytes(out_bytes)
            else:
                if hasattr(self.audio_track, "add_new_bytes"):
                    await self.audio_track.add_new_bytes(out_bytes)
            produced_seconds += frame_seconds
            sleep_for = max(0.0, (start_wall + produced_seconds - lead) - _time.monotonic())
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

        try:
            for src_frame in self._container.decode(stream):
                if not self.is_playing:
                    break
                for out_frame in self._resampler.resample(src_frame):
                    out_bytes = out_frame.to_ndarray().tobytes()
                    if not out_bytes:
                        continue
                    if self.config.volume < 1.0:
                        arr = np.frombuffer(out_bytes, dtype=np.int16)
                        arr = (arr.astype(np.float32) * self.config.volume).astype(np.int16)
                        out_bytes = arr.tobytes()
                    frame_seconds = out_frame.samples / target_rate
                    await push_and_pace(out_bytes, frame_seconds)
            for out_frame in self._resampler.resample(None):
                out_bytes = out_frame.to_ndarray().tobytes()
                if not out_bytes:
                    continue
                if self.config.volume < 1.0:
                    arr = np.frombuffer(out_bytes, dtype=np.int16)
                    arr = (arr.astype(np.float32) * self.config.volume).astype(np.int16)
                    out_bytes = arr.tobytes()
                frame_seconds = out_frame.samples / target_rate
                await push_and_pace(out_bytes, frame_seconds)
        except av.error.InvalidDataError as exc:
            logger.warning(f"Decode error in '{self.config.file_path}': {exc}")
        finally:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None
            self._resampler = None

    async def _replay_cache(self, target_rate: int, target_channels: int) -> None:
        """Stream self._cache cyclically until is_playing becomes False."""
        assert self._cache is not None
        chunk_size = int(0.02 * target_rate) * target_channels * 2  # 20 ms s16
        if chunk_size <= 0 or len(self._cache) == 0:
            return

        start_wall = _time.monotonic()
        produced_seconds = 0.0
        lead = 0.04
        position = 0
        cache_len = len(self._cache)

        while self.is_playing:
            end = position + chunk_size
            if end <= cache_len:
                chunk = self._cache[position:end]
                position = end
            else:
                # Wrap across the loop boundary
                chunk = self._cache[position:] + self._cache[: end - cache_len]
                position = end - cache_len
            if position == cache_len:
                position = 0

            if self.config.mode == "mixing":
                if hasattr(self.audio_track, "add_background_bytes"):
                    await self.audio_track.add_background_bytes(chunk)
            else:
                if hasattr(self.audio_track, "add_new_bytes"):
                    await self.audio_track.add_new_bytes(chunk)

            produced_seconds += (chunk_size / 2) / target_rate / target_channels
            sleep_for = max(0.0, (start_wall + produced_seconds - lead) - _time.monotonic())
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
