import asyncio
import wave
import logging
from typing import IO, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class BackgroundAudioHandlerConfig:
    file_path: str
    enabled: bool = True
    mode: str = 'playback' # 'playback' or 'mixing'
    volume: float = 1.0
    looping: bool = False

logger = logging.getLogger(__name__)

class BackgroundAudioHandler:
    def __init__(self, config: BackgroundAudioHandlerConfig, audio_track: Any, chunk_size: int = 320):
        self.config = config
        self.audio_track = audio_track
        self.chunk_size = chunk_size
        self._task: asyncio.Task | None = None
        self.is_playing = False
        self.wf: IO[bytes] | None = None

    async def start(self):
        if not self.is_playing and self.config.enabled:
            self.is_playing = True
            self._task = asyncio.create_task(self._loop_sound())

    async def stop(self):
        if self.is_playing:
            self.is_playing = False

            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                self._task = None
        if self.wf:
            self.wf.close()
            self.wf = None

    async def _loop_sound(self):
        try:
            self.wf = wave.open(self.config.file_path, 'rb')
            while self.is_playing:
                data = self.wf.readframes(self.chunk_size)
                if not data:
                    if self.config.looping:
                        self.wf.rewind()
                        data = self.wf.readframes(self.chunk_size)
                        if not data:
                            break
                    else:
                        break
                
                if self.config.volume < 1.0:
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    audio_array = (audio_array * self.config.volume).astype(np.int16)
                    data = audio_array.tobytes()

                if self.config.mode == 'mixing':
                    if hasattr(self.audio_track, 'add_background_bytes'):
                        await self.audio_track.add_background_bytes(data)
                else:
                    if hasattr(self.audio_track, 'add_new_bytes'):
                        await self.audio_track.add_new_bytes(data)
                
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error playing background audio: {e}")
        finally:
            if not self.config.looping:
                self.is_playing = False
            if self.wf:
                self.wf.close()
                self.wf = None