import asyncio
import wave
import logging
from typing import IO, Any
from dataclasses import dataclass

@dataclass
class BackgroundAudioConfig:
    file_path: str
    enabled: bool = True

logger = logging.getLogger(__name__)

class BackgroundAudio:
    def __init__(self, config: BackgroundAudioConfig, audio_track: Any, chunk_size: int = 320):
        self.config = config
        self.audio_track = audio_track
        self.chunk_size = chunk_size
        self._task: asyncio.Task | None = None
        self._is_playing = False
        self.wf: IO[bytes] | None = None

    async def start(self):
        if not self._is_playing and self.config.enabled:
            self._is_playing = True
            self._task = asyncio.create_task(self._loop_sound())

    async def stop(self):
        if self._is_playing:
            self._is_playing = False
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
            while self._is_playing:
                data = self.wf.readframes(self.chunk_size)
                if not data:
                    self.wf.rewind()
                    data = self.wf.readframes(self.chunk_size)
                
                if hasattr(self.audio_track, 'add_new_bytes'):
                    await self.audio_track.add_new_bytes(data)
                
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error playing background audio: {e}")
        finally:
            if self.wf:
                self.wf.close()
                self.wf = None
