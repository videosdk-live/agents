import asyncio
from fractions import Fraction
import threading
from time import time
import traceback
from typing import Iterator, Optional
from av import AudioFrame
import numpy as np
from videosdk import CustomAudioTrack

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
        self._process_audio_task_queue = asyncio.Queue()
        self._process_audio_thread = threading.Thread(target=self.run_process_audio)
        self._process_audio_thread.daemon = True
        self._process_audio_thread.start()
        self.skip_next_chunk = False

    def interrupt(self):
        length = len(self.frame_buffer)
        self.frame_buffer.clear()
        while not self._process_audio_task_queue.empty():
            self.skip_next_chunk = True
            self._process_audio_task_queue.get_nowait()
            self._process_audio_task_queue.task_done()

        if length > 0:
            self.skip_next_chunk = True

    async def add_new_bytes(self, audio_data_stream: Iterator[bytes]):
        # self.interrupt()
        await self._process_audio_task_queue.put(audio_data_stream)

    def run_process_audio(self):
        asyncio.run(self._process_audio())

    async def _process_audio(self):
        while True:
            try:
                    while True:
                        if len(self.frame_buffer) > 0:
                            await asyncio.sleep(0.1)
                            continue
                        break
            except Exception as e:
                print("Error while updating chracter state", e)

            try:
                audio_data_stream = asyncio.run_coroutine_threadsafe(
                    self._process_audio_task_queue.get(), self.loop
                ).result()
                for audio_data in audio_data_stream:
                    try:
                        self.audio_data_buffer += audio_data
                        while len(self.audio_data_buffer) > self.chunk_size:
                            chunk = self.audio_data_buffer[: self.chunk_size]
                            self.audio_data_buffer = self.audio_data_buffer[
                                self.chunk_size :
                            ]
                            audio_frame = self.buildAudioFrames(chunk)
                            self.frame_buffer.append(audio_frame)
                    except Exception as e:
                        print("Error while putting audio data stream", e)
            except Exception as e:
                traceback.print_exc()
                print("Error while process audio", e)

    def buildAudioFrames(self, chunk: bytes) -> AudioFrame:
        data = np.frombuffer(chunk, dtype=np.int16)
        data = data.reshape(-1, 1)
        audio_frame = AudioFrame.from_ndarray(data.T, format="s16", layout="mono")
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
            print("error while creating tts->rtc frame", e)
