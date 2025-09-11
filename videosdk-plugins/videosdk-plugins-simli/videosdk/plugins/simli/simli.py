import time
import traceback
import asyncio
import fractions
from aiortc import MediaStreamTrack, VideoStreamTrack
from av import VideoFrame
from simli import SimliConfig, SimliClient
from av.audio.resampler import AudioResampler

import numpy as np
from av.audio import AudioFrame
from vsaiortc.mediastreams import MediaStreamError

from videosdk import CustomVideoTrack, CustomAudioTrack
import logging

logger = logging.getLogger(__name__)

# --- Constants ---
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2
AUDIO_SAMPLE_WIDTH = 2
AUDIO_FRAME_DURATION_S = 0.02
AUDIO_SAMPLES_PER_FRAME = int(AUDIO_FRAME_DURATION_S * AUDIO_SAMPLE_RATE)
AUDIO_CHUNK_SIZE = AUDIO_SAMPLES_PER_FRAME * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH
AUDIO_TIME_BASE_FRACTION = fractions.Fraction(1, AUDIO_SAMPLE_RATE)
SIMLI_INPUT_SAMPLING_RATE = 16000

VIDEO_TIME_BASE = 90000

DEFAULT_SIMLI_HTTP_URL = "https://api.simli.ai"

simliInputResampler = AudioResampler(
    format="s16", layout="mono", rate=SIMLI_INPUT_SAMPLING_RATE
)
videosdkOutputResampler = AudioResampler(
    format="s16", layout="stereo", rate=AUDIO_SAMPLE_RATE
)


class SimliAudioTrack(CustomAudioTrack):
    def __init__(self, loop):
        super().__init__()
        self.kind = "audio"
        self.loop = loop
        self._timestamp = 0
        self.queue = asyncio.Queue(maxsize=200)
        self.audio_data_buffer = bytearray()
        self.frame_time = 0
        self.sample_rate = AUDIO_SAMPLE_RATE
        self.channels = AUDIO_CHANNELS
        self.sample_width = AUDIO_SAMPLE_WIDTH
        self.time_base_fraction = AUDIO_TIME_BASE_FRACTION
        self.samples = AUDIO_SAMPLES_PER_FRAME
        self.chunk_size = AUDIO_CHUNK_SIZE
        self._start_time = None
        self._shared_start_time = None
        self._frame_duration = AUDIO_FRAME_DURATION_S
        self._last_frame_time = 0
        self._frame_count = 0

    def interrupt(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.audio_data_buffer.clear()

    async def recv(self) -> AudioFrame:
        """Return next audio frame to VideoSDK."""
        try:
            if self.readyState != "live":
                raise MediaStreamError
            frame = await self.queue.get()
            return frame

        except Exception:
            return self._create_silence_frame()

    async def cleanup(self):
        self.interrupt()
        self.stop()

    def add_frame(self, frame: AudioFrame):
        """Add frame from Simli stream - add AudioFrame directly to buffer with quality validation"""

        if frame is None:
            return
        try:
            try:
                for resampledFrame in videosdkOutputResampler.resample(frame):
                    self.queue.put_nowait(resampledFrame)
            except asyncio.QueueEmpty:
                pass
            except asyncio.QueueFull:
                logger.warning("Simli: Audio frame queue is full. Frame dropped.")
        except Exception as e:
            logger.error(f"Error adding Simli audio frame: {e}")


class SimliVideoTrack(CustomVideoTrack):
    def __init__(self, is_trinity_avatar=True):
        super().__init__()
        self.kind = "video"
        self.queue = asyncio.Queue(maxsize=24)
        self._timestamp = 0
        self._start_time = None
        self._frame_count = 0
        self._readyState = "live"
        self._frame_rate = 25 if is_trinity_avatar else 30
        self._frame_duration = 1.0 / self._frame_rate
        self._shared_start_time = None

    @property
    def readyState(self):
        return self._readyState

    async def recv(self) -> VideoFrame:
        frame = await self.queue.get()
        if self._start_time is None:
            self._start_time = (
                self._shared_start_time if self._shared_start_time else time.time()
            )
            self._timestamp = 0
        return frame

    def add_frame(self, frame: VideoFrame):
        self.queue.put_nowait(frame)


class SimliAvatar:
    def __init__(
        self,
        config: SimliConfig,
        simli_url: str = DEFAULT_SIMLI_HTTP_URL,
        is_trinity_avatar: bool = False,
    ):
        """Initialize the Simli Avatar plugin.

        Args:
            config (SimliConfig): The configuration for the Simli avatar.
            simli_url (str): The Simli API URL. Defaults to "https://api.simli.ai".
        """
        super().__init__()
        self.config = config
        self._stream_start_time = None
        self.video_track = None
        self.audio_track = None
        self.run = True
        self._is_speaking = False
        self._speech_timeout_task = None
        self._avatar_speaking = False
        self._last_error = None
        self._stopping = False
        self._keep_alive_task = None
        self._last_audio_time = 0
        self._is_trinity_avatar = is_trinity_avatar
        self.video_first_frame_received = False
        self.audio_first_frame_received = False
        self.simliURL = simli_url

    async def connect(self):
        loop = asyncio.get_event_loop()
        await self._initialize_connection()

        self.audio_track = SimliAudioTrack(loop)
        self.video_track = SimliVideoTrack(self._is_trinity_avatar)
        if self._stream_start_time is None:
            self._stream_start_time = time.time()
            self.video_track._shared_start_time = self._stream_start_time
            self.audio_track._shared_start_time = self._stream_start_time

        if hasattr(self.video_track, "start"):
            self.video_track.start()
        if hasattr(self.audio_track, "start"):
            self.audio_track.start()

        self._last_audio_time = time.time()
        self._keep_alive_task = asyncio.create_task(self._keep_alive_loop())

    async def _initialize_connection(self):
        """Initialize connection with retry logic"""
        self.simliClient = SimliClient(self.config, True, 0, self.simliURL)
        await self.simliClient.Initialize()
        while not hasattr(self.simliClient, "audioReceiver"):
            await asyncio.sleep(0.0001)
        self._on_track(self.simliClient.audioReceiver)
        self._on_track(self.simliClient.videoReceiver)
        # self.simliClient.registerSilentEventCallback()

    def _on_track(self, track: MediaStreamTrack):
        if track.kind == "video":
            self.video_receiver_track: VideoStreamTrack = track
            asyncio.ensure_future(self._process_video_frames())
        elif track.kind == "audio":
            self.audio_receiver_track: MediaStreamTrack = track
            asyncio.ensure_future(self._process_audio_frames())

    async def _process_video_frames(self):
        """Simple video frame processing for real-time playback"""
        frame_count = 0
        while self.run and not self._stopping:
            try:
                if not hasattr(self, "video_receiver_track"):
                    await asyncio.sleep(0.001)
                    continue
                frame: VideoFrame = await self.video_receiver_track.recv()
                self.video_first_frame_received = True
                if frame is None:
                    continue
                while not self.audio_first_frame_received:
                    await asyncio.sleep(0.001)

                frame_count += 1
                self.video_track.add_frame(frame)

            except asyncio.TimeoutError:
                if self.run and not self._stopping:
                    continue
                else:
                    break
            except Exception as e:
                logger.error(f"Simli: Video processing error: {e}")
                if not self.run or self._stopping:
                    break
                await asyncio.sleep(0.1)
                continue

    async def _process_audio_frames(self):
        """Simple audio frame processing for real-time playback"""
        frame_count = 0
        while self.run and not self._stopping:
            try:
                if not hasattr(self, "audio_receiver_track"):
                    await asyncio.sleep(0.001)
                    continue

                frame: AudioFrame = await self.audio_receiver_track.recv()

                self.audio_first_frame_received = True
                if frame is None:
                    logger.warning("Simli: Received None audio frame, continuing...")
                    continue

                try:
                    while not self.video_first_frame_received:
                        await asyncio.sleep(0.001)
                    frame_count += 1
                    self.audio_track.add_frame(frame)
                except Exception as frame_error:
                    logger.error(
                        f"Simli: Error processing audio frame #{frame_count}: {frame_error}"
                    )
                    continue

            except asyncio.TimeoutError:
                if self.run and not self._stopping:
                    continue
                else:
                    break
            except Exception as e:
                logger.error(f"Simli: Audio processing error: {e}")
                if not self.run or self._stopping:
                    break
                await asyncio.sleep(0.1)
                continue

    async def sendSilence(self, duration: float = 0.1875):
        """Send silence to bootstrap the connection"""
        if self.simliClient.ready:
            await self.simliClient.sendSilence()

    async def _speech_timeout_handler(self):
        try:
            await asyncio.sleep(0.2)
            if self._is_speaking:
                await self.simliClient.send("SKIP")
                self._is_speaking = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in speech timeout handler: {e}")

    async def handle_audio_input(self, audio_data: bytes):
        if not self.run or self._stopping:
            return
        if self.simliClient.ready:
            try:
                if len(audio_data) % 2 != 0:
                    audio_data = audio_data + b"\x00"

                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                input_frame = AudioFrame.from_ndarray(
                    audio_array.reshape(1, -1), format="s16", layout="mono"
                )
                input_frame.sample_rate = 24000

                resampled_frames = simliInputResampler.resample(input_frame)
                for frame in resampled_frames:
                    resampled_data = frame.to_ndarray().tobytes()

                    await self.simliClient.send(resampled_data)

                    self._last_audio_time = time.time()

            except Exception as e:
                logger.error(f"Error processing/sending audio data: {e}")
        else:
            logger.error(
                f"Simli: Cannot send audio - ws available: {self.simliClient is not None}, ready: {self.simliClient.ready}"
            )

    async def aclose(self):
        if self._stopping:
            return
        self._stopping = True
        self.run = False

        if self._keep_alive_task and not self._keep_alive_task.done():
            self._keep_alive_task.cancel()

        if self._speech_timeout_task and not self._speech_timeout_task.done():
            self._speech_timeout_task.cancel()

        try:
            await self.simliClient.stop()
        except Exception:
            pass

        await self._cleanup_connections()

    async def _keep_alive_loop(self):
        """Send periodic keep-alive audio to maintain Simli session"""
        while self.run and not self._stopping:
            try:
                current_time = time.time()
                if current_time - self._last_audio_time > 5.0:
                    if self.simliClient.ready:
                        try:
                            await self.sendSilence()
                            self._last_audio_time = current_time
                        except Exception as e:
                            logger.warning(f"Simli: Keep-alive send failed: {e}")

                await asyncio.sleep(3.0)

            except Exception:
                if not self.run or self._stopping:
                    break
                await asyncio.sleep(1.0)
