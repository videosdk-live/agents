import asyncio
import time
import traceback
import logging
from typing import Optional

from av import VideoFrame, AudioFrame
from av.audio.resampler import AudioResampler
import numpy as np

from anam import AnamClient
from anam.types import PersonaConfig, AgentAudioInputConfig, ClientOptions
from videosdk import CustomVideoTrack, CustomAudioTrack

logger = logging.getLogger(__name__)

VIDEOSDK_AUDIO_SAMPLE_RATE = 48000
ANAM_INPUT_SAMPLE_RATE = 24000

anam_input_resampler = AudioResampler(
    format="s16", layout="mono", rate=ANAM_INPUT_SAMPLE_RATE
)
videosdk_output_resampler = AudioResampler(
    format="s16", layout="stereo", rate=VIDEOSDK_AUDIO_SAMPLE_RATE
)

class AnamAudioTrack(CustomAudioTrack):
    def __init__(self):
        super().__init__()
        self.kind = "audio"
        self.queue = asyncio.Queue()
        self._readyState = "live"

    def interrupt(self):
        """Clear the audio buffer."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # def on_last_audio_byte(self):
    #     """Called when the last byte of audio for a response has been consumed."""
    #     pass

    @property
    def readyState(self):
        return self._readyState

    async def recv(self) -> AudioFrame:
        """Return next audio frame to VideoSDK."""
        try:
            if self.readyState != "live":
                raise Exception("Track not live")
            frame = await self.queue.get()
            return frame

        except Exception:
            traceback.print_exc()
            return self._create_silence_frame()

    def add_frame(self, frame: AudioFrame):
        """Add frame from Anam stream - add AudioFrame directly to buffer with quality validation"""
        if frame is None:
            return
        try:
            for resampled_frame in videosdk_output_resampler.resample(frame):
                self.queue.put_nowait(resampled_frame)
        except asyncio.QueueEmpty:
            pass
        except Exception as e:
            logger.error(f"Error adding Anam audio frame: {e}")

    def stop(self):
        super().stop()
        self._readyState = "ended"


class AnamVideoTrack(CustomVideoTrack):
    def __init__(self):
        super().__init__()
        self.kind = "video"
        self.queue = asyncio.Queue()
        self._readyState = "live"

    @property
    def readyState(self):
        return self._readyState

    async def recv(self) -> VideoFrame:
        """Return next video frame to VideoSDK."""
        return await self.queue.get()

    def add_frame(self, frame: VideoFrame):
        """Add frame from Anam stream."""
        if self._readyState == "live":
            self.queue.put_nowait(frame)

    def stop(self):
        super().stop()
        self._readyState = "ended"


class AnamAvatar:
    def __init__(
        self,
        api_key: str,
        persona_id: Optional[str] = None,
        persona_config: Optional[PersonaConfig] = None,
    ):
        """Initialize the Anam Avatar plugin.

        Args:
            api_key (str): The Anam API key.
            persona_id (str, optional): The ID of the persona to use.
            persona_config (PersonaConfig, optional): Full persona configuration.
        """
        self.api_key = api_key
        
        if persona_id:
            self.persona_config = PersonaConfig(
                persona_id=persona_id,
                enable_audio_passthrough=True,
                llm_id="CUSTOMER_CLIENT_V1"
            )
        else:
            raise ValueError("Either persona_id or persona_config must be provided")

        self.client: Optional[AnamClient] = None
        self.session = None
        self.audio_track: Optional[AnamAudioTrack] = None
        self.video_track: Optional[AnamVideoTrack] = None
        
        self.run = True
        self._stopping = False
        self._input_stream = None
        self._tasks = []

    async def connect(self):
        """Connect to Anam and start processing streams."""
        try:
            client_options = ClientOptions()
            
            self.client = AnamClient(
                api_key=self.api_key,
                persona_config=self.persona_config,
                options=client_options
            )
            
            self.session = await self.client.connect_async()
            
            self.audio_track = AnamAudioTrack()
            self.video_track = AnamVideoTrack()
            
            input_config = AgentAudioInputConfig(
                sample_rate=ANAM_INPUT_SAMPLE_RATE,
                channels=1,
                encoding="pcm_s16le"
            )
            self._input_stream = self.session.create_agent_audio_input_stream(input_config)
            
            self._tasks.append(asyncio.create_task(self._process_video_frames()))
            self._tasks.append(asyncio.create_task(self._process_audio_frames()))
            
            logger.info("Connected to Anam Avatar")
            
        except Exception as e:
            logger.error(f"Failed to connect to Anam: {e}")
            raise e

    async def _process_video_frames(self):
        """Process video frames from Anam."""
        if not self.session:
            return

        try:
            async for frame in self.session.video_frames():
                if not self.run or self._stopping:
                    break
                if frame:
                    self.video_track.add_frame(frame)
        except Exception as e:
            logger.error(f"Anam: Video processing error: {e}")
        finally:
            logger.info("Anam video processing stopped")

    async def _process_audio_frames(self):
        """Process audio frames from Anam."""
        if not self.session:
            return

        try:
            async for frame in self.session.audio_frames():
                if not self.run or self._stopping:
                    break
                if frame:
                    self.audio_track.add_frame(frame)
        except Exception as e:
            logger.error(f"Anam: Audio processing error: {e}")
        finally:
            logger.info("Anam audio processing stopped")

    async def handle_audio_input(self, audio_data: bytes):
        """Handle audio input from VideoSDK pipeline and send to Anam."""
        if not self.run or self._stopping or not self._input_stream:
            return
            
        try:
            
            if len(audio_data) % 2 != 0:
                audio_data = audio_data + b"\x00"

            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            input_frame = AudioFrame.from_ndarray(
                audio_array.reshape(1, -1), format="s16", layout="mono"
            )
            
            input_frame.sample_rate = 24000 
            resampled_frames = anam_input_resampler.resample(input_frame)
            
            for frame in resampled_frames:
                resampled_data = frame.to_ndarray().tobytes()
                await self._input_stream.send_audio_chunk(resampled_data)

        except Exception as e:
            logger.error(f"Error processing/sending Anam audio data: {e}")

    async def interrupt(self):
        """Interrupt the avatar."""
        if self.session:
            try:
                await self.session.interrupt()
                
                if self.audio_track:
                    self.audio_track.interrupt()

            except Exception as e:
                logger.error(f"Error interrupting Anam: {e}")

    async def aclose(self):
        """Close the avatar plugin."""
        if self._stopping:
            return
        self._stopping = True
        self.run = False

        for task in self._tasks:
            task.cancel()
        
        if self.session:
            try:
                await self.session.close()
            except Exception:
                pass        

        if self.audio_track:
            self.audio_track.stop()
        if self.video_track:
            self.video_track.stop()
