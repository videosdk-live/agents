import time
import asyncio
import json
import fractions
from dataclasses import dataclass
from typing import Awaitable, Optional, Tuple
from httpx import AsyncClient
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceServer,
    RTCConfiguration,
)
from aiortc.mediastreams import MediaStreamTrack, VideoStreamTrack, AudioStreamTrack
from av import VideoFrame
from av.audio.resampler import AudioResampler
import websockets.asyncio.client
import numpy as np
from av import AudioFrame
from vsaiortc.mediastreams import MediaStreamError

from videosdk import CustomVideoTrack, CustomAudioTrack
from videosdk.agents.realtime_base_model import RealtimeBaseModel
import logging
logger = logging.getLogger(__name__)

# --- Constants ---
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH = 2
AUDIO_FRAME_DURATION_S = 0.02
AUDIO_SAMPLES_PER_FRAME = int(AUDIO_FRAME_DURATION_S * AUDIO_SAMPLE_RATE)
AUDIO_CHUNK_SIZE = AUDIO_SAMPLES_PER_FRAME * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH
AUDIO_TIME_BASE_FRACTION = fractions.Fraction(1, AUDIO_SAMPLE_RATE)
AUDIO_REsampler_RATE = 16000

VIDEO_FRAME_RATE = 30
VIDEO_TIME_BASE = 90000

DEFAULT_SIMLI_HTTP_URL = "https://api.simli.ai"
DEFAULT_SIMLI_WS_URL = "wss://api.simli.ai"


audioResampler = AudioResampler(format="s16", layout="mono", rate=AUDIO_REsampler_RATE)


class SimliAudioTrack(CustomAudioTrack):
    def __init__(self, loop):
        super().__init__()
        self.kind = "audio"
        self.loop = loop
        self._timestamp = 0
        self.queue = asyncio.Queue(maxsize=20)
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

    async def add_new_bytes(self, audio_data: bytes):
        """Required method for compatibility with existing audio track interface"""
        self.audio_data_buffer += audio_data

        while len(self.audio_data_buffer) >= self.chunk_size:
            chunk = self.audio_data_buffer[:self.chunk_size]
            self.audio_data_buffer = self.audio_data_buffer[self.chunk_size:]
            try:
                audio_frame = self.buildAudioFrames(chunk)
                if self.queue.full():
                    # Drop oldest frame
                    self.queue.get_nowait()
                self.queue.put_nowait(audio_frame)
            except Exception as e:
                break

    def buildAudioFrames(self, chunk: bytes) -> AudioFrame:
        if len(chunk) != self.chunk_size:
            logger.warning(f"Incorrect Simli chunk size received {len(chunk)}, expected {self.chunk_size}")

        if len(chunk) % 2 != 0:
            chunk = chunk + b'\x00'

        data = np.frombuffer(chunk, dtype=np.int16)
        expected_samples = self.samples * self.channels
        if len(data) != expected_samples:
            logger.warning(f"Incorrect number of samples in Simli chunk {len(data)}, expected {expected_samples}")

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
        """Return next audio frame to VideoSDK."""
        try:
            if self.readyState != "live":
                raise MediaStreamError

            if self._start_time is None:
                self._start_time = time.time()
                self._timestamp = 0
            else:
                self._timestamp += self.samples
            wait = self._start_time + (self._timestamp / self.sample_rate) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

            pts = self._timestamp
            time_base = self.time_base_fraction

            try:
                frame = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                frame = self._create_silence_frame()

            frame.pts = pts
            frame.time_base = time_base
            frame.sample_rate = self.sample_rate

            return frame

        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._create_silence_frame()
    
    def _create_silence_frame(self) -> AudioFrame:
        """Create a properly formatted silence frame"""
        layout = "mono" if self.channels == 1 else "stereo"
        frame = AudioFrame(format="s16", layout=layout, samples=self.samples)
        for p in frame.planes:
            p.update(bytes(p.buffer_size))
        frame.sample_rate = self.sample_rate
        return frame
            
    async def cleanup(self):
        self.interrupt()
        self.stop()

    def add_frame(self, frame: AudioFrame):
        """Add frame from Simli stream - add AudioFrame directly to buffer with quality validation"""
        if frame is None:
            return
        try:
            if hasattr(frame, 'sample_rate') and frame.sample_rate != self.sample_rate:
                frame.sample_rate = self.sample_rate
            
            try:
                frame_array = frame.to_ndarray()
                if len(frame_array.shape) == 2 and frame_array.shape[0] != self.channels:
                    print(f"Warning: Frame channels {frame_array.shape[0]} != expected {self.channels}")
                
                if frame_array.dtype != np.int16:
                    if frame_array.dtype == np.float32 or frame_array.dtype == np.float64:
                        frame_array = (frame_array * 32767).astype(np.int16)
                    else:
                        frame_array = frame_array.astype(np.int16)
                    
                    layout = "mono" if self.channels == 1 else "stereo"
                    corrected_frame = AudioFrame.from_ndarray(frame_array, format="s16", layout=layout)
                    corrected_frame.sample_rate = self.sample_rate
                    frame = corrected_frame
                    
            except Exception as format_error:
                frame = self._create_silence_frame()
            
            try:
                if self.queue.full():
                    self.queue.get_nowait()
                self.queue.put_nowait(frame)
            except asyncio.QueueEmpty:
                pass
            except asyncio.QueueFull:
                logger.warning("Simli: Audio frame queue is full. Frame dropped.")
                
        except Exception as e:  
            logger.error(f"Error adding Simli audio frame: {e}")
            try:
                array = frame.to_ndarray()
            except:
                pass


class SimliVideoTrack(CustomVideoTrack):
    def __init__(self):
        super().__init__()
        self.kind = "video"
        self.queue = asyncio.Queue(maxsize=2)
        self._timestamp = 0
        self._start_time = None
        self._frame_count = 0
        self._readyState = "live"
        self._frame_rate = VIDEO_FRAME_RATE
        self._frame_duration = 1.0 / self._frame_rate
        self._shared_start_time = None

    @property
    def readyState(self):
        return self._readyState

    async def recv(self) -> VideoFrame:
        frame = await self.queue.get()
        
        if self._start_time is None:
            self._start_time = self._shared_start_time if self._shared_start_time else time.time()
            self._timestamp = 0
        
        current_time = time.time()
        elapsed = current_time - self._start_time
        self._timestamp = int(elapsed * VIDEO_TIME_BASE)
        
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, VIDEO_TIME_BASE)
        
        self._frame_count += 1

        return frame

    def add_frame(self, frame: VideoFrame):
        # Keep only the latest frame by clearing the queue before adding a new one.
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        try:
            self.queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass


@dataclass
class SimliConfig:
    apiKey: str
    faceId: str = "0c2b8b04-5274-41f1-a21c-d5c98322efa9"
    syncAudio: bool = True
    handleSilence: bool = True
    maxSessionLength: int = 1800  # 30 minutes 
    maxIdleTime: int = 300       # 5 minutes

class SimliAvatar:
    def __init__(self, config: SimliConfig, simli_url: str = DEFAULT_SIMLI_HTTP_URL):
        super().__init__()
        self.config = config
        self._stream_start_time = None
        self.video_track = SimliVideoTrack()
        self.audio_track = None  
        self.pc: Optional[RTCPeerConnection] = None
        self.simli_http_url = simli_url
        self.simli_ws_url = simli_url.replace("http", "ws") if simli_url else DEFAULT_SIMLI_WS_URL
        self.run = True
        self._is_speaking = False
        self._speech_timeout_task = None
        self.ws: Optional[websockets.asyncio.client.WebSocketClientProtocol] = None
        self.dc = None
        self.ready = asyncio.Event()
        self._avatar_speaking = False
        self._last_reconnect_attempt = 0
        self._message_handler_task = None
        self._retry_count = 3
        self._last_error = None
        self._stopping = False
        self._keep_alive_task = None
        self._last_audio_time = 0

    async def connect(self):
        loop = asyncio.get_event_loop()
        self.audio_track = SimliAudioTrack(loop)
        
        if self._stream_start_time is None:
            self._stream_start_time = time.time()
            self.video_track._shared_start_time = self._stream_start_time
            self.audio_track._shared_start_time = self._stream_start_time
        
        await self._initialize_connection()
        
        if hasattr(self.video_track, 'start'):
            self.video_track.start()
        if hasattr(self.audio_track, 'start'):
            self.audio_track.start()
        
        self._last_audio_time = time.time()
        self._keep_alive_task = asyncio.create_task(self._keep_alive_loop())

    async def _initialize_connection(self):
        """Initialize connection with retry logic """
        if self._retry_count == 0:
            raise Exception(f"Failed to connect to Simli servers. Last error: {self._last_error}")
        
        try:
            await self._start_session()
            await self._wait_for_data_channel()
            await self.sendSilence()
        except Exception as e:
            self._last_error = e
            self._retry_count -= 1
            if self._retry_count > 0:
                await asyncio.sleep(2) 
                await self._initialize_connection()
            else:
                raise

    async def _wait_for_data_channel(self):
        """Wait for the data channel to be open"""
        max_wait = 10  
        waited = 0
        while waited < max_wait:
            if self.dc and self.dc.readyState == "open":
                return
            await asyncio.sleep(0.1)
            waited += 0.1
        raise Exception("Data channel did not open within timeout")

    async def _start_session(self):
        try:
            await self._cleanup_connections()
            
            session_token = await self._http_start_session()
            await self._negotiate_webrtc_via_ws(session_token)
        except Exception as e:
            logger.error(f"Error starting Simli session: {e}")
            raise

    async def _http_start_session(self) -> str:
        """Sends a request to start a session and returns the session token."""
        config_json = self.config.__dict__
        async with AsyncClient() as client:
            resp = await client.post(
                f"{self.simli_http_url}/startAudioToVideoSession", json=config_json
            )
            resp.raise_for_status()
            return resp.json()["session_token"]

    async def _negotiate_webrtc_via_ws(self, session_token: str):
        """Sets up WebRTC connection and negotiates it through WebSocket."""
        ice_servers = [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))

        self.pc.addTransceiver("audio", direction="recvonly")
        self.pc.addTransceiver("video", direction="recvonly")
        self.pc.on("track", self._on_track)
        self.dc = self.pc.createDataChannel("datachannel", ordered=True)

        await self.pc.setLocalDescription(await self.pc.createOffer())
        while self.pc.iceGatheringState != "complete":
            await asyncio.sleep(0.01)

        json_offer = self.pc.localDescription.__dict__

        ws_url = f"{self.simli_ws_url}/StartWebRTCSession"
        self.ws = await websockets.asyncio.client.connect(ws_url)
        await self.ws.send(json.dumps(json_offer))
        await self.ws.recv()  # ACK

        answer_str = await self.ws.recv()
        await self.ws.send(session_token)
        await self.ws.recv()  # ACK

        ready_msg = await self.ws.recv()
        if ready_msg != "START":
            raise Exception(f"Failed to start Simli session. Expected START, got {ready_msg}")

        self.ready.set()
        await self.pc.setRemoteDescription(RTCSessionDescription(**json.loads(answer_str)))

        if self._message_handler_task:
            self._message_handler_task.cancel()
        self._message_handler_task = asyncio.create_task(self._handle_ws_messages())

    async def _cleanup_connections(self):
        """Clean up existing connections before creating new ones"""
        if self._message_handler_task and not self._message_handler_task.done():
            self._message_handler_task.cancel()
            try:
                await self._message_handler_task
            except asyncio.CancelledError:
                pass
        
        if self.ws:
            try:
                await self.ws.close()
            except:
                pass
            self.ws = None
        
        if self.pc:
            try:
                await self.pc.close()
            except:
                pass
            self.pc = None
        
        self.ready.clear()
        self._is_speaking = False
        if self._speech_timeout_task and not self._speech_timeout_task.done():
            self._speech_timeout_task.cancel()

    def _on_track(self, track: MediaStreamTrack):
        if track.kind == "video":
            self.video_receiver_track = track
            asyncio.create_task(self._process_video_frames())
        elif track.kind == "audio":
            self.audio_receiver_track = track
            asyncio.create_task(self._process_audio_frames())

    async def _process_video_frames(self):
        """Simple video frame processing for real-time playback"""
        frame_count = 0
        while self.run and not self._stopping:
            try:
                if not hasattr(self, 'video_receiver_track'):
                    await asyncio.sleep(0.1)
                    continue
                    
                frame = await asyncio.wait_for(self.video_receiver_track.recv(), timeout=2.0)
                if frame is None:
                    continue
                    
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
                if not hasattr(self, 'audio_receiver_track'):
                    await asyncio.sleep(0.1)
                    continue
                    
                frame = await asyncio.wait_for(self.audio_receiver_track.recv(), timeout=2.0)
                if frame is None:   
                    logger.warning("Simli: Received None audio frame, continuing...")
                    continue
                    
                frame_count += 1
                
                try:
                    if not hasattr(frame, 'sample_rate') or frame.sample_rate != AUDIO_SAMPLE_RATE:
                        frame.sample_rate = AUDIO_SAMPLE_RATE
                        
                    self.audio_track.add_frame(frame)
                    
                except Exception as frame_error:    
                    logger.error(f"Simli: Error processing audio frame #{frame_count}: {frame_error}")
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
        await self.ready.wait()
        silence_data = (0).to_bytes(2, "little") * int(16000 * duration)
        try:
            await self._send_audio_data(silence_data)
        except Exception as e:
            logger.error(f"Error sending bootstrap silence: {e}")

    async def _handle_ws_messages(self):
        """Handle WebSocket messages """
        try:
            while self.run and not self._stopping:
                await self.ready.wait()
                message = await self.ws.recv()
                
                if message == "STOP":
                    self.run = False
                    await self.aclose()
                    break
                    
                elif "error" in message.lower():
                    self.run = False
                    await self.aclose()
                    break
                    
                elif message == "SPEAK":
                    self._avatar_speaking = True
                    
                elif message == "SILENT": 
                    self._avatar_speaking = False
                    
                elif message != "ACK":
                    pass
                    
        except Exception as e:
            logger.error(f"Error in Simli websocket message handler: {e}")
            if not self._stopping:
                self.run = False
                await self.aclose()

    async def _speech_timeout_handler(self):
        try:
            await asyncio.sleep(0.2)
            if self._is_speaking:
                await self.ws.send("AUDIO_END")
                self._is_speaking = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in speech timeout handler: {e}")

    async def handle_audio_input(self, audio_data: bytes):
        if not self.run or self._stopping:
            return
            
        if self.ws and self.ready.is_set():
            try:
                if len(audio_data) % 2 != 0:
                    audio_data = audio_data + b'\x00'
                
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                input_frame = AudioFrame.from_ndarray(
                    audio_array.reshape(1, -1), format="s16", layout="mono"
                )
                input_frame.sample_rate = 24000
                
                resampled_frames = audioResampler.resample(input_frame)
                if resampled_frames:
                    resampled_data = resampled_frames[0].to_ndarray().tobytes()
                    
                    await self._send_audio_data(resampled_data)
                    
                    self._last_audio_time = time.time()

            except Exception as e:
                logger.error(f"Error processing/sending audio data: {e}")
        else:
            logger.error(f"Simli: Cannot send audio - ws available: {self.ws is not None}, ready: {self.ready.is_set()}")

    async def _send_audio_data(self, data: bytes):
        """Send audio data via WebSocket to simli """
        try:
            for i in range(0, len(data), 6000):
                chunk = data[i:i + 6000]
                await self.ws.send(chunk)
        except Exception as e:
            logger.error(f"Error sending audio data via WebSocket: {e}")
    
    async def send_message(self, message: str):
        pass

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
            if self.ws:
                await self.ws.send(b"DONE")
        except:
            pass
        
        await self._cleanup_connections()

    def set_agent(self, agent):
        pass

    async def _keep_alive_loop(self):
        """Send periodic keep-alive audio to maintain Simli session"""
        silence_duration = 0.1875 
        silence_data = (0).to_bytes(2, "little") * int(16000 * silence_duration)
        
        while self.run and not self._stopping:
            try:
                current_time = time.time()
                if current_time - self._last_audio_time > 5.0:
                    if self.ws and self.ready.is_set():
                        try:
                            await self._send_audio_data(silence_data)
                            self._last_audio_time = current_time
                        except Exception as e:
                            print(f"Simli: Keep-alive send failed: {e}")
                
                await asyncio.sleep(3.0) 
                
            except Exception as e:
                if not self.run or self._stopping:
                    break
                await asyncio.sleep(1.0)
        