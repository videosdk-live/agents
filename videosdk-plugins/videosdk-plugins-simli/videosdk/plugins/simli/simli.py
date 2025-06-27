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

audioResampler = AudioResampler(format="s16", layout="mono", rate=16000)


class SimliAudioTrack(CustomAudioTrack):
    def __init__(self, loop):
        super().__init__()
        self.kind = "audio"
        self.loop = loop
        self._timestamp = 0
        self.queue = asyncio.Queue(maxsize=20)
        self.audio_data_buffer = bytearray()
        self.frame_time = 0
        self.sample_rate = 48000  
        self.channels = 1  
        self.sample_width = 2
        self.time_base_fraction = fractions.Fraction(1, self.sample_rate)
        self.samples = int(0.02 * self.sample_rate) 
        self.chunk_size = int(self.samples * self.channels * self.sample_width)
        self._start_time = None
        self._shared_start_time = None  
        self._frame_duration = 0.02
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
                print(f"Error building Simli audio frame: {e}")
                break

    def buildAudioFrames(self, chunk: bytes) -> AudioFrame:
        if len(chunk) != self.chunk_size:
            print(f"Warning: Incorrect Simli chunk size received {len(chunk)}, expected {self.chunk_size}")

        data = np.frombuffer(chunk, dtype=np.int16)
        expected_samples = self.samples * self.channels
        if len(data) != expected_samples:
            print(f"Warning: Incorrect number of samples in Simli chunk {len(data)}, expected {expected_samples}")

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
        """Return next audio frame to VideoSDK. Simplified to follow the same
        timing logic used by VideoSDK's reference `CustomAudioStreamTrack` so
        that the SDK continues to pull frames reliably. The earlier, more
        complex implementation occasionally caused the track's internal state
        to stall resulting in silence after the first few packets were
        forwarded."""
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
            print("Error while creating Simli->VideoSDK frame", e)
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
                print(f"Warning: Incoming frame sample rate {frame.sample_rate} doesn't match expected {self.sample_rate}")
                frame.sample_rate = self.sample_rate
            
            try:
                frame_array = frame.to_ndarray()
                if len(frame_array.shape) == 2 and frame_array.shape[0] != self.channels:
                    print(f"Warning: Frame channels {frame_array.shape[0]} != expected {self.channels}")
                
                if frame_array.dtype != np.int16:
                    print(f"Warning: Converting frame dtype from {frame_array.dtype} to int16")
                    if frame_array.dtype == np.float32 or frame_array.dtype == np.float64:
                        frame_array = (frame_array * 32767).astype(np.int16)
                    else:
                        frame_array = frame_array.astype(np.int16)
                    
                    layout = "mono" if self.channels == 1 else "stereo"
                    corrected_frame = AudioFrame.from_ndarray(frame_array, format="s16", layout=layout)
                    corrected_frame.sample_rate = self.sample_rate
                    frame = corrected_frame
                    
            except Exception as format_error:
                print(f"Warning: Frame format validation error: {format_error}")
                frame = self._create_silence_frame()
            
            try:
                if self.queue.full():
                    # Drop the oldest frame to reduce latency.
                    self.queue.get_nowait()
                self.queue.put_nowait(frame)
            except asyncio.QueueEmpty:
                # Can happen in a race condition if queue is emptied
                # between full() check and get_nowait(). Safe to ignore.
                pass
            except asyncio.QueueFull:
                # This should ideally not be reached.
                print("Simli: Audio frame queue is full. Frame dropped.")
                
        except Exception as e:
            print(f"Error adding Simli audio frame: {e}")
            try:
                array = frame.to_ndarray()
                print(f"Debug: Frame shape: {array.shape}, dtype: {array.dtype}")
            except:
                pass


class SimliVideoTrack(CustomVideoTrack):
    def __init__(self):
        super().__init__()
        self.kind = "video"
        self.queue = asyncio.Queue(maxsize=5)
        self._timestamp = 0
        self._start_time = None
        self._frame_count = 0
        self._readyState = "live"
        self._frame_rate = 30
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
        self._timestamp = int(elapsed * 90000)
        
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, 90000)
        
        self._frame_count += 1
        if self._frame_count % 30 == 1:
            print(f"SimliVideoTrack: Sending frame #{self._frame_count} to VideoSDK - {frame.width}x{frame.height}")
        
        return frame

    def add_frame(self, frame: VideoFrame):
        while self.queue.qsize() > 1:
            try:
                self.queue.get_nowait()
            except:
                break
        
        try:
            self.queue.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(frame)
            except:
                pass


@dataclass
class SimliConfig:
    apiKey: str
    faceId: str
    syncAudio: bool = True
    handleSilence: bool = True
    maxSessionLength: int = 1800  # 30 minutes 
    maxIdleTime: int = 300       # 5 minutes

class SimliAvatar:
    def __init__(self, api_key: str, face_id: str, simli_url: str = "https://api.simli.ai"):
        super().__init__()
        self.config = SimliConfig(
            apiKey=api_key, 
            faceId=face_id,
            maxSessionLength=1800, 
            maxIdleTime=120         
        )
        self._stream_start_time = None
        self.video_track = SimliVideoTrack()
        self.audio_track = None  
        self.pc: RTCPeerConnection = None
        self.simli_http_url = simli_url
        self.simli_ws_url = simli_url.replace("http", "ws")
        self.run = True
        self._is_speaking = False
        self._speech_timeout_task = None
        self.ws = None
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
        
        print(f"Simli: Avatar tracks ready - Video: {self.video_track is not None}, Audio: {self.audio_track is not None}")
        
        if hasattr(self.video_track, 'start'):
            self.video_track.start()
        if hasattr(self.audio_track, 'start'):
            self.audio_track.start()
        
        self._last_audio_time = time.time()
        self._keep_alive_task = asyncio.create_task(self._keep_alive_loop())

    async def _initialize_connection(self):
        """Initialize connection with retry logic similar to reference implementation"""
        if self._retry_count == 0:
            raise Exception(f"Failed to connect to Simli servers. Last error: {self._last_error}")
        
        try:
            await self._start_session()
            await self._wait_for_data_channel()
            await self.sendSilence()
            print("Simli connection established successfully")
        except Exception as e:
            print(f"Simli connection attempt failed: {e}")
            self._last_error = e
            self._retry_count -= 1
            if self._retry_count > 0:
                print(f"Retrying connection... ({self._retry_count} attempts left)")
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
                print("Simli: Data channel is now open")
                return
            await asyncio.sleep(0.1)
            waited += 0.1
        raise Exception("Data channel did not open within timeout")

    async def _start_session(self):
        try:
            await self._cleanup_connections()
            
            config_json = self.config.__dict__
            async with AsyncClient() as client:
                resp = await client.post(
                    f"{self.simli_http_url}/startAudioToVideoSession", json=config_json
                )
                resp.raise_for_status()
                session_token = resp.json()["session_token"]

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
            
            self.ws = await websockets.asyncio.client.connect(f"{self.simli_ws_url}/StartWebRTCSession")
            await self.ws.send(json.dumps(json_offer))
            await self.ws.recv()  # ACK

            answer = await self.ws.recv()
            await self.ws.send(session_token)
            await self.ws.recv()  # ACK
            
            ready = await self.ws.recv()
            if ready != "START":
                raise Exception("Failed to start Simli session")
            
            self.ready.set()
            await self.pc.setRemoteDescription(RTCSessionDescription(**json.loads(answer)))
            
            if self._message_handler_task:
                self._message_handler_task.cancel()
            self._message_handler_task = asyncio.create_task(self._handle_ws_messages())
            print("Simli session started.")

        except Exception as e:
            print(f"Error starting Simli session: {e}")
            raise

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
        print("Simli: Starting video frame processing task")
        frame_count = 0
        while self.run and not self._stopping:
            try:
                if not hasattr(self, 'video_receiver_track'):
                    await asyncio.sleep(0.1)
                    continue
                    
                frame = await asyncio.wait_for(self.video_receiver_track.recv(), timeout=2.0)
                if frame is None:
                    print("Simli: Received None video frame, continuing...")
                    continue
                    
                frame_count += 1
                if frame_count % 30 == 1:  
                    print(f"Simli: Processing video frame #{frame_count}: {frame.width}x{frame.height}")
                
                self.video_track.add_frame(frame)
                
            except asyncio.TimeoutError:
                if self.run and not self._stopping:
                    print("Simli: Video frame timeout, continuing...")
                    continue
                else:
                    break
            except Exception as e:
                print(f"Simli: Video processing error: {e}")
                if not self.run or self._stopping:
                    break
                await asyncio.sleep(0.1)
                continue
        print("Simli: Video frame processing task ended")

    async def _process_audio_frames(self):
        """Simple audio frame processing for real-time playback"""
        print("Simli: Starting audio frame processing task")
        frame_count = 0
        while self.run and not self._stopping:
            try:
                if not hasattr(self, 'audio_receiver_track'):
                    await asyncio.sleep(0.1)
                    continue
                    
                frame = await asyncio.wait_for(self.audio_receiver_track.recv(), timeout=2.0)
                if frame is None:
                    print("Simli: Received None audio frame, continuing...")
                    continue
                    
                frame_count += 1
                
                try:
                    frame_samples = frame.samples if hasattr(frame, 'samples') else 'unknown'
                    frame_rate = frame.sample_rate if hasattr(frame, 'sample_rate') else 'unknown'
                    
                    if frame_count % 50 == 1:
                        print(f"Simli: Processing audio frame #{frame_count}: {frame_samples} samples @ {frame_rate}Hz")
                    
                    if not hasattr(frame, 'sample_rate') or frame.sample_rate != 48000:
                        frame.sample_rate = 48000
                        
                    self.audio_track.add_frame(frame)
                    
                except Exception as frame_error:
                    print(f"Simli: Error processing audio frame #{frame_count}: {frame_error}")
                    continue
                
            except asyncio.TimeoutError:
                if self.run and not self._stopping:
                    print("Simli: Audio frame timeout, continuing...")
                    continue
                else:
                    break
            except Exception as e:
                print(f"Simli: Audio processing error: {e}")
                if not self.run or self._stopping:
                    break
                await asyncio.sleep(0.1)
                continue
        print("Simli: Audio frame processing task ended")

    async def sendSilence(self, duration: float = 0.1875):
        """Send silence to bootstrap the connection - following reference implementation"""
        await self.ready.wait()
        silence_data = (0).to_bytes(2, "little") * int(16000 * duration)
        print(f"Simli: Sending {len(silence_data)} bytes of silence to bootstrap connection via WebSocket")
        try:
            await self._send_audio_data(silence_data)
            print("Simli: Bootstrap silence sent successfully via WebSocket")
        except Exception as e:
            print(f"Error sending bootstrap silence: {e}")

    async def _handle_ws_messages(self):
        """Handle WebSocket messages - similar to reference implementation"""
        try:
            while self.run and not self._stopping:
                await self.ready.wait()
                message = await self.ws.recv()
                print(f"Simli: Received WebSocket message: {message}")
                
                if message == "STOP":
                    print("Simli: Received STOP message - session timeout reached")
                    print("Simli: Stopping session as per Simli server request")
                    self.run = False
                    await self.aclose()
                    break
                    
                elif "error" in message.lower():
                    print(f"Simli: Error message received: {message}")
                    self.run = False
                    await self.aclose()
                    break
                    
                elif message == "SPEAK":
                    print("Simli: Avatar is speaking")
                    self._avatar_speaking = True
                    
                elif message == "SILENT": 
                    print("Simli: Avatar is silent")
                    self._avatar_speaking = False
                    
                elif message != "ACK":
                    print(f"Simli: Unknown message: {message}")
                    
        except Exception as e:
            print(f"Error in Simli websocket message handler: {e}")
            if not self._stopping:
                self.run = False
                await self.aclose()

    async def _speech_timeout_handler(self):
        try:
            await asyncio.sleep(0.2)
            if self._is_speaking:
                print("Simli: Sending AUDIO_END message")
                await self.ws.send("AUDIO_END")
                self._is_speaking = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in speech timeout handler: {e}")

    async def handle_audio_input(self, audio_data: bytes):
        if not self.run or self._stopping:
            return
            
        if self.ws and self.ready.is_set():
            try:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                input_frame = AudioFrame.from_ndarray(
                    audio_array.reshape(1, -1), format="s16", layout="mono"
                )
                input_frame.sample_rate = 24000
                
                resampled_frames = audioResampler.resample(input_frame)
                if resampled_frames:
                    resampled_data = resampled_frames[0].to_ndarray().tobytes()
                    
                    print(f"Simli: Sending {len(resampled_data)} bytes of audio data via WebSocket")
                    await self._send_audio_data(resampled_data)
                    
                    self._last_audio_time = time.time()
                else:
                    print("Simli: Failed to resample audio")
            except Exception as e:
                print(f"Error processing/sending audio data: {e}")
        else:
            print(f"Simli: Cannot send audio - ws available: {self.ws is not None}, ready: {self.ready.is_set()}")

    async def _send_audio_data(self, data: bytes):
        """Send audio data via WebSocket to simli """
        try:
            for i in range(0, len(data), 6000):
                chunk = data[i:i + 6000]
                await self.ws.send(chunk)
        except Exception as e:
            print(f"Error sending audio data via WebSocket: {e}")
    
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
        print("Simli connection closed.")

    def set_agent(self, agent):
        pass

    async def _keep_alive_loop(self):
        """Send periodic keep-alive audio to maintain Simli session"""
        print("Simli: Starting keep-alive loop")
        silence_duration = 0.1875 
        silence_data = (0).to_bytes(2, "little") * int(16000 * silence_duration)
        
        while self.run and not self._stopping:
            try:
                current_time = time.time()
                if current_time - self._last_audio_time > 5.0:
                    if self.ws and self.ready.is_set():
                        try:
                            print("Simli: Sending keep-alive silence via WebSocket")
                            await self._send_audio_data(silence_data)
                            self._last_audio_time = current_time
                        except Exception as e:
                            print(f"Simli: Keep-alive send failed: {e}")
                
                await asyncio.sleep(3.0) 
                
            except Exception as e:
                print(f"Simli: Keep-alive loop error: {e}")
                if not self.run or self._stopping:
                    break
                await asyncio.sleep(1.0)
        
        print("Simli: Keep-alive loop ended")
