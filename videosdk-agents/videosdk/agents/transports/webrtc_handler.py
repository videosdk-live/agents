import asyncio
import logging
import json
import time
import fractions
import numpy as np
import av

try:
    import aiohttp
    from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
except ImportError:
    aiohttp = None
    RTCPeerConnection = None
    MediaStreamTrack = object

from .base import BaseTransportHandler

logger = logging.getLogger(__name__)

class WebRTCOutputTrack(MediaStreamTrack):
    """
    A MediaStreamTrack compatible with aiortc that buffers audio bytes 
    pushed to it and yields them via recv().
    """
    kind = "audio"

    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.sample_rate = 24000
        self.channels = 1
        self.samples_per_frame = int(0.02 * self.sample_rate)
        self.frame_size = self.samples_per_frame * self.channels * 2
        
        self.audio_buffer = bytearray()
        self.frame_queue = asyncio.Queue()
        self._start_time = None
        self._timestamp = 0
        self.sinks = []
        self._last_audio_callback = None
        self._is_speaking = False

    def add_sink(self, sink):
        """Mock add_sink for pipeline compatibility"""
        pass

    def remove_sink(self, sink):
        """Mock remove_sink for pipeline compatibility"""
        pass
        
    def interrupt(self):
        """Clear all buffers"""
        logger.info("WebRTCOutputTrack interrupted")
        self.audio_buffer = bytearray()
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._is_speaking = False

    def on_last_audio_byte(self, callback):
        """Set callback for when the final audio byte of synthesis is produced"""
        self._last_audio_callback = callback

    async def add_new_bytes(self, audio_data: bytes):
        """Called by the pipeline to push audio"""
        self.audio_buffer += audio_data
        
        # Process buffer into frames
        while len(self.audio_buffer) >= self.frame_size:
            chunk = self.audio_buffer[:self.frame_size]
            self.audio_buffer = self.audio_buffer[self.frame_size:]
            
            try:
                frame = self._create_audio_frame(chunk)
                await self.frame_queue.put(frame)
            except Exception as e:
                logger.error(f"Error creating audio frame: {e}")

    def _create_audio_frame(self, data_bytes):
        data_np = np.frombuffer(data_bytes, dtype=np.int16)
        data_np = data_np.reshape(-1, self.channels)
        
        frame = av.AudioFrame.from_ndarray(data_np.T, format='s16', layout='mono')
        frame.sample_rate = self.sample_rate
        return frame

    async def recv(self):
        """Called by aiortc to pull frames"""
        if self.readyState != "live":
            raise Exception("Track is not live")

        if self._start_time is None:
            self._start_time = time.time()
            self._timestamp = 0

        samples = self.samples_per_frame
        target_time = self._start_time + (self._timestamp / self.sample_rate)
        wait_time = target_time - time.time()
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                self._is_speaking = True
            else:
                if self._is_speaking:
                    self._is_speaking = False
                    if self._last_audio_callback:
                        asyncio.create_task(self._last_audio_callback())
                
                data = np.zeros(self.samples_per_frame * self.channels, dtype=np.int16)
                data = data.reshape(-1, self.channels)
                frame = av.AudioFrame.from_ndarray(data.T, format='s16', layout='mono')
                frame.sample_rate = self.sample_rate
        except Exception:
             data = np.zeros(self.samples_per_frame * self.channels, dtype=np.int16)
             data = data.reshape(-1, self.channels)
             frame = av.AudioFrame.from_ndarray(data.T, format='s16', layout='mono')
             frame.sample_rate = self.sample_rate

        # Set PTS
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        self._timestamp += samples
        
        return frame


class WebRTCTransportHandler(BaseTransportHandler):
    def __init__(self, loop, pipeline, signaling_url, ice_servers=None):
        super().__init__(loop, pipeline)
        self.signaling_url = signaling_url
        self.pc = None
        self._participant_joined = asyncio.Event()
        self._signaling_task = None
        self.audio_track = WebRTCOutputTrack(loop=loop)    
        self._on_session_end = None
        self.ice_servers = ice_servers or [{"urls": "stun:stun.l.google.com:19302"}]

    async def connect(self):
        if not RTCPeerConnection:
             raise ImportError("aiortc and aiohttp are required for WebRTCConnectionHandler. Install with `pip install aiortc aiohttp`.")
             
        logger.info(f"Connecting to Signaling Server: {self.signaling_url}")
        logger.info(f"Using ICE servers: {self.ice_servers}")
        
        from aiortc import RTCConfiguration, RTCIceServer
        
        ice_server_configs = []
        for server in self.ice_servers:
            if isinstance(server, str):
                ice_server_configs.append(RTCIceServer(urls=server))
            elif isinstance(server, dict):
                urls = server.get("urls") or server.get("url")
                username = server.get("username")
                credential = server.get("credential")
                
                if username and credential:
                    ice_server_configs.append(
                        RTCIceServer(urls=urls, username=username, credential=credential)
                    )
                else:
                    ice_server_configs.append(RTCIceServer(urls=urls))
        
        config = RTCConfiguration(iceServers=ice_server_configs)
        self.pc = RTCPeerConnection(configuration=config)
        
        if self.audio_track:
            self.pc.addTrack(self.audio_track)

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                logger.info("WebRTC Audio Track received")
                asyncio.create_task(self._consume_audio_track(track))
            
            @track.on("ended")
            async def on_ended():
                logger.info("Track ended")
                
        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state is {self.pc.connectionState}")
            if self.pc.connectionState in ["failed", "closed", "disconnected"]:
                if self._on_session_end:
                    try:
                        self._on_session_end(f"webrtc_{self.pc.connectionState}")
                    except Exception as e:
                        logger.error(f"Error in session end callback: {e}")

        if self.signaling_url:
            self._signaling_task = asyncio.create_task(self._run_signaling())
        else:
            logger.warning("No signaling URL provided for WebRTC connection")

    async def _consume_audio_track(self, track):
        self._participant_joined.set()
        while True:
            try:
                frame = await track.recv()
                audio_bytes = frame.to_ndarray().tobytes()
                if self.pipeline:
                    await self.pipeline.on_audio_delta(audio_bytes)
            except Exception as e:
                logger.warning(f"Error receiving audio frame: {e}")
                break

    async def _run_signaling(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(self.signaling_url) as ws:
                    
                    self.pc.addTransceiver("audio", direction="sendrecv")
                    offer = await self.pc.createOffer()
                    await self.pc.setLocalDescription(offer)
                    
                    payload = {
                        "sdp": self.pc.localDescription.sdp,
                        "type": self.pc.localDescription.type
                    }
                    await ws.send_json(payload)

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if "sdp" in data:
                                desc = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                                await self.pc.setRemoteDescription(desc)
                            elif "candidate" in data:
                                candidate = data["candidate"]
                                pass
        except Exception as e:
            logger.error(f"Signaling error: {e}")

    async def wait_for_participant(self, participant_id=None):
        await self._participant_joined.wait()
        return "webrtc_peer"

    async def disconnect(self):
        if self._signaling_task:
            self._signaling_task.cancel()
        if self.pc:
            await self.pc.close()

    async def cleanup(self):
        await self.disconnect()
