from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional, Union
import os
import asyncio
import json
import aiohttp
from dataclasses import dataclass

from videosdk.agents import TTS

LMNT_SAMPLE_RATE = 24000
LMNT_CHANNELS = 1
LMNT_WEBSOCKET_URL = "wss://api.lmnt.com/v1/ai/speech/stream"

DEFAULT_VOICE = "morgan"
DEFAULT_FORMAT = "raw"
DEFAULT_LANGUAGE = "auto"
DEFAULT_SAMPLE_RATE = 24000

try:
    import pydub
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

@dataclass
class LMNTVoiceConfig:
    """Configuration for LMNT voice settings"""
    voice: str = DEFAULT_VOICE
    format: str = DEFAULT_FORMAT
    language: str = DEFAULT_LANGUAGE
    sample_rate: int = DEFAULT_SAMPLE_RATE
    return_extras: bool = False
    auto_flush: bool = False  

class LMNTTTS(TTS):
    """
    LMNT TTS implementation using WebSocket streaming API (plug-and-play for VideoSDK Agents).
    Usage:
        tts = LMNTTTS()  # All config from env, or override via kwargs
    """
    
    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        format: str = DEFAULT_FORMAT,
        language: str = DEFAULT_LANGUAGE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        return_extras: bool = False,
        auto_flush: bool = False,  
        flush_interval: float = 1.0,  
        api_key: Optional[str] = None,
        **kwargs: Any
    ):
        super().__init__(sample_rate=LMNT_SAMPLE_RATE, num_channels=LMNT_CHANNELS)
        
        self.voice = voice
        self.format = format
        self.language = language
        self._sample_rate = sample_rate  
        self.return_extras = return_extras
        self.auto_flush = auto_flush
        self.flush_interval = flush_interval
        self.audio_track = None
        self.loop = None
        self.stt_component = None
        
        self.api_key = api_key or os.getenv("LMNT_API_KEY")
        if not self.api_key:
            raise ValueError("LMNT API key must be provided either through api_key parameter or LMNT_API_KEY environment variable")
        
        self._session = None
        self._ws = None
        self._last_flush_time = 0.0
    
    async def synthesize(self, text_or_generator: Union[str, AsyncIterator[str]], **kwargs) -> None:
        if not self.audio_track or not self.loop:
            self.emit("error", "Audio track or event loop not initialized.")
            return

        if self.stt_component:
            self.stt_component.set_agent_speaking(True)

        try:
            if isinstance(text_or_generator, str):
                await self._stream_synthesis(text_or_generator)
            else:
                await self._stream_synthesis_with_flush(text_or_generator)

        except Exception as e:
            self.emit("error", f"Error in LMNT TTS synthesis: {e}")
        finally:
            if self.stt_component:
                self.stt_component.set_agent_speaking(False)

    async def _stream_synthesis_with_flush(self, text_generator: AsyncIterator[str]) -> None:
        """Stream text with automatic flushing for real-time synthesis"""
        if not self._session:
            self._session = aiohttp.ClientSession()
        
        try:
            async with self._session.ws_connect(LMNT_WEBSOCKET_URL) as ws:
                self._ws = ws
                
                init_message = {
                    "X-API-Key": self.api_key,
                    "voice": self.voice,
                    "format": self.format,
                    "language": self.language,
                    "sample_rate": self._sample_rate,
                    "return_extras": self.return_extras
                }
                await ws.send_str(json.dumps(init_message))
                
                current_text = ""
                start_time = asyncio.get_event_loop().time()
                
                async for text_chunk in text_generator:
                    current_text += text_chunk
                    
                    # Send text chunk
                    text_message = {"text": text_chunk}
                    print(text_message)
                    await ws.send_str(json.dumps(text_message))
                    
                    if self.auto_flush:
                        current_time = asyncio.get_event_loop().time()
                        if current_time - start_time >= self.flush_interval:
                            await self._flush()
                            start_time = current_time
                
                eof_message = {"eof": True}
                await ws.send_str(json.dumps(eof_message))
                
                await self._process_audio_responses(ws)
                
        except aiohttp.ClientError as e:
            self.emit("error", f"LMNT WebSocket connection error: {e}")
        except Exception as e:
            self.emit("error", f"LMNT TTS streaming error: {e}")
        finally:
            self._ws = None

    async def _stream_synthesis(self, text: str) -> None:
        """Stream text to LMNT WebSocket API and receive synthesized audio (traditional approach)"""
        if not self._session:
            self._session = aiohttp.ClientSession()
        
        try:
            async with self._session.ws_connect(LMNT_WEBSOCKET_URL) as ws:
                init_message = {
                    "X-API-Key": self.api_key,
                    "voice": self.voice,
                    "format": self.format,
                    "language": self.language,
                    "sample_rate": self._sample_rate,
                    "return_extras": self.return_extras
                }
                await ws.send_str(json.dumps(init_message))
                
                text_message = {"text": text}
                await ws.send_str(json.dumps(text_message))
                
                eof_message = {"eof": True}
                await ws.send_str(json.dumps(eof_message))
                
                await self._process_audio_responses(ws)
                        
        except aiohttp.ClientError as e:
            self.emit("error", f"LMNT WebSocket connection error: {e}")
        except Exception as e:
            self.emit("error", f"LMNT TTS streaming error: {e}")

    async def _process_audio_responses(self, ws) -> None:
        """Process audio responses from the WebSocket"""
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    
                    if "error" in data:
                        self.emit("error", f"LMNT API error: {data['error']}")
                        break
                    
                    if self.return_extras and ("durations" in data or "buffer_empty" in data):
                        if "buffer_empty" in data and data["buffer_empty"]:
                            pass
                        
                except json.JSONDecodeError:
                    pass
                    
            elif msg.type == aiohttp.WSMsgType.BINARY:
                if msg.data:
                    await self._stream_audio_chunk(msg.data)
                    
            elif msg.type == aiohttp.WSMsgType.ERROR:
                self.emit("error", f"WebSocket error: {ws.exception()}")
                break

    async def flush(self) -> None:
        """Manually flush the current text buffer to force synthesis"""
        if self._ws:
            try:
                flush_message = {"flush": True}
                await self._ws.send_str(json.dumps(flush_message))
                self._last_flush_time = asyncio.get_event_loop().time()
            except Exception as e:
                self.emit("error", f"Error during manual flush: {e}")

    async def _flush(self) -> None:
        """Internal flush method"""
        await self.flush()

    async def _stream_audio_chunk(self, audio_data: bytes) -> None:
        """Stream audio data in chunks to the audio track"""
        if not audio_data:
            return

        try:
            pcm_data = await self._convert_to_pcm(audio_data)
            
            if not pcm_data:
                return

            chunk_size = int(self._sample_rate * self.num_channels * 2 * 20 / 1000)
            
            for i in range(0, len(pcm_data), chunk_size):
                chunk = pcm_data[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    chunk += b'\x00' * (chunk_size - len(chunk))
                
                if self.audio_track and self.loop:
                    self.loop.create_task(self.audio_track.add_new_bytes(chunk))
                    await asyncio.sleep(0.01)  
                    
        except Exception as e:
            self.emit("error", f"Error in audio streaming: {e}")

    async def _convert_to_pcm(self, audio_data: bytes) -> bytes:
        """Convert audio data to PCM format for VideoSDK compatibility"""
        if self.format.lower() == "raw":
            return audio_data
        elif self.format.lower() == "mp3":
            if not PYDUB_AVAILABLE:
                self.emit("error", "pydub is required for MP3 decoding. Install with: pip install pydub")
                return b""
            
            try:
                from pydub import AudioSegment
                import io
                
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
                
                if audio_segment.channels != 1:
                    audio_segment = audio_segment.set_channels(1)
                
                if audio_segment.frame_rate != self._sample_rate:
                    audio_segment = audio_segment.set_frame_rate(self._sample_rate)
                
                pcm_bytes = audio_segment.raw_data
                return pcm_bytes
                
            except Exception as e:
                self.emit("error", f"Error converting MP3 to PCM: {e}")
                return b""
        elif self.format.lower() == "ulaw":
            if not PYDUB_AVAILABLE:
                self.emit("error", "pydub is required for µ-law decoding. Install with: pip install pydub")
                return b""
            
            try:
                from pydub import AudioSegment
                import io
                
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
                
                if audio_segment.channels != 1:
                    audio_segment = audio_segment.set_channels(1)
                
                if audio_segment.frame_rate != self._sample_rate:
                    audio_segment = audio_segment.set_frame_rate(self._sample_rate)
                
                pcm_bytes = audio_segment.raw_data
                return pcm_bytes
                
            except Exception as e:
                self.emit("error", f"Error converting µ-law to PCM: {e}")
                return b""
        else:
            self.emit("error", f"Unsupported audio format: {self.format}")
            return b""

    def set_stt_component(self, stt_component):
        """Set reference to STT component for conversation flow control"""
        self.stt_component = stt_component
        
    async def aclose(self):
        """Close the TTS connection and cleanup resources"""
        if self._session:
            await self._session.close()
        await super().aclose()

    async def interrupt(self) -> None:
        """Interrupt the TTS process"""
        if self.audio_track:
            self.audio_track.interrupt() 