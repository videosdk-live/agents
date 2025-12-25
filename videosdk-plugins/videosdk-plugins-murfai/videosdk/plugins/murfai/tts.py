from __future__ import annotations

from typing import Any, AsyncIterator, Optional, Union
import os
import httpx
import asyncio
import json
import aiohttp
import weakref
from dataclasses import dataclass
from videosdk.agents import TTS, segment_text
import base64
import uuid

MURFAI_SAMPLE_RATE = 24000
MURFAI_CHANNELS = 1
DEFAULT_MODEL = "Falcon"
DEFAULT_VOICE_ID = "en-US-natalie" 
DEFAULT_REGION = "GLOBAL"

REGION_URLS = {
    "GLOBAL": "global.api.murf.ai",
    "US_EAST": "us-east.api.murf.ai",
    "US_WEST": "us-west.api.murf.ai",
    "INDIA": "in.api.murf.ai",
    "CANADA": "ca.api.murf.ai",
    "SOUTH_KOREA": "kr.api.murf.ai",
    "UAE": "me.api.murf.ai",
    "JAPAN": "jp.api.murf.ai",
    "AUSTRALIA": "au.api.murf.ai",
    "EU_CENTRAL": "eu-central.api.murf.ai",
    "UK": "uk.api.murf.ai",
    "SOUTH_AFRICA": "sa-east.api.murf.ai",
}

@dataclass
class MurfAIVoiceSettings:
    """Settings specific to Murf.ai voice generation."""
    pitch: int = 0
    rate: int = 0
    style: str = "Conversational"
    variation: int = 1
    multi_native_locale: Optional[str] = None


class MurfAITTS(TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        region: str = DEFAULT_REGION,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE_ID,
        voice_settings: MurfAIVoiceSettings | None = None,
        enable_streaming: bool = True,
    ) -> None:
        """Initialize the Murf.ai TTS plugin.

        Args:
            api_key (Optional[str]): Murf API key. Uses MURFAI_API_KEY env var if not provided.
            region (str): The region code (GLOBAL, US_EAST, UK, INDIA, etc.). Defaults to US_EAST.
            model (str): The model to use (GEN2, FALCON). Defaults to FALCON.
            voice (str): The voice ID to use.
            voice_settings (Optional[MurfAIVoiceSettings]): Advanced voice settings (pitch, rate, style).
            enable_streaming (bool): Whether to use WebSocket streaming (low latency) or HTTP chunks.
        """
        super().__init__(
            sample_rate=MURFAI_SAMPLE_RATE, num_channels=MURFAI_CHANNELS
        )

        self.api_key = api_key or os.getenv("MURFAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Murf API key must be provided either through api_key parameter or MURFAI_API_KEY environment variable"
            )

        self.model = model
        self.voice = voice
        self.enable_streaming = enable_streaming
        self.voice_settings = voice_settings or MurfAIVoiceSettings()
        
        base_domain = REGION_URLS.get(region.upper(), REGION_URLS["US_EAST"])
        self.http_base_url = f"https://{base_domain}/v1/speech"
        self.ws_base_url = f"wss://{base_domain}/v1/speech"

        self.audio_track = None
        self.loop = None
        self._first_chunk_sent = False
        
        self._session = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=30.0, write=5.0, pool=5.0),
            follow_redirects=True,
        )

        self._ws_session = None
        self._ws_connection = None
        self._streams = weakref.WeakSet()
        self._send_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None
        self._should_stop = False
        self._connection_lock = asyncio.Lock()
        
        self._active_contexts: set[str] = set()
        self._context_futures: dict[str, asyncio.Future[None]] = {}

    def reset_first_audio_tracking(self) -> None:
        self._first_chunk_sent = False

    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        voice_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        try:
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or event loop not set")
                return

            target_voice = voice_id or self.voice
            self._should_stop = False

            if self.enable_streaming:
                await self._stream_synthesis(text, target_voice)
            else:
                if isinstance(text, AsyncIterator):
                    async for segment in segment_text(text):
                        if self._should_stop:
                            break
                        await self._chunked_synthesis(segment, target_voice)
                else:
                    await self._chunked_synthesis(text, target_voice)

        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")
            raise

    async def _chunked_synthesis(self, text: str, voice_id: str) -> None:
        """Non-streaming synthesis using the HTTP POST API"""
        url = f"{self.http_base_url}/stream"

        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
            "voiceId": voice_id,
            "model": self.model,
            "format": "PCM",  
            "sampleRate": MURFAI_SAMPLE_RATE,
            "style": self.voice_settings.style,
            "rate": self.voice_settings.rate,
            "pitch": self.voice_settings.pitch,
            "variation": self.voice_settings.variation
        }

        if self.voice_settings.multi_native_locale:
            payload["multiNativeLocale"] = self.voice_settings.multi_native_locale

        try:
            async with self._session.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    response.raise_for_status()
                
                async for chunk in response.aiter_bytes():
                    if self._should_stop:
                        break
                    if chunk:
                        await self._stream_audio_chunks(chunk)

        except httpx.HTTPStatusError as e:
            self.emit("error", f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            self.emit("error", f"Chunked synthesis failed: {str(e)}")
            raise

    async def _stream_synthesis(self, text: Union[AsyncIterator[str], str], voice_id: str) -> None:
        """WebSocket-based streaming synthesis"""
        try:
            await self._ensure_connection()

            context_id = uuid.uuid4().hex[:12]
            done_future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
            self.register_context(context_id, done_future)

            async def _single_chunk_gen(s: str) -> AsyncIterator[str]:
                yield s

            async def _send_chunks() -> None:
                try:
                    segments = []
                    if isinstance(text, str):
                        async for segment in segment_text(_single_chunk_gen(text)):
                            segments.append(segment)
                    else:
                        async for chunk in text:
                            segments.append(chunk)
                    
                    for i, segment in enumerate(segments):
                        if self._should_stop:
                            break
                        is_last = (i == len(segments) - 1)
                        await self.send_text(
                            context_id, 
                            f"{segment} ", 
                            voice_id,
                            send_voice_config=False,
                            is_end=is_last
                        )

                except Exception as e:
                    if not done_future.done():
                        done_future.set_exception(e)

            sender = asyncio.create_task(_send_chunks())
            
            await done_future
            await sender

        except Exception as e:
            self.emit("error", f"Streaming synthesis failed: {str(e)}")
            raise

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        if not audio_bytes or self._should_stop:
            return

        if not self._first_chunk_sent and hasattr(self, '_first_audio_callback') and self._first_audio_callback:
            self._first_chunk_sent = True
            asyncio.create_task(self._first_audio_callback())

        if self.audio_track and self.loop:
            await self.audio_track.add_new_bytes(audio_bytes)

    async def interrupt(self) -> None:
        """Interrupt current synthesis and clear all contexts."""
        self._should_stop = True
        
        if self.audio_track:
            self.audio_track.interrupt()
        
        # Clear all pending futures
        for fut in list(self._context_futures.values()):
            if not fut.done():
                fut.cancel()
        self._context_futures.clear()
        
        # Close all active contexts
        await self.close_all_contexts()

    async def aclose(self) -> None:
        self._should_stop = True
        
        for task in [self._send_task, self._recv_task]:
            if task and not task.done():
                task.cancel()

        for stream in list(self._streams):
            try:
                await stream.aclose()
            except Exception:
                pass
        self._streams.clear()

        if self._ws_connection and not self._ws_connection.closed:
            await self._ws_connection.close()
        
        if self._ws_session and not self._ws_session.closed:
            await self._ws_session.close()

        self._ws_connection = None
        self._ws_session = None
        
        if self._session:
            await self._session.aclose()
        
        await super().aclose()

    async def _ensure_connection(self) -> None:
        async with self._connection_lock:
            if self._ws_connection and not self._ws_connection.closed:
                return

            if self._ws_session and not self._ws_session.closed:
                await self._ws_session.close()

            self._ws_session = aiohttp.ClientSession()

            params = {
                "api_key": self.api_key,
                "model": self.model,
                "sample_rate": str(MURFAI_SAMPLE_RATE),
                "format": "PCM",
                "channel_type": "MONO"
            }
            
            param_string = "&".join([f"{k}={v}" for k, v in params.items()])
            full_ws_url = f"{self.ws_base_url}/stream-input?{param_string}"
            
            headers = {"api_key": self.api_key}

            self._ws_connection = await asyncio.wait_for(
                self._ws_session.ws_connect(full_ws_url, headers=headers), 
                timeout=10.0
            )

            if self._recv_task and not self._recv_task.done():
                self._recv_task.cancel()
            self._recv_task = asyncio.create_task(self._recv_loop())

    def register_context(self, context_id: str, done_future: asyncio.Future[None]) -> None:
        self._context_futures[context_id] = done_future

    def _get_voice_config(self, voice_id: str) -> dict:
        config = {
            "voice_id": voice_id,
            "style": self.voice_settings.style,
            "rate": self.voice_settings.rate,
            "pitch": self.voice_settings.pitch,
            "variation": self.voice_settings.variation,
        }
        if self.voice_settings.multi_native_locale:
            config["multi_native_locale"] = self.voice_settings.multi_native_locale
        return config

    async def send_text(self, context_id: str, text: str, voice_id: str, send_voice_config: bool = False, is_end: bool = False) -> None:
        """Sends a text segment to Murf."""
        if not self._ws_connection or self._ws_connection.closed:
            raise RuntimeError("WebSocket connection is closed")
        
        if not text or not text.strip():
            return
        
        payload = {
            "text": text,
            "context_id": context_id,
            "end": is_end
        }
        
        if context_id not in self._active_contexts:
            payload["voice_config"] = self._get_voice_config(voice_id)
            self._active_contexts.add(context_id)
        
        await self._ws_connection.send_str(json.dumps(payload))
    
    async def send_end(self, context_id: str, voice_id: str) -> None:
        """Sends the end message to finalize the context."""
        if not self._ws_connection or self._ws_connection.closed:
            return
        
        payload = {
            "text": " ",
            "context_id": context_id,
            "end": True
        }
        
        await self._ws_connection.send_str(json.dumps(payload))

    async def close_context(self, context_id: str) -> None:
        """Clears a specific context."""
        if not self._ws_connection or self._ws_connection.closed:
            return
        
        try:
            payload = {
                "clear": True,
                "context_id": context_id
            }
            await self._ws_connection.send_str(json.dumps(payload))
            self._active_contexts.discard(context_id)
        except Exception:
            pass

    async def close_all_contexts(self) -> None:
        try:
            for context_id in list(self._active_contexts):
                await self.close_context(context_id)
        except Exception:
            pass

    async def _recv_loop(self) -> None:
        try:
            while self._ws_connection and not self._ws_connection.closed:
                msg = await self._ws_connection.receive()
                
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    if "error" in data:
                        self.emit("error", f"WebSocket error: {data.get('error')}")
                        ctx_id = data.get("context_id")
                        if ctx_id:
                            fut = self._context_futures.get(ctx_id)
                            if fut and not fut.done():
                                fut.set_exception(RuntimeError(data.get("error", "Unknown error")))
                        continue

                    if "audio" in data and data["audio"]:
                        try:
                            audio_chunk = base64.b64decode(data["audio"])
                            if audio_chunk and not self._should_stop:
                                await self._stream_audio_chunks(audio_chunk)
                        except Exception:
                            continue

                    if data.get("final") is True:
                        ctx_id = data.get("context_id")
                        if ctx_id:
                            fut = self._context_futures.pop(ctx_id, None)
                            self._active_contexts.discard(ctx_id)
                            if fut and not fut.done():
                                fut.set_result(None)

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                    break
                    
        except Exception as e:
            self.emit("error", f"WebSocket receive loop error: {str(e)}")
            for fut in self._context_futures.values():
                if not fut.done():
                    fut.set_exception(RuntimeError(f"WebSocket receive loop error: {e}"))
            self._context_futures.clear()
