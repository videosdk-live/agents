from __future__ import annotations

import asyncio
import time
from typing import Any, Optional
import os
from urllib.parse import urlencode
import aiohttp
from videosdk.agents import STT as BaseSTT, STTResponse, SpeechEventType, SpeechData, global_event_emitter
import logging

logger = logging.getLogger(__name__)

class DeepgramSTT(BaseSTT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "nova-2",
        language: str = "en-US",
        interim_results: bool = True,
        punctuate: bool = True,
        smart_format: bool = True,
        sample_rate: int = 48000,
        endpointing: int = 50,
        filler_words: bool = True,
        base_url: str = "wss://api.deepgram.com/v1/listen",
    ) -> None:
        super().__init__()
        
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("Deepgram API key must be provided either through api_key parameter or DEEPGRAM_API_KEY environment variable")
        
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.interim_results = interim_results
        self.punctuate = punctuate
        self.smart_format = smart_format
        self.endpointing = endpointing
        self.filler_words = filler_words
        self.base_url = base_url
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._last_speech_event_time = 0.0
        self._previous_speech_event_time = 0.0
        
    async def process_audio(
        self,
        audio_frames: bytes,
        language: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Process audio frames and send to Deepgram's Streaming API"""
        
        if not self._ws:
            await self._connect_ws()
            self._ws_task = asyncio.create_task(self._listen_for_responses())
            
        try:
            await self._ws.send_bytes(audio_frames)
        except Exception as e:
            logger.error(f"Error in process_audio: {str(e)}")
            self.emit("error", str(e))
            if self._ws:
                await self._ws.close()
                self._ws = None
                if self._ws_task:
                    self._ws_task.cancel()
                    self._ws_task = None

    async def _listen_for_responses(self) -> None:
        """Background task to listen for WebSocket responses"""
        if not self._ws:
            return
            
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = msg.json()
                    responses = self._handle_ws_message(data)
                    for response in responses:
                        if self._transcript_callback:
                            await self._transcript_callback(response)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    self.emit("error", f"WebSocket error: {self._ws.exception()}")
                    break
        except Exception as e:
            logger.error(f"Error in WebSocket listener: {str(e)}")
            self.emit("error", f"Error in WebSocket listener: {str(e)}")
        finally:
            if self._ws:
                await self._ws.close()
                self._ws = None
                
    async def _connect_ws(self) -> None:
        """Establish WebSocket connection with Deepgram's Streaming API"""
        
        if not self._session:
            self._session = aiohttp.ClientSession()
            
        query_params = {
            "model": self.model,
            "language": self.language,
            "interim_results": str(self.interim_results).lower(),
            "punctuate": str(self.punctuate).lower(),
            "smart_format": str(self.smart_format).lower(),
            "encoding": "linear16",
            "sample_rate": str(self.sample_rate), 
            "channels": 2,
            "endpointing": self.endpointing,
            "filler_words": str(self.filler_words).lower(),
            "vad_events": "true",
            "no_delay": "true",
        }
        headers = {
            "Authorization": f"Token {self.api_key}",
        }
        
        ws_url = f"{self.base_url}?{urlencode(query_params)}"
            
        try:
            self._ws = await self._session.ws_connect(ws_url, headers=headers)
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {str(e)}")
            raise
        
    def _handle_ws_message(self, msg: dict) -> list[STTResponse]:
        """Handle incoming WebSocket messages and generate STT responses"""
        responses = []
        try:
            if msg["type"] == "SpeechStarted":
                current_time = time.time()
        
                if self._last_speech_event_time == 0.0:
                    self._last_speech_event_time = current_time
                    return responses

                if current_time - self._last_speech_event_time < 1.0:
                    global_event_emitter.emit("speech_started")

                self._previous_speech_event_time = self._last_speech_event_time
                self._last_speech_event_time = current_time
            
            if msg["type"] == "Results":
                channel = msg["channel"]
                alternatives = channel["alternatives"]
                
                if alternatives and len(alternatives) > 0:
                    alt = alternatives[0]
                    is_final = msg["is_final"]
                    if alt["transcript"] == "":
                        return responses
                    
                    response = STTResponse(
                        event_type=SpeechEventType.FINAL if is_final else SpeechEventType.INTERIM,
                        data=SpeechData(
                            text=alt["transcript"],
                            language=self.language,
                            confidence=alt.get("confidence", 0.0),
                            start_time=alt["words"][0]["start"] if alt["words"] else 0.0,
                            end_time=alt["words"][-1]["end"] if alt["words"] else 0.0,
                        ),
                        metadata={"model": self.model}
                    )
                    responses.append(response)
                    
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {str(e)}")
        
        return responses

    async def aclose(self) -> None:
        """Cleanup resources"""
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None
            
        if self._ws:
            await self._ws.close()
            self._ws = None
            
        if self._session:
            await self._session.close()
            self._session = None
