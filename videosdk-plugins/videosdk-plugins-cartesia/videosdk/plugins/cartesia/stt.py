from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional
from urllib.parse import urlencode
import aiohttp
import numpy as np
from scipy import signal
from videosdk.agents import STT as BaseSTT, STTResponse, SpeechEventType, SpeechData, global_event_emitter
import logging

logger = logging.getLogger(__name__)

class CartesiaSTT(BaseSTT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "ink-whisper",
        language: str = "en",
        sample_rate: int = 48000,
        base_url: str = "wss://api.cartesia.ai/stt/websocket",
    ) -> None:
        super().__init__()
        
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY")
        if not self.api_key:
            raise ValueError("Cartesia API key must be provided either through api_key parameter or CARTESIA_API_KEY environment variable")
        
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.base_url = base_url
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._last_interim_at = 0.0
        self.input_sample_rate = sample_rate
        self.target_sample_rate = 16000  
        
    async def process_audio(
        self,
        audio_frames: bytes,
        language: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Process audio frames and send to Cartesia's STT API"""
        
        if not self._ws:
            await self._connect_ws()
            
            self._ws_task = asyncio.create_task(self._listen_for_responses())
            
        try:
            
            audio_data = np.frombuffer(audio_frames, dtype=np.int16)
            if self.input_sample_rate != self.target_sample_rate:
                audio_data = signal.resample(
                    audio_data, 
                    int(len(audio_data) * self.target_sample_rate / self.input_sample_rate)
                )
            audio_bytes = audio_data.astype(np.int16).tobytes()
            await self._ws.send_bytes(audio_bytes)
            
        except Exception as e:
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
                    error = f"WebSocket error: {self._ws.exception()}"
                    self.emit("error", error)
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("WebSocket connection closed")
                    break
        except Exception as e:
            self.emit("error", f"Error listening for responses: {str(e)}")
        finally:
            if self._ws:
                await self._ws.close()
                self._ws = None

    async def _connect_ws(self) -> None:
        """Establish WebSocket connection with Cartesia's STT API"""
        
        if not self._session:
            self._session = aiohttp.ClientSession()
            
        query_params = {
            "model": self.model,
            "language": self.language,
            "encoding": "pcm_s16le", 
            "sample_rate": str(self.target_sample_rate),  
            "api_key": self.api_key, 
        }
        
        headers = {
            "Cartesia-Version": "2024-11-13", 
            "User-Agent": "VideoSDK-Cartesia-STT",
        }
        
        ws_url = f"{self.base_url}?{urlencode(query_params)}"
            
        try:
            self._ws = await self._session.ws_connect(ws_url, headers=headers)
            
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {str(e)}")
            if self._ws:
                await self._ws.close()
                self._ws = None
            raise
        
    def _handle_ws_message(self, msg: dict) -> list[STTResponse]:
        """Handle incoming WebSocket messages and generate STT responses"""
        responses = []
        try:
            msg_type = msg.get("type")
            
            if msg_type == "transcript":
                transcript = msg.get("text", "")
                is_final = msg.get("is_final", False)
                language = msg.get("language", self.language)
                duration = msg.get("duration", 0.0)
                
                if transcript:
                    current_time = time.time()
                    
                    if is_final:
                        responses.append(STTResponse(
                            event_type=SpeechEventType.FINAL,
                            data=SpeechData(
                                text=transcript,
                                confidence=1.0,  
                                language=language,
                                start_time=0.0,  
                                end_time=duration,
                            ),
                            metadata={
                                "model": self.model,
                                "request_id": msg.get("request_id"),
                                "duration": duration,
                            }
                        ))
                    else:
                        if current_time - self._last_interim_at > 0.1:  
                            responses.append(STTResponse(
                                event_type=SpeechEventType.INTERIM,
                                data=SpeechData(
                                    text=transcript,
                                    confidence=1.0,
                                    language=language,
                                    start_time=0.0,
                                    end_time=duration,
                                ),
                                metadata={
                                    "model": self.model,
                                    "request_id": msg.get("request_id"),
                                    "duration": duration,
                                }
                            ))
                            self._last_interim_at = current_time
                            
            elif msg_type == "flush_done":
                logger.info("Cartesia STT: Flush completed")
                
            elif msg_type == "done":
                logger.info("Cartesia STT: Session ended")
                
            elif msg_type == "error":
                error_msg = msg.get("message", "Unknown error")
                error_code = msg.get("code", "unknown")
                self.emit("error", f"{error_code}: {error_msg}")
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {str(e)}")
        
        return responses

    async def aclose(self) -> None:
        """Cleanup resources"""
        if self._ws and not self._ws.closed:
            try:
                await self._ws.send_str("done")
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error sending done command: {str(e)}")
        
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