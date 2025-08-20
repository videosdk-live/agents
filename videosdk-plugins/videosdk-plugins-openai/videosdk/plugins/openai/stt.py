from __future__ import annotations

import asyncio
import base64
import os
from typing import Any, Optional
from urllib.parse import urlencode
from scipy import signal
import aiohttp
import httpx
import openai
import numpy as np
from videosdk.agents import STT as BaseSTT, STTResponse, SpeechEventType, SpeechData, global_event_emitter

class OpenAISTT(BaseSTT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "whisper-1",
        base_url: str | None = None,
        prompt: str | None = None,
        language: str = "en",
        turn_detection: dict | None = None,
    ) -> None:
        super().__init__()
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either through api_key parameter or OPENAI_API_KEY environment variable")
        
        self.model = model
        self.language = language
        self.prompt = prompt
        self.turn_detection = turn_detection or {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
        }
        
        self.client = openai.AsyncClient(
            max_retries=0,
            api_key=self.api_key,
            base_url=base_url or None,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._current_text = ""
        self._last_interim_at = 0
    
        self.input_sample_rate = 48000
        self.target_sample_rate = 16000
        
    async def process_audio(
        self,
        audio_frames: bytes,
        language: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Process audio frames and send to OpenAI's Realtime API"""
        
        if not self._ws:
            await self._connect_ws()
            self._ws_task = asyncio.create_task(self._listen_for_responses())
            
        try:
            audio_data = np.frombuffer(audio_frames, dtype=np.int16)
            audio_data = signal.resample(audio_data, int(len(audio_data) * self.target_sample_rate / self.input_sample_rate))
            audio_data = audio_data.astype(np.int16).tobytes()
            audio_data = base64.b64encode(audio_data).decode("utf-8")
            message = {
                "type": "input_audio_buffer.append",
                "audio": audio_data,
            }
            await self._ws.send_json(message)
        except Exception as e:
            print(f"Error in process_audio: {str(e)}")
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
                    print(error)
                    self.emit("error", error)
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    print("WebSocket connection closed")
                    break
        except Exception as e:
            error = f"Error in WebSocket listener: {str(e)}"
            print(error)
            self.emit("error", error)
        finally:
            if self._ws:
                await self._ws.close()
                self._ws = None
                
    async def _connect_ws(self) -> None:
        """Establish WebSocket connection with OpenAI's Realtime API"""
        
        if not self._session:
            self._session = aiohttp.ClientSession()
            
        config = {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": self.model,
                    "prompt": self.prompt or "",
                    "language": self.language if self.language else None,
                },
                "turn_detection": self.turn_detection,
                "input_audio_noise_reduction": {
                    "type": "near_field"
                },
                "include": ["item.input_audio_transcription.logprobs"]
            }
        }
        
        query_params = {
            "intent": "transcription",
        }
        headers = {
            "User-Agent": "VideoSDK",
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        
        base_url = str(self.client.base_url).rstrip('/')
        ws_url = f"{base_url}/realtime?{urlencode(query_params)}"
        if ws_url.startswith("http"):
            ws_url = ws_url.replace("http", "ws", 1)

        try:
            self._ws = await self._session.ws_connect(ws_url, headers=headers)
            
            initial_response = await self._ws.receive_json()
            
            if initial_response.get("type") != "transcription_session.created":
                raise Exception(f"Expected session creation, got: {initial_response}")
            
            await self._ws.send_json(config)
            
            update_response = await self._ws.receive_json()
            
            if update_response.get("type") != "transcription_session.updated":
                raise Exception(f"Configuration update failed: {update_response}")
            
        except Exception as e:
            print(f"Error connecting to WebSocket: {str(e)}")
            if self._ws:
                await self._ws.close()
                self._ws = None
            raise
        
    def _handle_ws_message(self, msg: dict) -> list[STTResponse]:
        """Handle incoming WebSocket messages and generate STT responses"""
        responses = []
        
        try:
            msg_type = msg.get("type")
            if msg_type == "conversation.item.input_audio_transcription.delta":
                delta = msg.get("delta", "")
                if delta:
                    self._current_text += delta
                    current_time = asyncio.get_event_loop().time()
                    
                    if current_time - self._last_interim_at > 0.5:
                        responses.append(STTResponse(
                            event_type=SpeechEventType.INTERIM,
                            data=SpeechData(
                                text=self._current_text,
                                language=self.language,
                            ),
                            metadata={"model": self.model}
                        ))
                        self._last_interim_at = current_time
                        
            elif msg_type == "conversation.item.input_audio_transcription.completed":
                transcript = msg.get("transcript", "")
                if transcript:
                    responses.append(STTResponse(
                        event_type=SpeechEventType.FINAL,
                        data=SpeechData(
                            text=transcript,
                            language=self.language,
                        ),
                        metadata={"model": self.model}
                    ))
                    self._current_text = ""
            
            elif msg_type == "input_audio_buffer.speech_started":
                global_event_emitter.emit("speech_started")
            
            elif msg_type == "input_audio_buffer.speech_stopped":
                global_event_emitter.emit("speech_stopped")
                
        except Exception as e:
            print(f"Error handling WebSocket message: {str(e)}")
        
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
            
        await self.client.close()

    async def _ensure_ws_connection(self):
        """Ensure WebSocket is connected, reconnect if necessary"""
        if not self._ws or self._ws.closed:
            await self._connect_ws()



