from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional
from urllib.parse import urlencode
import logging

import numpy as np
import aiohttp
from videosdk.agents import STT as BaseSTT, STTResponse, SpeechData, SpeechEventType, global_event_emitter

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class AssemblyAISTT(BaseSTT):

    def __init__(
        self,
        *,
        api_key: str | None = None,
        input_sample_rate: int = 48000,
        target_sample_rate: int = 16000,
        format_turns: bool = True,
        keyterms_prompt: list[str] | None = None,
        end_of_turn_confidence_threshold: float = 0.5,
        min_end_of_turn_silence_when_confident: int = 800,
        max_turn_silence: int = 2000,
    ) -> None:
        """Initialize the AssemblyAI STT plugin.

        Args:
            api_key (str | None, optional): AssemblyAI API key. Uses ASSEMBLYAI_API_KEY environment variable if not provided. Defaults to None.
            input_sample_rate (int): The input sample rate to use for the STT plugin. Defaults to 48000.
            target_sample_rate (int): The target sample rate to use for the STT plugin. Defaults to 16000.
            format_turns (bool): Whether to format turns. Defaults to True.
            keyterms_prompt (list[str] | None, optional): The word boost list to use for the STT plugin. Defaults to None.
            end_of_turn_confidence_threshold (float): The end of turn confidence threshold to use for the STT plugin. Defaults to 0.5.
            min_end_of_turn_silence_when_confident (int): The minimum end of turn silence when confident to use for the STT plugin. Defaults to 800.
            max_turn_silence (int): The maximum turn silence to use for the STT plugin. Defaults to 2000.
        """
        super().__init__()
        
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is not installed. Please install it with 'pip install scipy'")

        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "AssemblyAI API key must be provided either through the 'api_key' parameter "
                "or the 'ASSEMBLYAI_API_KEY' environment variable."
            )
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.format_turns = format_turns
        self.keyterms_prompt = keyterms_prompt or []
        self.end_of_turn_confidence_threshold = end_of_turn_confidence_threshold
        self.min_end_of_turn_silence_when_confident = min_end_of_turn_silence_when_confident
        self.max_turn_silence = max_turn_silence

        connection_params = {
            "sample_rate": self.target_sample_rate,
            "format_turns": self.format_turns,
        }
        
            
        if self.end_of_turn_confidence_threshold != 0.7:
            connection_params["end_of_turn_confidence_threshold"] = self.end_of_turn_confidence_threshold
        if self.min_end_of_turn_silence_when_confident != 1500:
            connection_params["min_end_of_turn_silence_when_confident"] = self.min_end_of_turn_silence_when_confident
        if self.max_turn_silence != 3000:
            connection_params["max_turn_silence"] = self.max_turn_silence
        
        if self.keyterms_prompt:
            connection_params["keyterms_prompt"] = json.dumps(self.keyterms_prompt)

        self.ws_url = f"wss://streaming.assemblyai.com/v3/ws?{urlencode(connection_params)}"
        logger.info(f"[AssemblyAI] WebSocket URL: {self.ws_url}")

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        
        self._stream_buffer = bytearray()
        self._target_chunk_size = int(0.1 * self.target_sample_rate * 2)  
        self._min_chunk_size = int(0.05 * self.target_sample_rate * 2)   
        
        self._last_speech_event_time = 0.0
        self._last_transcript = ""
        self._is_speaking = False

    async def process_audio(
        self,
        audio_frames: bytes,
        **kwargs: Any
    ) -> None:
        """Process audio frames and send to AssemblyAI's Streaming API"""
        
        if not self._ws:
            await self._connect_ws()
            self._ws_task = asyncio.create_task(self._listen_for_responses())
            
        try:
            resampled_audio = self._resample_audio(audio_frames)
            if not resampled_audio:
                return
                
            self._stream_buffer.extend(resampled_audio)
            
            while len(self._stream_buffer) >= self._target_chunk_size:
                chunk_to_send = bytes(self._stream_buffer[:self._target_chunk_size])
                self._stream_buffer = self._stream_buffer[self._target_chunk_size:]
                
                await self._ws.send_bytes(chunk_to_send)
                
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
        """Establish WebSocket connection with AssemblyAI's Streaming API"""
        
        if not self._session:
            self._session = aiohttp.ClientSession()
            
        headers = {
            "Authorization": self.api_key,
            "User-Agent": "AssemblyAI/1.0 (integration=VideoSDK)"
        }
        
        try:
            self._ws = await self._session.ws_connect(self.ws_url, headers=headers)
            logger.info("[AssemblyAI] WebSocket connection opened")
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {str(e)}")
            raise
        
    def _handle_ws_message(self, msg: dict) -> list[STTResponse]:
        """Handle incoming WebSocket messages and generate STT responses"""
        responses = []
        
        try:
            msg_type = msg.get('type')
            logger.info(f"[AssemblyAI] Message type: {msg_type}")

            if msg_type == "Begin":
                session_id = msg.get('id')
                logger.info(f"[AssemblyAI] Session began: ID={session_id}")
                
            elif msg_type == "Turn":
                transcript = msg.get('transcript', '')
                formatted = msg.get('turn_is_formatted', False)
                confidence = msg.get('confidence', 1.0)
                
                if transcript and transcript.strip():
                    self._last_transcript = transcript.strip()
                    
                    event_type = SpeechEventType.FINAL if formatted else SpeechEventType.INTERIM
                    
                    response = STTResponse(
                        event_type=event_type,
                        data=SpeechData(
                            text=transcript.strip(),
                            confidence=confidence
                        )
                    )
                    
                    responses.append(response)
                    
                    if not self._is_speaking:
                        self._is_speaking = True
                        global_event_emitter.emit("speech_started")
                        
                    if formatted:
                        self._is_speaking = False
                        self._last_transcript = ""
                        
            elif msg_type == "Termination":
                if self._last_transcript and self._is_speaking:
                    final_response = STTResponse(
                        event_type=SpeechEventType.FINAL,
                        data=SpeechData(
                            text=self._last_transcript,
                            confidence=1.0
                        )
                    )
                    responses.append(final_response)
                    self._last_transcript = ""
                    self._is_speaking = False
                
            elif msg_type == "Error":
                error_msg = msg.get('error', 'Unknown error')
                logger.error(f"AssemblyAI Error: {error_msg}")

        except Exception as e:
            logger.error(f"Error handling WebSocket message: {str(e)}")
        
        return responses

    def _resample_audio(self, audio_bytes: bytes) -> bytes:
        """Resample audio from input sample rate to target sample rate and convert to mono."""
        try:
            if not audio_bytes:
                return b''

            raw_audio = np.frombuffer(audio_bytes, dtype=np.int16)
            if raw_audio.size == 0:
                return b''

            if raw_audio.size % 2 == 0: 
                stereo_audio = raw_audio.reshape(-1, 2)
                mono_audio = stereo_audio.astype(np.float32).mean(axis=1)
            else:
                mono_audio = raw_audio.astype(np.float32)

            if self.input_sample_rate != self.target_sample_rate:
                target_length = int(len(mono_audio) * self.target_sample_rate / self.input_sample_rate)
                resampled_data = signal.resample(mono_audio, target_length)
            else:
                resampled_data = mono_audio

            resampled_data = np.clip(resampled_data, -32767, 32767)
            return resampled_data.astype(np.int16).tobytes()

        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return b''

    async def aclose(self) -> None:
        """Cleanup resources"""
        
        if len(self._stream_buffer) >= self._min_chunk_size and self._ws:
            try:
                final_chunk = bytes(self._stream_buffer)
                await self._ws.send_bytes(final_chunk)
            except Exception as e:
                logger.error(f"Error sending final audio: {e}")
        
        if self._ws:
            try:
                await self._ws.send_str(json.dumps({"type": "Terminate"}))
                await asyncio.sleep(0.5)  
            except Exception as e:
                logger.error(f"Error sending termination: {e}")

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
