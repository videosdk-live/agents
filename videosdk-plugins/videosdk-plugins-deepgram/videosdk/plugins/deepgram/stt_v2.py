from __future__ import annotations

import asyncio
import json
import numpy as np
from typing import Any, Optional
import os
from urllib.parse import urlencode
import aiohttp
from videosdk.agents import STT as BaseSTT, STTResponse, SpeechEventType, SpeechData, global_event_emitter
import logging

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
logger = logging.getLogger(__name__)


class DeepgramSTTV2(BaseSTT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "flux-general-en",
        input_sample_rate: int = 48000,
        target_sample_rate: int = 16000,
        eager_eot_threshold:float=0.6,
        eot_threshold:float=0.8,
        eot_timeout_ms:int=7000,
        base_url: str = "wss://api.deepgram.com/v2/listen",
        enable_preemptive_generation: bool = False,
    ) -> None:
        """Initialize the Deepgram STT plugin

        Args:
            api_key (str | None, optional): Deepgram API key. Uses DEEPGRAM_API_KEY environment variable if not provided. Defaults to None.
            model (str): The model to use for the STT plugin. Defaults to "flux-general-en".
            input_sample_rate (int): The input sample rate to use for the STT plugin. Defaults to 48000.
            target_sample_rate (int): The target sample rate to use for the STT plugin. Defaults to 16000.
            eager_eot_threshold (float): Eager end-of-turn threshold. Defaults to 0.6.
            eot_threshold (float): End-of-turn threshold. Defaults to 0.8.
            eot_timeout_ms (int): End-of-turn timeout in milliseconds. Defaults to 7000.
            base_url (str): The base URL to use for the STT plugin. Defaults to "wss://api.deepgram.com/v2/listen".
            enable_preemptive_generation (bool): Enable preemptive generation based on EagerEndOfTurn events. Defaults to False.
        """
        super().__init__()

        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Deepgram API key must be provided either through api_key parameter or DEEPGRAM_API_KEY environment variable")

        self.model = model
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.eager_eot_threshold = eager_eot_threshold
        self.eot_threshold=eot_threshold
        self.eot_timeout_ms = eot_timeout_ms
        self.base_url = base_url
        self.enable_preemptive_generation = enable_preemptive_generation

        self._stream_buffer = bytearray()
        self._target_chunk_size = int(0.1 * self.target_sample_rate * 2)
        self._min_chunk_size = int(0.05 * self.target_sample_rate * 2)

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._last_transcript: str = ""
        self._ws_task = None
    

    async def process_audio(
        self,
        audio_frames: bytes,
        **kwargs: Any
    ) -> None:
        """Process audio frames and send to Deeepgram's Flux API"""
        
        if not self._ws:
            await self._connect_ws()
            self._ws_task = asyncio.create_task(self._listen_for_responses())
            
        try:
            resampled_audio = self._resample_audio(audio_frames)
            if not resampled_audio:
                return
                
            self._stream_buffer.extend(resampled_audio)
             # chunk size 100ms
            while len(self._stream_buffer) >= self._target_chunk_size:
                chunk_to_send = bytes(self._stream_buffer[:self._target_chunk_size])
                self._stream_buffer = self._stream_buffer[self._target_chunk_size:]
                
                await self._ws.send_bytes(bytes(chunk_to_send))
                
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
                    self.emit(
                        "error", f"WebSocket error: {self._ws.exception()}")
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
            "encoding": "linear16",
            "sample_rate": self.target_sample_rate,
            "eot_threshold": self.eot_threshold,
            "eot_timeout_ms": self.eot_timeout_ms,
            "eager_eot_threshold": self.eager_eot_threshold,
        }
        headers = {"Authorization": f"Token {self.api_key}"}
        ws_url = f"{self.base_url}?{urlencode(query_params)}"

        try:
            self._ws = await self._session.ws_connect(ws_url, headers=headers)
            logger.info("Connected to Deepgram V2 WebSocket.")
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {str(e)}")
            raise

    def _handle_ws_message(self, msg: dict) -> list[STTResponse]:
        """Handle incoming WebSocket messages and generate STT responses"""
        responses = []

        try:
            if msg.get("type") != "TurnInfo":
                return responses

            event = msg.get("event")
            transcript = msg.get("transcript", "")
            # logger.info(f"{event} and {transcript}")
            start_time = msg.get("audio_window_start", 0.0)
            end_time = msg.get("audio_window_end", 0.0)
            confidence = msg.get("end_of_turn_confidence", 0.0)

            self._last_transcript = transcript
            # Emit turn-related events
            if event == "StartOfTurn":
                global_event_emitter.emit("speech_started")
            elif event == "EagerEndOfTurn":
                # Handle EagerEndOfTurn for preemptive generation
                if self.enable_preemptive_generation and transcript and self._transcript_callback:
                    responses.append(
                        STTResponse(
                            event_type=SpeechEventType.PREFLIGHT,
                            data=SpeechData(
                                text=transcript,
                                confidence=confidence,
                                start_time=start_time,
                                end_time=end_time,
                            ),
                            metadata={"model": self.model},
                        )
                    )
            elif event == "EndOfTurn":
                logger.info(f"EndOfTurn (FINAL) Transcript: {transcript} and Confidence: {confidence}")
                global_event_emitter.emit("speech_stopped")
                if transcript and self._transcript_callback:
                    responses.append(
                        STTResponse(
                            event_type=SpeechEventType.FINAL,
                            data=SpeechData(
                                text=transcript,
                                confidence=confidence,
                                start_time=start_time,
                                end_time=end_time,
                            ),
                            metadata={"model": self.model},
                        )
                    )
            elif event == "TurnResumed":
                # Send interim to signal user continued speaking
                if self.enable_preemptive_generation and transcript:
                    responses.append(
                            STTResponse(
                                event_type=SpeechEventType.INTERIM,
                                data=SpeechData(
                                    text=transcript,
                                    confidence=confidence,
                                    start_time=start_time,
                                    end_time=end_time,
                                ),
                                metadata={"model": self.model, "turn_resumed": True},
                            )
                    )

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
        await super().aclose()