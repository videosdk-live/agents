from __future__ import annotations

import asyncio
import base64
import json
import os
import logging
import time
from typing import Any, Optional, List
from urllib.parse import urlencode
import aiohttp
import numpy as np
from videosdk.agents import STT as BaseSTT, STTResponse, SpeechEventType, SpeechData, global_event_emitter

logger = logging.getLogger(__name__)

STT_ERROR_MSGS = {"input_error", "auth_error", "quota_exceeded", "transcriber_error", "error"}
SUPPORTED_SAMPLE_RATES = {8000, 16000, 22050, 24000, 44100, 48000}

class ElevenLabsSTT(BaseSTT):
    """
    ElevenLabs Realtime Speech-to-Text (STT) client.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str = "scribe_v2_realtime",
        language_code: str = "en",
        sample_rate: int = 48000,
        commit_strategy: str = "vad",
        vad_silence_threshold_secs: float = 0.8,
        vad_threshold: float = 0.4,
        min_speech_duration_ms: int = 50,
        min_silence_duration_ms: int = 50,
        base_url: str = "wss://api.elevenlabs.io/v1/speech-to-text/realtime",
    ) -> None:
        """
        Initialize the ElevenLabs STT client.

        Args:
            api_key: ElevenLabs API key for authentication. Defaults to env variable ELEVENLABS_API_KEY.
            model_id: STT model identifier.
            language_code: Language code for transcription.
            sample_rate: Sample rate of input audio in Hz.
            commit_strategy: Strategy for committing transcripts ('vad' is by default).
            vad_silence_threshold_secs: Duration of silence to detect end-of-speech.
            vad_threshold: Threshold for detecting voice activity.
            min_speech_duration_ms: Minimum duration in milliseconds for a speech segment.
            min_silence_duration_ms: Minimum duration in milliseconds of silence to consider end-of-speech.
            base_url: WebSocket endpoint for ElevenLabs STT.
        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        super().__init__()

        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key must be provided via api_key or ELEVENLABS_API_KEY env var")

        self.model_id = model_id
        self.language_code = language_code
        self.commit_strategy = commit_strategy
        self.base_url = base_url
        self.sample_rate = sample_rate

        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Unsupported sample_rate: {self.sample_rate}. Supported rates: {SUPPORTED_SAMPLE_RATES}")
        
        self.vad_silence_threshold_secs = vad_silence_threshold_secs
        self.vad_threshold = vad_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms

        self._last_final_text = ""
        self._last_final_time = 0.0
        self._duplicate_suppression_window = 0.75
        
        self._stream_buffer = bytearray()
        self._target_chunk_size = int(0.1 * self.sample_rate * 2) 

        self.heartbeat = 15.0
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None

    async def process_audio(
        self, 
        audio_frames: bytes, 
        **kwargs: Any
    ) -> None:
        """
        Process and send audio frames. 
        Converts to mono (required by ElevenLabs) and buffers 100ms chunks to reduce overhead.
        """

        if not self._ws or self._ws.closed:
            await self._connect_ws()
            if not self._ws_task or self._ws_task.done():
                self._ws_task = asyncio.create_task(self._listen_for_responses())

        elif self._ws_task and self._ws_task.done():
            logger.warning("WebSocket listener stopped unexpectedly, restarting")
            self._ws_task = asyncio.create_task(self._listen_for_responses())

        try:
            mono_audio = self._convert_to_mono(audio_frames)
            if not mono_audio:
                return
            
            self._stream_buffer.extend(mono_audio)
            
            while len(self._stream_buffer) >= self._target_chunk_size:
                chunk = self._stream_buffer[:self._target_chunk_size]
                await self._send_audio(chunk)
                self._stream_buffer = self._stream_buffer[self._target_chunk_size:]

        except Exception as e:
            logger.exception("Error in process_audio: %s", e)
            self.emit("error", str(e))
            if self._ws:
                await self._ws.close()
                self._ws = None

    async def _connect_ws(self) -> None:
        if not self._session:
            self._session = aiohttp.ClientSession()

        query_params = {
            "model_id": str(self.model_id),
            "language_code": str(self.language_code),
            "audio_format": f"pcm_{self.sample_rate}",
            "commit_strategy": str(self.commit_strategy),
            "vad_silence_threshold_secs": self.vad_silence_threshold_secs,
            "vad_threshold": self.vad_threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
        }

        ws_url = f"{self.base_url}?{urlencode(query_params)}"
        headers = {"xi-api-key": self.api_key}

        try:
            self._ws = await self._session.ws_connect(ws_url, headers=headers, heartbeat=self.heartbeat)
            logger.info("Connected to ElevenLabs Realtime STT WebSocket.")
        except Exception as e:
            logger.exception("Error connecting to ElevenLabs WebSocket: %s", e)
            raise

    async def _send_audio(self, audio_bytes: bytes) -> None:
        if not self._ws:
            return

        payload = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(audio_bytes).decode(),
            "sample_rate": self.sample_rate,
        }

        try:
            await self._ws.send_str(json.dumps(payload))
        except Exception as e:
            logger.exception("Error sending audio chunk: %s", e)
            self.emit("error", str(e))
            await self.aclose()

    def _convert_to_mono(self, audio_bytes: bytes) -> bytes:
        """
        Convert input audio bytes to mono. 
        """
        if not audio_bytes:
            return b""
        try:
            raw_audio = np.frombuffer(audio_bytes, dtype=np.int16)
            if raw_audio.size == 0:
                return b""

            if raw_audio.size % 2 == 0:
                try:
                    stereo = raw_audio.reshape(-1, 2).astype(np.float32)
                    mono = stereo.mean(axis=1)
                    return mono.astype(np.int16).tobytes()
                except ValueError:
                    pass
            
            return audio_bytes
        except Exception as e:
            logger.error("Error converting to mono: %s", e)
            return b""
            
    async def _listen_for_responses(self) -> None:
        """
        Listen for incoming WebSocket messages from ElevenLabs STT.
        """
        if not self._ws:
            return

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = None
                    try:
                        data = msg.json()
                    except Exception:
                        try:
                            data = json.loads(msg.data)
                        except Exception:
                            logger.debug("Received non-json ws text message")
                            continue

                    responses = await self._handle_ws_event(data)
                    if responses:
                        for r in responses:
                            if self._transcript_callback:
                                try:
                                    await self._transcript_callback(r)
                                except Exception:
                                    logger.exception("Error in transcript callback")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("WebSocket error: %s", self._ws.exception())
                    self.emit("error", f"WebSocket error: {self._ws.exception()}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("WebSocket closed by server.")
                    break
        except asyncio.CancelledError:
            logger.debug("WebSocket listener cancelled")
        except Exception as e:
            logger.exception("Error in WebSocket listener: %s", e)
            self.emit("error", str(e))
        finally:
            if self._ws:
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None
            self._ws_task = None

    async def _handle_ws_event(self, data: dict) -> List[STTResponse]:
        """
        Process a single WebSocket event from ElevenLabs STT.

        Args:
            data: JSON-decoded WebSocket message.

        Returns:
            List of STTResponse objects for this event.
        """
        responses: List[STTResponse] = []
        message_type = data.get("message_type")
        logger.debug("Received WS event: %s", message_type)

        if message_type in STT_ERROR_MSGS:
            logger.error("ElevenLabs STT error: %s", data)
            self.emit("error", data)
            return responses
        
        if message_type == "session_started":
            global_event_emitter.emit("speech_session_started")
            return responses

        if message_type == "committed_transcript":
            logger.info("==== Received final transcript event: %s", data)
            text = data.get("text", "")
            clean_text = text.strip()
            confidence = float(data.get("confidence", 0.0))
            now = time.time()

            if clean_text == "":
                global_event_emitter.emit("speech_stopped")
                self._last_final_text = ""
                self._last_final_time = now
                return responses

            resp = STTResponse(
                event_type=SpeechEventType.FINAL,
                data=SpeechData(
                    text=clean_text,
                    confidence=confidence,
                ),
                metadata={"model": self.model_id, "raw_event": data},
            )
            responses.append(resp)

            global_event_emitter.emit("speech_stopped")
            self._last_final_text = clean_text
            self._last_final_time = now
            return responses


        if message_type == "partial_transcript":
            text = data.get("text", "")
            clean_text = text.strip()

            if (
                self._last_final_text
                and clean_text
                and clean_text == self._last_final_text
                and (time.time() - self._last_final_time) < self._duplicate_suppression_window
            ):
                logger.debug("Dropping duplicate partial matching recent final transcript")
                return responses

            resp = STTResponse(
                event_type=SpeechEventType.INTERIM,
                data=SpeechData(
                    text=text,
                    confidence=float(data.get("confidence", 0.0)),
                ),
                metadata={"model": self.model_id, "raw_event": data},
            )
            responses.append(resp)

            if clean_text:
                global_event_emitter.emit("speech_started")

            return responses
        
        

        logger.debug("Ignoring unrecognized message_type: %s", message_type)
        return responses

    async def aclose(self) -> None:
        """
        Close the WebSocket connection and cleanup session resources.

        Cancels the listener task, closes WebSocket and HTTP session,
        and calls the parent class cleanup.
        """
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        if self._session:
            try:
                await self._session.close()
            except Exception:
                pass
            finally:
                self._session = None

        await super().aclose()