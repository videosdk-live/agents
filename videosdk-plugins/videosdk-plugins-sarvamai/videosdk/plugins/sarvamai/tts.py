from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from typing import Any, AsyncIterator, Optional,Literal

import aiohttp
import httpx
from videosdk.agents import TTS
import logging

logger = logging.getLogger(__name__)

SARVAM_SAMPLE_RATE = 24000
SARVAM_CHANNELS = 1
DEFAULT_MODEL = "bulbul:v2"
DEFAULT_SPEAKER = "anushka"
DEFAULT_LANGUAGE = "en-IN"
SARVAM_TTS_URL_STREAMING = "wss://api.sarvam.ai/text-to-speech/ws"
SARVAM_TTS_URL_HTTP = "https://api.sarvam.ai/text-to-speech"

PITCH_SUPPORTED_MODELS = {"bulbul:v2"}
PITCH_RANGE = (-0.75, 0.75)
PITCH_DEFAULT = 0.0

LOUDNESS_SUPPORTED_MODELS = {"bulbul:v2"}
LOUDNESS_RANGE = (0.3, 3.0)
LOUDNESS_DEFAULT = 1.0

PACE_RANGES = {
    "bulbul:v2": (0.3, 3.0),
    "bulbul:v3": (0.5, 2.0),
    "bulbul:v3-beta": (0.5, 2.0),
}
PACE_DEFAULT_RANGE = (0.5, 2.0)
PACE_DEFAULT = 1.0

TEMPERATURE_SUPPORTED_MODELS = {"bulbul:v3","bulbul:v3-beta"}
TEMPERATURE_RANGE = (0.01, 1.0)
TEMPERATURE_DEFAULT = 0.6

ENABLE_PREPROCESSING_SUPPORTED_MODELS = {"bulbul:v2"}
ENABLE_PREPROCESSING_DEFAULT = False

SarvamAITTSModel = Literal["bulbul:v2", "bulbul:v3-beta", "bulbul:v3"]
SarvamTTSOutputAudioBitrate = Literal["32k", "64k", "96k", "128k", "192k"]

ALLOWED_OUTPUT_AUDIO_BITRATES: set[str] = {"32k", "64k", "96k", "128k", "192k"}

def _pace_range(model: str) -> tuple[float, float]:
    return PACE_RANGES.get(model, PACE_DEFAULT_RANGE)


class SarvamAITTS(TTS):
    """
    A unified Sarvam.ai Text-to-Speech (TTS) plugin that supports both real-time
    streaming via WebSockets and batch synthesis via HTTP. This version is optimized
    for robust, long-running sessions and responsive non-streaming playback.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: SarvamAITTSModel= DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        speaker: str = DEFAULT_SPEAKER,
        enable_streaming: bool = True,
        sample_rate: int = SARVAM_SAMPLE_RATE,
        output_audio_codec: str = "linear16",
        pitch: float | None = PITCH_DEFAULT,
        pace: float | None = PACE_DEFAULT,
        loudness: float | None = LOUDNESS_DEFAULT,
        temperature:float| None = 0.6,
        output_audio_bitrate: SarvamTTSOutputAudioBitrate | str = "128k",
        min_buffer_size: int = 50,
        max_chunk_length: int = 150,
        enable_preprocessing: bool = False,
    ) -> None:
        """
        Initializes the SarvamAITTS plugin.

        Args:
            api_key (Optional[str]): The Sarvam.ai API key. If not provided, it will
                be read from the SARVAMAI_API_KEY environment variable.
            model (str): The TTS model to use (e.g. ``"bulbul:v2"``, ``"bulbul:v3"``).
            language (str): The target language code (e.g., "en-IN").
            speaker (str): The desired speaker for the voice.
            enable_streaming (bool): If True, uses WebSockets for low-latency streaming.
                If False, uses HTTP for batch synthesis.
            sample_rate (int): The audio sample rate.
            output_audio_codec (str): The desired output audio codec.
            pitch (float | None): Pitch of the voice. Only for ``bulbul:v2``.
                Range [-0.75, 0.75]. Default 0.0. Set to ``None`` to omit.
            pace (float | None): Pace of the voice.
                ``bulbul:v2`` → [0.3, 3.0]; ``bulbul:v3`` → [0.5, 2.0]. Default 1.0.
                Set to ``None`` to omit.
            loudness (float | None): Loudness of the voice. Only for ``bulbul:v2``.
                Range [0.3, 3.0]. Default 1.0. Set to ``None`` to omit.
            temperature: Sampling temperature (0.01 to 1.0), used for v3 and v3-beta
            output_audio_bitrate: Output audio bitrate
            min_buffer_size: Minimum character length for flushing
            max_chunk_length: Maximum chunk length for sentence splitting
            enable_preprocessing (bool): Controls whether normalization of English words and numeric entities (e.g., numbers, dates) is performed. 
                Set to true for better handling of mixed-language text. Default False. Only for ``bulbul:v2``.
        """
        super().__init__(sample_rate=sample_rate, num_channels=SARVAM_CHANNELS)

        self.api_key = api_key or os.getenv("SARVAMAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Sarvam AI API key required. Provide either:\n"
                "1. api_key parameter, OR\n"
                "2. SARVAMAI_API_KEY environment variable"
            )

        self.model = model
        self.language = language
        self.speaker = speaker
        self.enable_streaming = enable_streaming
        self.output_audio_codec = output_audio_codec
        self.base_url_ws = SARVAM_TTS_URL_STREAMING
        self.base_url_http = SARVAM_TTS_URL_HTTP

        # Validate and store speech parameters
        self.pitch = self._validate_pitch(pitch, model)
        self.pace = self._validate_pace(pace, model)
        self.loudness = self._validate_loudness(loudness, model)
        self.temperature = self._validate_temperature(temperature, model)
        self.enable_preprocessing = self._validate_enable_preprocessing(enable_preprocessing, model)

        if output_audio_bitrate is not None:
            if output_audio_bitrate not in ALLOWED_OUTPUT_AUDIO_BITRATES:
                raise ValueError(
                    "output_audio_bitrate must be one of "
                    f"{', '.join(sorted(ALLOWED_OUTPUT_AUDIO_BITRATES))}"
                )
            self.output_audio_bitrate = output_audio_bitrate

        if min_buffer_size is not None:
            if not 30 <= min_buffer_size <= 200:
                raise ValueError("min_buffer_size must be between 30 and 200")
            self.min_buffer_size = min_buffer_size

        if max_chunk_length is not None:
            if not 50 <= max_chunk_length <= 500:
                raise ValueError("max_chunk_length must be between 50 and 500")
            self.max_chunk_length = max_chunk_length

        self._ws_session: aiohttp.ClientSession | None = None
        self._ws_connection: aiohttp.ClientWebSocketResponse | None = None
        self._receive_task: asyncio.Task | None = None
        self._connection_lock = asyncio.Lock()

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=30.0, write=5.0, pool=5.0),
            follow_redirects=True,
        )

        self._interrupted = False
        self._first_chunk_sent = False
        self.ws_count = 0

    @staticmethod
    def _validate_pitch(pitch: float | None, model: str) -> float | None:
        """ Validate pitch for the given model. """
        if pitch is None:
            return None
        if model not in PITCH_SUPPORTED_MODELS:
            logger.warning(
                f"pitch is not supported for model '{model}' "
                f"(supported: {PITCH_SUPPORTED_MODELS}). Ignoring pitch value."
            )
            return None
        lo, hi = PITCH_RANGE
        if not lo <= pitch <= hi:
            raise ValueError(
                f"pitch must be between {lo} and {hi} for model '{model}', got {pitch}."
            )
        return pitch

    @staticmethod
    def _validate_pace(pace: float | None, model: str) -> float | None:
        """ Validate pace for the given model. """
        if pace is None:
            return None
        lo, hi = _pace_range(model)
        if not lo <= pace <= hi:
            raise ValueError(
                f"pace must be between {lo} and {hi} for model '{model}', got {pace}."
            )
        return pace

    @staticmethod
    def _validate_loudness(loudness: float | None, model: str) -> float | None:
        """ Validate loudness for the given model. """
        if loudness is None:
            return None
        if model not in LOUDNESS_SUPPORTED_MODELS:
            logger.warning(
                f"loudness is not supported for model '{model}' "
                f"(supported: {LOUDNESS_SUPPORTED_MODELS}). Ignoring loudness value."
            )
            return None
        lo, hi = LOUDNESS_RANGE
        if not lo <= loudness <= hi:
            raise ValueError(
                f"loudness must be between {lo} and {hi} for model '{model}', got {loudness}."
            )
        return loudness
    @staticmethod
    def _validate_temperature(temperature: float | None, model: str) -> float | None:
        """ Validate temperature for the given model. """
        if temperature is None:
            return None
        if model not in TEMPERATURE_SUPPORTED_MODELS:
            logger.warning(
                f"temperature is not supported for model '{model}' "
                f"(supported: {TEMPERATURE_SUPPORTED_MODELS}). Ignoring temperature value."
            )
            return None
        lo, hi = TEMPERATURE_RANGE
        if not lo <= temperature <= hi:
            raise ValueError(
                f"temperature must be between {lo} and {hi} for model '{model}', got {temperature}."
            )
        return temperature

    @staticmethod
    def _validate_enable_preprocessing(enable_preprocessing: bool | None, model: str) -> bool | None:
        """ Validate enable_preprocessing for the given model. """
        if enable_preprocessing is None:
            return None
        if model not in ENABLE_PREPROCESSING_SUPPORTED_MODELS:
            logger.warning(
                f"enable_preprocessing is not supported for model '{model}' "
                f"(supported: {ENABLE_PREPROCESSING_SUPPORTED_MODELS}). Ignoring enable_preprocessing value."
            )
            return None
        return enable_preprocessing

    def _build_speech_params(self) -> dict[str, Any]:
        """ Returns a dict containing only the speech keys whose values are not None """
        self.validate_parameters()
        params: dict[str, Any] = {}
        if self.pitch is not None:
            params["pitch"] = self.pitch
        if self.pace is not None:
            params["pace"] = self.pace
        if self.loudness is not None:
            params["loudness"] = self.loudness
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.enable_preprocessing is not None:
            params["enable_preprocessing"] = self.enable_preprocessing
        return params
    
    def reset_first_audio_tracking(self) -> None:
        """Resets tracking for the first audio chunk latency."""
        self._first_chunk_sent = False

    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        *,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Main entry point for synthesizing audio. Routes to either streaming
        or batch (HTTP) mode. The HTTP mode uses a smart buffering strategy
        to balance responsiveness and audio quality.
        """
        try:
            if not self.audio_track or not self.loop:
                logger.error("error", "Audio track or event loop not initialized")
                return

            self.language = language or self.language
            self.speaker = speaker or self.speaker
            self._interrupted = False
            self.reset_first_audio_tracking()

            if self.enable_streaming:
                await self._stream_synthesis(text)
            else:
                if isinstance(text, str):
                    if text.strip():
                        await self._http_synthesis(text)
                else:
                    chunk_buffer = []
                    HTTP_CHUNK_BUFFER_SIZE = 4
                    LLM_PAUSE_TIMEOUT = 1.0 

                    text_iterator = text.__aiter__()
                    while not self._interrupted:
                        try:
                            chunk = await asyncio.wait_for(text_iterator.__anext__(), timeout=LLM_PAUSE_TIMEOUT)
                            
                            if chunk and chunk.strip():
                                chunk_buffer.append(chunk)
                            
                            if len(chunk_buffer) >= HTTP_CHUNK_BUFFER_SIZE:
                                combined_text = "".join(chunk_buffer)
                                await self._http_synthesis(combined_text)
                                chunk_buffer.clear()

                        except asyncio.TimeoutError:
                            if chunk_buffer:
                                combined_text = "".join(chunk_buffer)
                                await self._http_synthesis(combined_text)
                                chunk_buffer.clear()
                        
                        except StopAsyncIteration:
                            if chunk_buffer:
                                combined_text = "".join(chunk_buffer)
                                await self._http_synthesis(combined_text)
                            break

        except Exception as e:
            logger.error("error", f"Sarvam TTS synthesis failed: {e}")

    async def _stream_synthesis(self, text: AsyncIterator[str] | str) -> None:
        """
        Manages the WebSocket synthesis workflow, ensuring a fresh connection
        for each synthesis task to guarantee reliability.
        """
        try:
            # await self._close_ws_resources()
            await self._ensure_ws_connection()
            

            if isinstance(text, str):
                async def _str_iter():
                    yield text
                text_iter = _str_iter()
            else:
                text_iter = text

            await self._send_text_chunks(text_iter)
        except Exception as e:
            logger.error("error", f"WebSocket streaming failed: {e}. Trying HTTP fallback.")
            try:
                full_text = ""
                if isinstance(text, str):
                    full_text = text
                else:
                    async for chunk in text:
                        full_text += chunk
                
                if full_text.strip():
                    await self._http_synthesis(full_text.strip())
            except Exception as http_e:
                logger.error("error", f"HTTP fallback also failed: {http_e}")

    async def _ensure_ws_connection(self) -> None:
        """Establishes and maintains a persistent WebSocket connection."""
        async with self._connection_lock:
            if self._ws_connection and not self._ws_connection.closed:
                return
            try:
                self._ws_session = aiohttp.ClientSession()
                headers = {"Api-Subscription-Key": self.api_key}
                base_url = f"{self.base_url_ws}?model={self.model}"
                self._ws_connection = await asyncio.wait_for(
                    self._ws_session.ws_connect(
                        base_url, headers=headers, heartbeat=20
                    ),
                    timeout=10.0,
                )
                self._receive_task = asyncio.create_task(self._recv_loop())
                await self._send_initial_config()
                self.ws_count = self.ws_count + 1
                logger.info(f"WS connection numbers: {self.ws_count}")
            except Exception as e:
                logger.error("error", f"Failed to connect to WebSocket: {e}")
                raise

    async def _send_initial_config(self) -> None:
        """Sends the initial configuration message to the WebSocket server."""
        config_payload = {
            "type": "config",
            "data": {
                "target_language_code": self.language,
                "speaker": self.speaker,
                "speech_sample_rate": str(self.sample_rate),
                "output_audio_codec": self.output_audio_codec,
                "output_audio_bitrate": self.output_audio_bitrate,
                "min_buffer_size": self.min_buffer_size,
                "max_chunk_length": self.max_chunk_length,
            },
        }

        config_payload["data"].update(self._build_speech_params())

        if self._ws_connection:
            await self._ws_connection.send_str(json.dumps(config_payload))

    async def _send_text_chunks(self, text_iterator: AsyncIterator[str]):
        """Sends text to the WebSocket, chunking it by word count or time."""
        if not self._ws_connection:
            raise ConnectionError("WebSocket is not connected.")
        try:
            buffer = []
            MIN_WORDS, MAX_DELAY = 4, 1.0
            last_send_time = asyncio.get_event_loop().time()

            async for text_chunk in text_iterator:
                if self._interrupted:
                    break

                words = re.findall(r"\b[\w'-]+\b", text_chunk)
                if not words:
                    continue

                buffer.extend(words)
                now = asyncio.get_event_loop().time()

                if len(buffer) >= MIN_WORDS or (now - last_send_time > MAX_DELAY):
                    combined_text = " ".join(buffer).strip()
                    if combined_text:
                        payload = {"type": "text", "data": {"text": combined_text}}
                        await self._ws_connection.send_str(json.dumps(payload))
                    buffer.clear()
                    last_send_time = now

            if buffer and not self._interrupted:
                combined_text = " ".join(buffer).strip()
                if combined_text:
                    payload = {"type": "text", "data": {"text": combined_text}}
                    await self._ws_connection.send_str(json.dumps(payload))
                    if not self._first_chunk_sent and hasattr(self, '_first_audio_callback') and self._first_audio_callback:
                        self._first_chunk_sent = True
                        asyncio.create_task(self._first_audio_callback())

            if not self._interrupted:
                await self._ws_connection.send_str(json.dumps({"type": "flush"}))
        except Exception as e:
            logger.error("error", f"Failed to send text chunks via WebSocket: {e}")

    async def _recv_loop(self):
        """Continuously listens for and processes incoming WebSocket messages."""
        try:
            while self._ws_connection and not self._ws_connection.closed:
                msg = await self._ws_connection.receive()
                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(msg.data)
                msg_type = data.get("type")

                if msg_type == "audio":

                    if not self._first_chunk_sent and hasattr(self, '_first_audio_callback') and self._first_audio_callback:
                        self._first_chunk_sent = True
                        asyncio.create_task(self._first_audio_callback())
                    
                    await self._handle_audio_data(data.get("data"))
                
                elif msg_type == "event" and data.get("data", {}).get("event_type") == "final":
                    logger.error("done", "TTS completed")
                
                elif msg_type == "error":
                    error_msg = data.get("data", {}).get("message", "Unknown WS error")
                    logger.error("error", f"Sarvam WebSocket error: {error_msg}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("error", f"WebSocket receive loop error: {e}")

    async def _handle_audio_data(self, audio_data: Optional[dict[str, Any]]):
        """Processes audio data received from the WebSocket."""
        if not audio_data or self._interrupted:
            return

        audio_b64 = audio_data.get("audio")
        if not audio_b64:
            return

        try:
            audio_bytes = base64.b64decode(audio_b64)
            if not self.audio_track:
                return


            await self.audio_track.add_new_bytes(audio_bytes)
        except Exception as e:
            logger.error("error", f"Failed to process WebSocket audio: {e}")


    async def _reinitialize_http_client(self):
        """Safely closes the current httpx client and creates a new one."""
        logger.info("Re-initializing HTTP client.")
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=30.0, write=5.0, pool=5.0),
            follow_redirects=True,
        )

    async def _http_synthesis(self, text: str) -> None:
        """Performs TTS synthesis using HTTP with a retry for connection errors."""
        payload = { "text": text, "target_language_code": self.language, "speaker": self.speaker, "speech_sample_rate": str(self.sample_rate), "model": self.model, "output_audio_codec": self.output_audio_codec }
        
        payload.update(self._build_speech_params())

        headers = { "Content-Type": "application/json", "api-subscription-key": self.api_key }
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                if self._http_client.is_closed:
                    await self._reinitialize_http_client()
                response = await self._http_client.post(self.base_url_http, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                if not data.get("audios"):
                    logger.error("error", f"No audio data in HTTP response: {data}")
                    return
                audio_b64 = data["audios"][0]
                audio_bytes = base64.b64decode(audio_b64)
                if not self._first_chunk_sent and self._first_audio_callback:
                    self._first_chunk_sent = True
                    await self._first_audio_callback()

                await self._stream_http_audio(audio_bytes)
                return
            except httpx.HTTPStatusError as e:
                logger.error("error", f"HTTP error: {e.response.status_code} - {e.response.text}")
                logger.info(response)
                raise e
            except (httpx.NetworkError, httpx.ConnectError, httpx.ReadTimeout) as e:
                logger.warning(f"HTTP connection error on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    await self._reinitialize_http_client()
                    continue
                else:
                    logger.error("error", f"HTTP synthesis failed after {max_attempts} connection attempts.")
                    raise e
            except Exception as e:
                logger.error("error", f"An unexpected HTTP synthesis error occurred: {e}")
                raise e

    async def _stream_http_audio(self, audio_bytes: bytes) -> None:
        """
        Streams decoded HTTP audio bytes to the audio track by sending two
        20ms chunks at a time (a 40ms block) to ensure real-time playback.
        """
        single_chunk_size = int(self.sample_rate * self.num_channels * 2 * 20 / 1000)
        
        block_size = single_chunk_size * 2
        
        raw_audio = self._remove_wav_header(audio_bytes)

        for i in range(0, len(raw_audio), block_size):
            if self._interrupted:
                break
            
            block = raw_audio[i : i + block_size]

            if 0 < len(block) < block_size:
                block += b"\x00" * (block_size - len(block))

                
            if self.audio_track:
                asyncio.create_task(self.audio_track.add_new_bytes(block))

    def validate_parameters(self, **kwargs: Any) -> None:
        """
        Validates a set of TTS parameters against the current model's constraints.
        Raises ``ValueError`` for any out-of-range value.
        """
        self._validate_pitch(self.pitch, self.model)
        self._validate_pace(self.pace, self.model)
        self._validate_loudness(self.loudness, self.model)
        self._validate_temperature(self.temperature, self.model)
        self._validate_enable_preprocessing(self.enable_preprocessing, self.model)

    def _remove_wav_header(self, audio_bytes: bytes) -> bytes:
        """Removes the WAV header if present."""
        if audio_bytes.startswith(b"RIFF"):
            data_pos = audio_bytes.find(b"data")
            if data_pos != -1:
                return audio_bytes[data_pos + 8:]
        return audio_bytes

    async def interrupt(self) -> None:
        """Interrupts any ongoing TTS synthesis."""
        self._interrupted = True
        if self.audio_track:
            self.audio_track.interrupt()
        
    async def _close_ws_resources(self) -> None:
        """Helper to clean up all WebSocket-related resources."""
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
        if self._ws_connection and not self._ws_connection.closed:
            await self._ws_connection.close()
        if self._ws_session and not self._ws_session.closed:
            await self._ws_session.close()
        self._receive_task = self._ws_connection = self._ws_session = None

    async def aclose(self) -> None:
        """Gracefully closes all connections and cleans up resources."""
        self._interrupted = True
        await self._close_ws_resources()
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
        await super().aclose()
