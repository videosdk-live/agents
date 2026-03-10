from __future__ import annotations

import asyncio
import json
import os
import base64
from typing import Any, Literal

import aiohttp
import numpy as np

from videosdk.agents import STT, STTResponse, SpeechData, SpeechEventType, global_event_emitter
import logging
logger = logging.getLogger(__name__)
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

SARVAM_STT_STREAMING_URL = "wss://api.sarvam.ai/speech-to-text/ws"
SARVAM_STT_TRANSLATE_URL = "wss://api.sarvam.ai/speech-to-text-translate/ws"
DEFAULT_MODEL = "saaras:v3"

class SarvamAISTT(STT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        language: str = "en-IN",
        input_sample_rate: int = 48000,
        output_sample_rate: int = 16000,
        mode: Literal["transcribe", "translate", "verbatim", "translit", "codemix"] | None = None,
        high_vad_sensitivity:bool|None = None,
        flush_signal:bool|None = None,
        translation:bool = False,
        prompt: str | None = None,
    ) -> None:
        """Initialize the SarvamAI STT plugin with WebSocket support.
        Args:
            api_key: SarvamAI API key
            model: The model to use (default: "saarika:v2.5")
            language: The language code (default: "en-IN")
            input_sample_rate: Input sample rate (default: 48000)
            output_sample_rate: Output sample rate (default: 16000)
            mode: Mode of operation. Only applicable when using the ``saaras:v3``
            high_vad_sensitivity: Whether to use high sensitivity VAD (default: False)
            vad_signal: Whether to send VAD signal (default: False)
            flush_signal: Whether to send flush signal (default: False)
            translation: Whether to enable translation (default: False)
            prompt: Prompt to send to the model (default: None), only applicable when translation is True
        """
        super().__init__()
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is not installed. Please install it with 'pip install scipy'")

        self.api_key = api_key or os.getenv("SARVAMAI_API_KEY")
        if not self.api_key:
            raise ValueError("Sarvam AI API key must be provided either through api_key parameter or SARVAMAI_API_KEY environment variable")

        self.model = model
        self.language = language
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.mode = mode
        self.high_vad_sensitivity = high_vad_sensitivity
        self.flush_signal = flush_signal
        self.translation = translation
        self.prompt = prompt

        # WebSocket related
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ws_task: asyncio.Task | None = None
        self._is_speaking = False
        self._lock = asyncio.Lock()

    async def _ensure_websocket(self) -> aiohttp.ClientWebSocketResponse:
        """Ensure WebSocket connection is established."""
        if self._ws is None or self._ws.closed:
            await self._connect_websocket()
        
        if self._ws is None:
            raise RuntimeError("Failed to establish WebSocket connection")
        
        return self._ws

    async def _connect_websocket(self) -> None:
        """Connect to Sarvam WebSocket API."""
        if self._session is None:
            self._session = aiohttp.ClientSession()

        if self.translation and self.validate_translation_for_model():
            ws_url = SARVAM_STT_TRANSLATE_URL
        else:
            ws_url = SARVAM_STT_STREAMING_URL
        resolved_mode = self.validate_mode_for_model()
        ws_url = f"{ws_url}?language-code={self.language}&model={self.model}&vad_signals=true"
        if resolved_mode is not None:
            ws_url += f"&mode={resolved_mode}"
        if self.high_vad_sensitivity is not None:
            ws_url += f"&high_vad_sensitivity={self.high_vad_sensitivity}"
        if self.flush_signal is not None:
            ws_url += f"&flush_signals={self.flush_signal}"
        if self.prompt is not None and self.translation:
            ws_url += f"&prompt={self.prompt}"

        headers = {"api-subscription-key": self.api_key}
        
        self._ws = await self._session.ws_connect(ws_url, headers=headers)
        
        self._ws_task = asyncio.create_task(self._process_messages())


    async def process_audio(self, audio_frames: bytes, **kwargs: Any) -> None:
        """Process audio frames and send to WebSocket."""
        try:
            resampled_audio = self._resample_audio(audio_frames)
            
            audio_array = np.frombuffer(resampled_audio, dtype=np.int16)
            
            base64_audio = base64.b64encode(audio_array.tobytes()).decode('utf-8')
            
            audio_message = {
                "audio": {
                    "data": base64_audio,
                    "encoding": "audio/wav",
                    "sample_rate": self.output_sample_rate,
                }
            }
            
            ws = await self._ensure_websocket()
            await ws.send_str(json.dumps(audio_message))
            
        except Exception as e:
            logger.error(f"[SarvamAISTT] Error processing audio: {e}")
            self.emit("error", str(e))

    async def _process_messages(self) -> None:
        """Process incoming WebSocket messages."""
        if self._ws is None:
            return
            
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_message(data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"[SarvamAISTT] WebSocket error: {self._ws.exception()}")
                    self.emit("error", f"WebSocket error: {self._ws.exception()}")
                    break
        except Exception as e:
            logger.error(f"[SarvamAISTT] Error in message processing: {e}")
            self.emit("error", str(e))

    async def _handle_message(self, data: dict) -> None:
        """Handle different message types from Sarvam API."""
        msg_type = data.get("type")
        
        if msg_type == "data":
            transcript_data = data.get("data", {})
            transcript_text = transcript_data.get("transcript", "")
            language = transcript_data.get("language_code", self.language)
            duration = transcript_data.get("metrics", {}).get("audio_duration", 0.0)
            
            if transcript_text and self._transcript_callback:
                event = STTResponse(
                    event_type=SpeechEventType.FINAL,
                    data=SpeechData(
                        text=transcript_text,
                        language=language,
                        confidence=1.0,
                        duration=duration
                    )
                )
                await self._transcript_callback(event)
                
        elif msg_type == "events":
            event_data = data.get("data", {})
            signal_type = event_data.get("signal_type")
            
            if signal_type == "START_SPEECH":
                if not self._is_speaking:
                    self._is_speaking = True
                    global_event_emitter.emit("speech_started")
                    
                    
            elif signal_type == "END_SPEECH":
                if self._is_speaking:    
                    flush_message = {"type": "flush"}
                    await self._ws.send_str(json.dumps(flush_message))
                    self._is_speaking = False
                    global_event_emitter.emit("speech_stopped")
                    
        elif msg_type == "error":
            error_info = data.get("error", "Unknown error")
            logger.error(f"[SarvamAISTT] API error: {error_info}")
            self.emit("error", str(error_info))

    def _resample_audio(self, audio_bytes: bytes) -> bytes:
        """Resample audio from input sample rate to output sample rate and convert to mono."""
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

            if self.input_sample_rate != self.output_sample_rate:
                output_length = int(len(mono_audio) * self.output_sample_rate / self.input_sample_rate)
                resampled_data = signal.resample(mono_audio, output_length)
            else:
                resampled_data = mono_audio

            resampled_data = np.clip(resampled_data, -32767, 32767)
            return resampled_data.astype(np.int16).tobytes()

        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return b''

    def validate_mode_for_model(self) -> str | None:
        """Validate and resolve the mode for the current model."""
        if self.model in ["saaras:v3"]:
            return self.mode if self.mode is not None else "transcribe"
        return None
    def validate_translation_for_model(self) -> str | None:
        """Validate and resolve the translation for the current model."""
        if self.translation and self.model in ["saaras:v3", "saaras:v2.5"]:
            return self.translation
        return False


    async def aclose(self) -> None:
        """Close WebSocket connection and cleanup."""
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        
        if self._ws and not self._ws.closed:
            await self._ws.close()
        
        if self._session and not self._session.closed:
            await self._session.close()
        
        await super().aclose()
    
    async def flush(self) -> None:
        """Send flush signal to Sarvam to trigger immediate transcription."""
        if self._ws and not self._ws.closed:
            flush_message = {"type": "flush"}
            await self._ws.send_str(json.dumps(flush_message))
