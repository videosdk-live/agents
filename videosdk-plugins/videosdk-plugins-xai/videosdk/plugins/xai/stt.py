from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Literal, Optional
from urllib.parse import urlencode

import aiohttp
import numpy as np

from videosdk.agents import (
    STT as BaseSTT,
    STTResponse,
    SpeechData,
    SpeechEventType,
)

logger = logging.getLogger(__name__)

XAI_STT_BASE_URL = "wss://api.x.ai/v1/stt"
SUPPORTED_SAMPLE_RATES = {8000, 16000, 22050, 24000, 44100, 48000}
SUPPORTED_ENCODINGS = {"pcm", "mulaw", "alaw"}
_SILENCE_BYTE = {"pcm": b"\x00", "mulaw": b"\xff", "alaw": b"\xd5"}


class XAISTT(BaseSTT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        sample_rate: int = 48000,
        encoding: Literal["pcm", "mulaw", "alaw"] = "pcm",
        interim_results: bool = True,
        endpointing: int = 50,
        language: str | None = "en",
        diarize: bool = False,
        multichannel: bool = False,
        channels: int = 1,
        base_url: str = XAI_STT_BASE_URL,
    ) -> None:
        """Initialize the xAI STT plugin.

        Args:
            api_key: xAI API key. Falls back to XAI_API_KEY env var.
            sample_rate: Audio sample rate in Hz. xAI accepts 8000/16000/22050/24000/44100/48000.
                Defaults to 48000 to match the framework's native input rate.
            encoding: Raw audio encoding. One of "pcm" (signed 16-bit LE), "mulaw", "alaw".
            interim_results: Emit partial transcripts (is_final=false) as they arrive.
            endpointing: Silence duration (ms) before xAI fires utterance-final. Range 0–5000.
                Kept low (50ms default) because the framework's VAD is the primary turn
                detector; flush() injects synthetic silence to force utterance-final, and
                a low threshold means flush latency is short.
            language: BCP-47 language code (e.g. "en", "fr"). Pass None to skip the param.
                xAI transcribes any supported language regardless of this — the value only
                enables Inverse Text Normalization (numbers, currencies in written form).
            diarize: When true, each word in the response includes a `speaker` field.
            multichannel: When true, transcribes each input channel independently. Requires
                interleaved multi-channel audio. When false, input is downmixed to mono.
            channels: Number of input channels (only relevant with multichannel=True).
            base_url: WebSocket endpoint URL.
        """
        super().__init__()

        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "xAI API key must be provided either through the api_key parameter "
                "or the XAI_API_KEY environment variable"
            )

        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {sorted(SUPPORTED_SAMPLE_RATES)}, got {sample_rate}"
            )
        if encoding not in SUPPORTED_ENCODINGS:
            raise ValueError(
                f"encoding must be one of {sorted(SUPPORTED_ENCODINGS)}, got {encoding}"
            )
        if not 0 <= endpointing <= 5000:
            raise ValueError(f"endpointing must be in [0, 5000], got {endpointing}")

        self.sample_rate = sample_rate
        self.encoding = encoding
        self.interim_results = interim_results
        self.endpointing = endpointing
        self.language = language
        self.diarize = diarize
        self.multichannel = multichannel
        self.channels = channels
        self.base_url = base_url

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._server_ready: asyncio.Event = asyncio.Event()
        self._closed = False

    async def process_audio(
        self,
        audio_frames: bytes,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Process audio frames and stream them to xAI's WebSocket STT API."""
        if self._closed:
            return

        if not self._ws:
            await self._connect_ws()
            self._ws_task = asyncio.create_task(self._listen_for_responses())
            try:
                await asyncio.wait_for(self._server_ready.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.emit("error", "Timed out waiting for xAI transcript.created")
                return

        try:
            audio_bytes = audio_frames
            if (
                not self.multichannel
                and self.encoding == "pcm"
                and len(audio_frames) % 4 == 0
            ):
                audio_data = np.frombuffer(audio_frames, dtype=np.int16)
                if audio_data.size > 0 and audio_data.size % 2 == 0:
                    audio_data = (
                        audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    )
                    audio_bytes = audio_data.tobytes()

            await self._ws.send_bytes(audio_bytes)
        except Exception as e:
            logger.error(f"Error sending audio to xAI STT: {e}")
            self.emit("error", str(e))
            await self._reset_connection()

    async def _connect_ws(self) -> None:
        """Open the WebSocket connection to xAI's STT endpoint."""
        if not self._session:
            self._session = aiohttp.ClientSession()

        params: list[tuple[str, str]] = [
            ("sample_rate", str(self.sample_rate)),
            ("encoding", self.encoding),
            ("interim_results", str(self.interim_results).lower()),
            ("endpointing", str(self.endpointing)),
            ("diarize", str(self.diarize).lower()),
            ("multichannel", str(self.multichannel).lower()),
            ("channels", str(1 if not self.multichannel else self.channels)),
        ]
        if self.language:
            params.append(("language", self.language))

        ws_url = f"{self.base_url}?{urlencode(params)}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        self._server_ready = asyncio.Event()

        try:
            self._ws = await self._session.ws_connect(
                ws_url, headers=headers, heartbeat=30.0
            )
        except Exception as e:
            logger.error(f"Error connecting to xAI STT WebSocket: {e}")
            raise

    async def _listen_for_responses(self) -> None:
        """Background task that reads transcript events from the WebSocket."""
        if not self._ws:
            return

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = msg.json()
                    except Exception as e:
                        logger.error(f"Failed to parse xAI STT message: {e}")
                        continue

                    event_type = data.get("type")
                    if event_type == "transcript.created":
                        self._server_ready.set()
                    elif event_type == "transcript.partial":
                        response = self._handle_partial(data)
                        if response and self._transcript_callback:
                            await self._transcript_callback(response)
                    elif event_type == "transcript.done":
                        response = self._handle_partial(data)
                        if response and self._transcript_callback:
                            await self._transcript_callback(response)
                    elif event_type == "error":
                        message = data.get("message", "unknown error")
                        logger.error(f"xAI STT error event: {message}")
                        self.emit("error", message)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    err = self._ws.exception()
                    logger.error(f"xAI STT WebSocket error: {err}")
                    self.emit("error", f"WebSocket error: {err}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error in xAI STT listener: {e}")
            self.emit("error", f"Error in WebSocket listener: {e}")
        finally:
            if self._ws and not self._ws.closed:
                await self._ws.close()
            self._ws = None
            self._server_ready.clear()

    def _handle_partial(self, event: dict) -> Optional[STTResponse]:
        """Map an xAI transcript event to an STTResponse."""
        text = event.get("text", "")
        if not text:
            return None

        is_final = bool(event.get("is_final", False))
        speech_final = bool(event.get("speech_final", False))
        event_is_done = event.get("type") == "transcript.done"
        event_type = (
            SpeechEventType.FINAL
            if (is_final and speech_final) or event_is_done
            else SpeechEventType.INTERIM
        )

        words = event.get("words") or []
        start = event.get("start", 0.0) or 0.0
        duration = event.get("duration", 0.0) or 0.0
        if words:
            start_time = float(words[0].get("start", start))
            end_time = float(words[-1].get("end", start + duration))
        else:
            start_time = float(start)
            end_time = float(start) + float(duration)

        return STTResponse(
            event_type=event_type,
            data=SpeechData(
                text=text,
                language=self.language,
                confidence=0.0,
                start_time=start_time,
                end_time=end_time,
                duration=float(duration),
            ),
            metadata={
                "is_final": is_final,
                "speech_final": speech_final,
                "channel_index": event.get("channel_index"),
            },
        )

    async def flush(self) -> None:
        """Force xAI to emit the current utterance-final transcript.

        We inject a short chunk of silence bytes — once the configured
        endpointing window elapses, xAI fires `transcript.partial` with
        speech_final=true, which we map to SpeechEventType.FINAL.
        """
        if self._closed or not self._ws or self._ws.closed:
            return
        if not self._server_ready.is_set():
            return

        try:
            silence_ms = max(self.endpointing + 50, 100)
            bytes_per_sample = 2 if self.encoding == "pcm" else 1
            channel_count = self.channels if self.multichannel else 1
            n_bytes = int(
                self.sample_rate * bytes_per_sample * channel_count * silence_ms / 1000
            )
            silence = _SILENCE_BYTE[self.encoding] * n_bytes
            await self._ws.send_bytes(silence)
        except Exception as e:
            logger.warning(f"xAI STT flush failed: {e}")

    async def _reset_connection(self) -> None:
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass
            self._ws_task = None
        self._server_ready.clear()

    async def aclose(self) -> None:
        """Cleanup resources."""
        self._closed = True

        if self._ws and not self._ws.closed:
            try:
                await self._ws.send_str(json.dumps({"type": "audio.done"}))
            except Exception:
                pass

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass
            self._ws_task = None

        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = None

        if self._session:
            await self._session.close()
            self._session = None

        await super().aclose()
