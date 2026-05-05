from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any, AsyncIterator, Optional

import aiohttp
import httpx

from videosdk.agents import TTS

logger = logging.getLogger(__name__)

INWORLD_SAMPLE_RATE = 24000
INWORLD_CHANNELS = 1
INWORLD_TTS_HTTP_ENDPOINT = "https://api.inworld.ai/tts/v1/voice:stream"
INWORLD_TTS_WSS_ENDPOINT = "wss://api.inworld.ai/tts/v1/voice:streamBidirectional"

DEFAULT_MODEL = "inworld-tts-1.5-max"
DEFAULT_VOICE = "Sarah"
DEFAULT_TEMPERATURE = 0.8


class InworldAITTS(TTS):
    """
    Inworld AI Text-to-Speech plugin.

    Supports two transports:

      - WebSocket (default, ``enable_streaming=True``) — uses the Inworld
        bidirectional streaming endpoint with per-call contexts. Server-side
        ``autoMode`` handles intelligent flushing for low-latency, full-quality
        prosody. Best fit for the agent pipeline because the server combines
        sentence chunks into one continuous synthesis context.

      - HTTP (``enable_streaming=False``) — uses ``voice:stream`` with chunked
        transfer encoding. One POST per ``synthesize()`` call.

    Both transports forward incoming text chunks verbatim and never re-segment
    client-side: the upstream tokenizer / text filter already delivers
    sentence-sized, verbalized segments, and re-tokenizing here would split
    currency / multi-digit tokens (``$50,000``, ``₹50,00,000``) and break
    prosody at chunk boundaries.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str = DEFAULT_MODEL,
        voice_id: str = DEFAULT_VOICE,
        temperature: float = DEFAULT_TEMPERATURE,
        sample_rate: int = INWORLD_SAMPLE_RATE,
        enable_streaming: bool = True,
        auto_mode: bool = True,
        max_buffer_delay_ms: int | None = None,
        buffer_char_threshold: int | None = None,
        apply_text_normalization: str | None = None,
        speaking_rate: float | None = None,
    ) -> None:
        """Initialize the InworldAI TTS plugin.

        Audio is always requested as raw 16-bit signed LE PCM (Inworld
        ``"PCM"`` encoding) so it can be forwarded directly to the agent's
        audio track without header parsing.

        Args:
            api_key: Inworld API key. Falls back to ``INWORLD_API_KEY`` env var.
            model_id: TTS model id (e.g. ``"inworld-tts-1.5-max"``, ``"inworld-tts-1"``).
            voice_id: Voice id (e.g. ``"Sarah"``, ``"Hades"``, ``"Ashley"``).
            temperature: Sampling temperature, 0.0–2.0. Defaults to 0.8.
            sample_rate: Output sample rate in Hz. Defaults to 24000.
            enable_streaming: ``True`` (default) → WebSocket bidirectional
                streaming. ``False`` → HTTP streaming POST.
            auto_mode: WSS only — when ``True`` (default), the server controls
                buffer flushing for minimal latency. Recommended when text
                arrives as full sentences/phrases (which is what the agent
                pipeline produces).
            max_buffer_delay_ms: WSS only — server-side max wait time before
                flushing accumulated text. ``None`` = unbounded.
            buffer_char_threshold: WSS only — server-side character count that
                auto-triggers flushing. Defaults to 1000 server-side; cannot
                exceed 1000.
            apply_text_normalization: ``"ON"``, ``"OFF"``, or ``None`` (server
                decides). When on, ``Dr. Smith`` → ``Doctor Smith`` and
                ``3/10/25`` → ``March tenth, twenty twenty-five``.
            speaking_rate: Speed multiplier in the range [0.5, 1.5]. ``None``
                uses the voice's natural rate (1.0).
        """
        super().__init__(sample_rate=sample_rate, num_channels=INWORLD_CHANNELS)

        self.api_key = api_key or os.getenv("INWORLD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "InworldAI API key must be provided either through:\n"
                "1. api_key parameter, OR\n"
                "2. INWORLD_API_KEY environment variable"
            )

        self.model_id = model_id
        self.voice_id = voice_id
        self.temperature = temperature
        self.enable_streaming = enable_streaming
        self.auto_mode = auto_mode
        self.max_buffer_delay_ms = max_buffer_delay_ms
        self.buffer_char_threshold = buffer_char_threshold
        self.apply_text_normalization = apply_text_normalization
        self.speaking_rate = speaking_rate

        self.audio_track = None
        self.loop = None
        self._first_chunk_sent = False
        self._interrupted = False

        self._auth_header = f"Basic {self.api_key}"

        # HTTP transport
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=30.0, write=5.0, pool=5.0),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=50,
                keepalive_expiry=120,
            ),
        )

        # WSS transport
        self._ws_session: aiohttp.ClientSession | None = None
        self._ws_connection: aiohttp.ClientWebSocketResponse | None = None
        self._connection_lock = asyncio.Lock()
        self._receive_task: asyncio.Task | None = None
        self._context_futures: dict[str, asyncio.Future[None]] = {}

    def reset_first_audio_tracking(self) -> None:
        """Reset the first audio tracking state for next TTS task"""
        self._first_chunk_sent = False


    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        voice_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Convert text to speech and stream PCM frames to the audio track.

        For ``AsyncIterator`` inputs, chunks are forwarded verbatim into a
        single synthesis context (WSS) or accumulated into a single request
        (HTTP). Either way, no client-side re-segmentation happens — currency
        tokens, comma-grouped digits, and Devanagari verbalizations span a
        single coherent synthesis call.
        """
        try:
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or event loop not set")
                return

            self._interrupted = False

            if self.enable_streaming:
                await self._stream_synthesis(text, voice_id)
            else:
                await self._http_synthesis(text, voice_id)

        except Exception as e:
            self.emit("error", f"InworldAI TTS synthesis failed: {str(e)}")

    async def aclose(self) -> None:
        """Cleanup resources"""
        self._interrupted = True
        await self._close_ws_resources()
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
        await super().aclose()

    async def interrupt(self) -> None:
        """Interrupt the TTS process"""
        self._interrupted = True
        if self.audio_track:
            self.audio_track.interrupt()
        await self._close_ws_resources()

    # ── WSS path ───────────────────────────────────────────────────────────

    async def _stream_synthesis(
        self,
        text: AsyncIterator[str] | str,
        voice_id: Optional[str],
    ) -> None:
        """WebSocket-based synthesis using the Inworld bidirectional endpoint.

        Per call: open a fresh context with audio config, forward the text
        chunks verbatim, send a trailing ``flush_context`` (and ``close_context``),
        then await ``flushCompleted`` to know synthesis finished.
        """
        context_id = ""
        try:
            await self._ensure_ws_connection()
            if not self._ws_connection:
                raise RuntimeError("WebSocket connection is not available.")

            context_id = os.urandom(8).hex()
            done_future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
            self._context_futures[context_id] = done_future

            await self._send_create_context(context_id, voice_id)

            if isinstance(text, str):
                if text and text.strip():
                    await self._send_text(context_id, text)
            else:
                async for chunk in text:
                    if self._interrupted:
                        break
                    if not chunk or not chunk.strip():
                        continue
                    await self._send_text(context_id, chunk)

            if not self._interrupted:
                await self._send_flush(context_id)

            try:
                await done_future
            except asyncio.CancelledError:
                return
            finally:
                if not self._interrupted:
                    await self._send_close(context_id)

        except Exception as e:
            future = self._context_futures.get(context_id)
            if future and not future.done():
                future.set_exception(e)
            self.emit("error", f"InworldAI WSS synthesis failed: {str(e)}")
            raise
        finally:
            if context_id and context_id in self._context_futures:
                del self._context_futures[context_id]

    async def _ensure_ws_connection(self) -> None:
        """Open a WebSocket connection if one isn't already alive."""
        async with self._connection_lock:
            if self._ws_connection and not self._ws_connection.closed:
                return

            if self._receive_task and not self._receive_task.done():
                self._receive_task.cancel()
            if self._ws_connection:
                await self._ws_connection.close()
            if self._ws_session:
                await self._ws_session.close()

            try:
                self._ws_session = aiohttp.ClientSession()
                self._ws_connection = await asyncio.wait_for(
                    self._ws_session.ws_connect(
                        INWORLD_TTS_WSS_ENDPOINT,
                        headers={"Authorization": self._auth_header},
                        heartbeat=30.0,
                    ),
                    timeout=10.0,
                )
                self._receive_task = asyncio.create_task(self._recv_loop())
            except Exception as e:
                self.emit("error", f"Failed to connect to Inworld WSS: {e}")
                raise

    def _build_audio_config(self) -> dict[str, Any]:
        # Always request raw PCM — no header parsing, direct push to track.
        cfg: dict[str, Any] = {
            "audioEncoding": "PCM",
            "sampleRateHertz": self._sample_rate,
        }
        if self.speaking_rate is not None:
            cfg["speakingRate"] = self.speaking_rate
        return cfg

    async def _send_create_context(
        self,
        context_id: str,
        voice_id: Optional[str],
    ) -> None:
        create_payload: dict[str, Any] = {
            "voiceId": voice_id or self.voice_id,
            "modelId": self.model_id,
            "audioConfig": self._build_audio_config(),
            "temperature": self.temperature,
            "autoMode": self.auto_mode,
        }
        if self.max_buffer_delay_ms is not None:
            create_payload["maxBufferDelayMs"] = self.max_buffer_delay_ms
        if self.buffer_char_threshold is not None:
            create_payload["bufferCharThreshold"] = self.buffer_char_threshold
        if self.apply_text_normalization is not None:
            create_payload["applyTextNormalization"] = self.apply_text_normalization

        await self._ws_send({"create": create_payload, "contextId": context_id})

    async def _send_text(self, context_id: str, text: str) -> None:
        for piece in self._slice_for_inworld(text, max_len=1000):
            await self._ws_send({
                "send_text": {"text": piece},
                "contextId": context_id,
            })

    async def _send_flush(self, context_id: str) -> None:
        await self._ws_send({"flush_context": {}, "contextId": context_id})

    async def _send_close(self, context_id: str) -> None:
        await self._ws_send({"close_context": {}, "contextId": context_id})

    async def _ws_send(self, payload: dict[str, Any]) -> None:
        if self._ws_connection and not self._ws_connection.closed:
            await self._ws_connection.send_str(json.dumps(payload))

    @staticmethod
    def _slice_for_inworld(text: str, max_len: int) -> list[str]:
        """Split a string into ≤ max_len pieces without breaking words.

        Almost always returns a single piece (sentence chunks from the
        framework are well under 1000 chars); the splitter only kicks in for
        the rare case where a single ``send_text`` would exceed Inworld's
        per-message cap.
        """
        if len(text) <= max_len:
            return [text]
        pieces: list[str] = []
        remaining = text
        while len(remaining) > max_len:
            cut = remaining.rfind(" ", 0, max_len)
            if cut <= 0:
                cut = max_len
            pieces.append(remaining[:cut])
            remaining = remaining[cut:].lstrip()
        if remaining:
            pieces.append(remaining)
        return pieces

    async def _recv_loop(self) -> None:
        """Long-running receive task: parses every server message and routes
        audio bytes to the audio track + signals flush completion."""
        try:
            while self._ws_connection and not self._ws_connection.closed:
                msg = await self._ws_connection.receive()
                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                result = data.get("result") or {}
                context_id = result.get("contextId")

                # Top-level error (e.g. auth, malformed payload) — fail any
                # in-flight contexts so synthesize() doesn't hang.
                if "error" in data:
                    err = data["error"]
                    msg_str = err.get("message", "Unknown error") if isinstance(err, dict) else str(err)
                    self._fail_all_pending(RuntimeError(f"Inworld API error: {msg_str}"))
                    continue

                # Per-context status with non-zero gRPC code → error
                status = result.get("status") or {}
                if status and status.get("code", 0) != 0:
                    err_msg = status.get("message", f"gRPC code {status.get('code')}")
                    fut = self._context_futures.get(context_id)
                    if fut and not fut.done():
                        fut.set_exception(RuntimeError(f"Inworld error: {err_msg}"))
                    continue

                if "audioChunk" in result:
                    chunk = result["audioChunk"] or {}
                    audio_b64 = chunk.get("audioContent")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        await self._stream_audio_chunk(audio_bytes)
                elif "flushCompleted" in result:
                    fut = self._context_futures.get(context_id)
                    if fut and not fut.done():
                        fut.set_result(None)
                elif "contextCreated" in result or "contextClosed" in result:
                    # Lifecycle events; nothing to do.
                    pass

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Inworld WSS receive loop error: {e}")
            self._fail_all_pending(e)
        finally:
            self._fail_all_pending(RuntimeError("Inworld WSS connection closed"))

    def _fail_all_pending(self, exc: BaseException) -> None:
        for fut in list(self._context_futures.values()):
            if not fut.done():
                fut.set_exception(exc)

    async def _close_ws_resources(self) -> None:
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
        if self._ws_connection and not self._ws_connection.closed:
            await self._ws_connection.close()
        if self._ws_session and not self._ws_session.closed:
            await self._ws_session.close()
        self._receive_task = None
        self._ws_connection = None
        self._ws_session = None
        self._context_futures.clear()

    # ── HTTP path ──────────────────────────────────────────────────────────

    async def _http_synthesis(
        self,
        text: AsyncIterator[str] | str,
        voice_id: Optional[str],
    ) -> None:
        """HTTP streaming synthesis. For AsyncIterator inputs, accumulates
        all chunks into a single POST so currency / number patterns span a
        single request — same fix pattern as Sarvam HTTP and OpenAI."""
        if isinstance(text, str):
            if text.strip():
                await self._http_post(text, voice_id)
            return

        parts: list[str] = []
        async for chunk in text:
            if self._interrupted:
                break
            if chunk and chunk.strip():
                parts.append(chunk)

        if parts and not self._interrupted:
            combined = "".join(parts)
            if combined.strip():
                await self._http_post(combined, voice_id)

    async def _http_post(self, text: str, voice_id: Optional[str]) -> None:
        payload = {
            "text": text,
            "voiceId": voice_id or self.voice_id,
            "modelId": self.model_id,
            "audioConfig": {
                "temperature": self.temperature,
                "audioEncoding": "PCM",
                "sampleRateHertz": self._sample_rate,
            },
        }
        if self.speaking_rate is not None:
            payload["audioConfig"]["speakingRate"] = self.speaking_rate

        headers = {
            "Authorization": self._auth_header,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            async with self._http_client.stream(
                "POST",
                INWORLD_TTS_HTTP_ENDPOINT,
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if self._interrupted:
                        break
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "error" in data:
                        err = data["error"]
                        msg = err.get("message", "Unknown error") if isinstance(err, dict) else str(err)
                        self.emit("error", f"InworldAI API error: {msg}")
                        return

                    audio_b64 = (
                        data.get("result", {}).get("audioContent")
                        or data.get("result", {}).get("audioChunk", {}).get("audioContent")
                    )
                    if audio_b64:
                        await self._stream_audio_chunk(base64.b64decode(audio_b64))

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self.emit("error", "InworldAI authentication failed. Please check your API key.")
            elif e.response.status_code == 400:
                try:
                    err = e.response.json().get("error", {}).get("message", "Bad request")
                except Exception:
                    err = "Bad request"
                self.emit("error", f"InworldAI request error: {err}")
            else:
                self.emit("error", f"InworldAI HTTP error: {e.response.status_code}")
            raise

    # ── Audio output ───────────────────────────────────────────────────────

    async def _stream_audio_chunk(self, audio_bytes: bytes) -> None:
        """Push raw PCM bytes straight to the audio track."""
        if not audio_bytes or self._interrupted:
            return

        if not self._first_chunk_sent and self._first_audio_callback:
            self._first_chunk_sent = True
            await self._first_audio_callback()

        if self.audio_track:
            asyncio.create_task(self.audio_track.add_new_bytes(audio_bytes))
            await asyncio.sleep(0.001)
