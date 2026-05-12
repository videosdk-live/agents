from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, Optional, Union

import aiohttp

from videosdk.agents import TTS, FlushMarker, segment_text


@dataclass
class _WSRequestState:
    """Per-request state for Smallest AI WS synthesis.
    """

    future: asyncio.Future
    audio_event: asyncio.Event

logger = logging.getLogger(__name__)

SMALLESTAI_SAMPLE_RATE = 24000
SMALLESTAI_CHANNELS = 1

SmallestAIModel = Literal["lightning", "lightning-large", "lightning-v2", "lightning-v3.1"]
SUPPORTED_MODELS: tuple[SmallestAIModel, ...] = (
    "lightning",
    "lightning-large",
    "lightning-v2",
    "lightning-v3.1",
)
DEFAULT_MODEL: SmallestAIModel = "lightning-v3.1"
DEFAULT_VOICE_ID = "magnus"
DEFAULT_LANGUAGE: Literal["en", "hi"] = "en"
DEFAULT_BASE_URL = "https://api.smallest.ai/waves/v1"
DEFAULT_OUTPUT_FORMAT = "pcm"
SUPPORTED_SAMPLE_RATES = (8000, 16000, 24000, 44100)
HTTP_TIMEOUT_S = 30.0
LEGACY_PROSODY_MODELS = frozenset({"lightning-v2"})
WS_SUPPORTED_MODELS: frozenset[SmallestAIModel] = frozenset({"lightning-v3.1"})
DEFAULT_CONNECTION_MAX_AGE_SEC = 300.0
DEFAULT_COMPLETE_BACKOFF_MS = 0
DEFAULT_MAX_BUFFER_FLUSH_MS = 0
WS_FIRST_AUDIO_TIMEOUT_S = 10.0
WS_AUDIO_SILENCE_THRESHOLD_S = 0.25

class SmallestAITTS(TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: SmallestAIModel = DEFAULT_MODEL,
        voice_id: str = DEFAULT_VOICE_ID,
        language: Literal["en", "hi"] = DEFAULT_LANGUAGE,
        sample_rate: int = SMALLESTAI_SAMPLE_RATE,
        speed: float = 1.0,
        consistency: float = 0.5,
        similarity: float = 0.0,
        enhancement: float = 1.0,
        sentence_streaming: bool = True,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = HTTP_TIMEOUT_S,
        enable_streaming: bool = True,
        max_connection_age_sec: float = DEFAULT_CONNECTION_MAX_AGE_SEC,
        complete_backoff_ms: int = DEFAULT_COMPLETE_BACKOFF_MS,
        max_buffer_flush_ms: int = DEFAULT_MAX_BUFFER_FLUSH_MS,
    ) -> None:
        """Initialize the SmallestAI TTS plugin.

        Args:
            api_key: SmallestAI API key. Falls back to ``SMALLEST_API_KEY`` env var.
            model: Model identifier. Defaults to ``lightning-v3.1`` (latest;
                80+ voices, ~100ms TTFB). Older accepted values:
                ``lightning``, ``lightning-large``, ``lightning-v2``.
            voice_id: Voice id. Defaults to ``magnus`` — a Lightning v3.1
                voice from Smallest AI's official streaming example. The v2
                voice catalog (e.g. ``emily``) is rejected by v3.1 with
                ``Invalid input data``; pass an explicit v2 voice id together
                with ``model="lightning-v2"`` if you need that catalog. For
                Hindi, pair a Hindi-capable v3.1 voice with ``language="hi"``.
            language: ``en`` or ``hi``.
            sample_rate: Output sample rate. One of 8000, 16000, 24000, 44100.
            speed: Speaking speed multiplier.
            consistency: Lightning v2 only — voice-clone consistency.
            similarity: Lightning v2 only — voice-clone similarity.
            enhancement: Lightning v2 only — output enhancement level.
            sentence_streaming: When True, upstream tokens are batched into
                sentence-sized requests via ``segment_text`` so audio for
                sentence N starts playing while sentence N+1 is still being
                produced upstream.
            base_url: HTTPS base URL for the Smallest AI Waves API.
            timeout: Per-request HTTP timeout in seconds.
            enable_streaming: ``True`` (default) opens a persistent WebSocket
                to the streaming endpoint; ``False`` falls back to per-segment
                HTTP POSTs. Streaming is only valid for ``lightning-v3.1`` —
                older models lack the streaming endpoint.
            max_connection_age_sec: WebSocket only. Refresh after this many
                seconds to avoid Smallest AI's idle/session limits.
            complete_backoff_ms: WebSocket only. Server-side delay after the
                last audio chunk before ``status: complete`` is emitted.
                Default ``0``. Note: Smallest AI appears to clamp this
                server-side to ~4000ms regardless of the client value, which
                is why the synthesize loop uses a 250ms silence-watchdog
                instead of awaiting ``complete``.
            max_buffer_flush_ms: WebSocket only. Server-side flush trigger —
                max wait for more input before generating output. Default
                ``0`` (flush immediately when enough text is buffered).
        """
        if model not in SUPPORTED_MODELS:
            raise ValueError(
                f"model must be one of {SUPPORTED_MODELS}, got {model!r}"
            )
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {SUPPORTED_SAMPLE_RATES}, got {sample_rate}"
            )
        if enable_streaming and model not in WS_SUPPORTED_MODELS:
            raise ValueError(
                f"enable_streaming=True is only supported for {sorted(WS_SUPPORTED_MODELS)}, "
                f"got model={model!r}. Pass enable_streaming=False to use the HTTP "
                f"endpoint with older Lightning models."
            )

        api_key = api_key or os.getenv("SMALLEST_API_KEY")
        if not api_key:
            raise ValueError(
                "SmallestAI API key required. Provide it via the api_key "
                "argument or the SMALLEST_API_KEY environment variable."
            )

        super().__init__(sample_rate=sample_rate, num_channels=SMALLESTAI_CHANNELS)

        self.model = model
        self.voice_id = voice_id
        self.language = language
        self.speed = speed
        self.consistency = consistency
        self.similarity = similarity
        self.enhancement = enhancement
        self.sentence_streaming = sentence_streaming
        self.timeout = timeout
        self.enable_streaming = enable_streaming
        self._max_connection_age_sec = max_connection_age_sec
        self.complete_backoff_ms = complete_backoff_ms
        self.max_buffer_flush_ms = max_buffer_flush_ms

        normalized_base = base_url.rstrip("/")
        self._endpoint = f"{normalized_base}/{model}/get_speech"
        ws_base = normalized_base.replace("https://", "wss://", 1).replace(
            "http://", "ws://", 1
        )
        self._ws_endpoint = f"{ws_base}/{model}/get_speech/stream"

        self._auth_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._ws_auth_headers = {"Authorization": f"Bearer {api_key}"}
        self._api_key = api_key

        self.audio_track = None
        self.loop = None
        self._first_chunk_sent = False
        self._interrupted = False
        self._http_session: aiohttp.ClientSession | None = None
        self._ws_session: aiohttp.ClientSession | None = None
        self._ws_connection: aiohttp.ClientWebSocketResponse | None = None
        self._ws_connect_time: float = 0.0
        self._connection_lock = asyncio.Lock()
        self._receive_task: asyncio.Task | None = None
        self._active_requests: dict[str, _WSRequestState] = {}

    def reset_first_audio_tracking(self) -> None:
        """Reset first-audio state so the next synthesize call re-fires the callback."""
        self._first_chunk_sent = False

    async def prewarm(self) -> None:
        """Pre-establish the WebSocket when streaming is enabled so the first
        ``synthesize()`` call doesn't pay the TLS + auth + upgrade cost. For
        the HTTP fallback this is a no-op — the shared
        ``aiohttp.ClientSession`` handles keep-alive automatically."""
        if not self.enable_streaming:
            return
        try:
            await self._ensure_ws_connection()
        except Exception as e:
            logger.warning(f"SmallestAI TTS prewarm failed (non-fatal): {e}")

    async def synthesize(
        self,
        text: AsyncIterator[Union[str, FlushMarker]] | str,
        voice_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Convert text to speech using SmallestAI's HTTP endpoint and stream
        to the audio track.

        """
        try:
            if not self.audio_track or not self.loop:
                logger.error("SmallestAI TTS: audio track or event loop not set")
                self.emit("error", "Audio track or event loop not set")
                return

            self._interrupted = False

            if isinstance(text, str):
                if not self._interrupted:
                    await self._synthesize_segment(text, voice_id, **kwargs)
                return
            
            source = segment_text(text) if self.sentence_streaming else text
            async for segment in source:
                if self._interrupted:
                    break
                if isinstance(segment, FlushMarker):
                    continue
                cleaned = (segment or "").strip()
                if cleaned:
                    await self._synthesize_segment(cleaned, voice_id, **kwargs)
        except Exception as exc:
            logger.exception("SmallestAI TTS synthesis failed")
            self.emit("error", f"SmallestAI TTS synthesis failed: {exc}")
            raise

    async def _synthesize_segment(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Synthesize a single text segment.
        """
        if not text or self._interrupted:
            return

        target_voice = voice_id or self.voice_id

        if self.enable_streaming:
            await self._synthesize_segment_ws(text, target_voice, kwargs)
            return

        body = self._build_speech_params(text, target_voice, kwargs)

        last_exc: Optional[BaseException] = None
        for attempt in range(2):
            try:
                await self._dispatch_segment(body)
                return
            except (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt == 0 and not self._interrupted:
                    await asyncio.sleep(0.25 * (2 ** attempt))
                    continue
                logger.error("SmallestAI network failure: %s", exc)
                self.emit("error", f"SmallestAI network failure: {exc}")
                return

        if last_exc is not None:
            raise last_exc

    async def _dispatch_segment(self, body: dict[str, Any]) -> None:
        """Issue one HTTP POST and stream the chunked PCM response."""
        session = await self._ensure_http_session()
        post_start = time.monotonic()
        first_byte_ms: float | None = None
        bytes_seen = 0

        try:
            async with session.post(
                self._endpoint,
                headers=self._auth_headers,
                json=body,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status >= 400:
                    detail = (await resp.text())[:500]
                    logger.error(
                        "SmallestAI HTTP %d on /%s/get_speech: %s",
                        resp.status,
                        self.model,
                        detail,
                    )
                    raise RuntimeError(
                        f"SmallestAI returned HTTP {resp.status}: {detail}"
                    )
                async for chunk, _ in resp.content.iter_chunks():
                    if self._interrupted:
                        return
                    if not chunk:
                        continue
                    if first_byte_ms is None:
                        first_byte_ms = (time.monotonic() - post_start) * 1000.0
                    bytes_seen += len(chunk)
                    await self._stream_audio_chunks(chunk)
        finally:
            total_ms = (time.monotonic() - post_start) * 1000.0
            if not self._interrupted and bytes_seen == 0:
                logger.warning(
                    "SmallestAI segment returned no audio | total_ms=%.0f",
                    total_ms,
                )
            logger.debug(
                "SmallestAI segment | ttfb_ms=%s total_ms=%.0f audio_bytes=%d",
                f"{first_byte_ms:.0f}" if first_byte_ms is not None else "n/a",
                total_ms,
                bytes_seen,
            )

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        """Forward one PCM chunk to the audio track and fire the first-audio callback."""
        if self._interrupted or not audio_bytes:
            return
        if not self._first_chunk_sent and self._first_audio_callback:
            self._first_chunk_sent = True
            await self._first_audio_callback()
        if self.audio_track:
            await self.audio_track.add_new_bytes(audio_bytes)

    def _build_speech_params(
        self, text: str, voice: str, overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """Compose the JSON body for one ``get_speech`` request.

        The voice-clone prosody knobs (``consistency``, ``similarity``,
        ``enhancement``) are appended only for ``LEGACY_PROSODY_MODELS`` —
        newer Lightning revisions reject the unknown fields.
        """
        params: dict[str, Any] = {
            "voice_id": voice,
            "text": text,
            "language": overrides.get("language", self.language),
            "sample_rate": overrides.get("sample_rate", self.sample_rate),
            "speed": overrides.get("speed", self.speed),
            "output_format": overrides.get("output_format", DEFAULT_OUTPUT_FORMAT),
        }
        if self.model in LEGACY_PROSODY_MODELS:
            params["consistency"] = overrides.get("consistency", self.consistency)
            params["similarity"] = overrides.get("similarity", self.similarity)
            params["enhancement"] = overrides.get("enhancement", self.enhancement)
        return params

    async def _ensure_http_session(self) -> aiohttp.ClientSession:
        """Return a live ``aiohttp.ClientSession``, creating one lazily.

        The session is created on first use and reused for every subsequent
        segment. If a previous session was closed (e.g. after ``aclose``), a
        fresh one is created on the next call.
        """
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
            logger.info(
                "SmallestAI HTTP session opened | endpoint=%s",
                self._endpoint,
            )
        return self._http_session


    async def _synthesize_segment_ws(
        self,
        text: str,
        voice: str,
        overrides: dict[str, Any],
    ) -> None:
        """Send one segment as a one-shot WebSocket request.
        """
        await self._ensure_ws_connection()
        if not self._ws_connection or self._ws_connection.closed:
            raise RuntimeError("SmallestAI WebSocket connection is not available.")

        request_id = uuid.uuid4().hex
        state = _WSRequestState(
            future=asyncio.get_event_loop().create_future(),
            audio_event=asyncio.Event(),
        )
        self._active_requests[request_id] = state

        body: dict[str, Any] = {
            "voice_id": voice,
            "text": text,
            "language": overrides.get("language", self.language),
            "sample_rate": overrides.get("sample_rate", self.sample_rate),
            "speed": overrides.get("speed", self.speed),
            "request_id": request_id,
        }
        complete_backoff = overrides.get(
            "complete_backoff_ms", self.complete_backoff_ms
        )
        if complete_backoff and complete_backoff > 0:
            body["complete_backoff_ms"] = complete_backoff
        max_buffer_flush = overrides.get(
            "max_buffer_flush_ms", self.max_buffer_flush_ms
        )
        if max_buffer_flush and max_buffer_flush > 0:
            body["max_buffer_flush_ms"] = max_buffer_flush

        post_start = time.monotonic()
        try:
            await self._ws_connection.send_str(json.dumps(body))

            try:
                await asyncio.wait_for(
                    state.audio_event.wait(),
                    timeout=WS_FIRST_AUDIO_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "SmallestAI WS: no audio in %.1fs for request %s",
                    WS_FIRST_AUDIO_TIMEOUT_S,
                    request_id,
                )
                return

            while not state.future.done() and not self._interrupted:
                state.audio_event.clear()
                try:
                    await asyncio.wait_for(
                        state.audio_event.wait(),
                        timeout=WS_AUDIO_SILENCE_THRESHOLD_S,
                    )
                except asyncio.TimeoutError:
                    break
        except Exception as exc:
            if not self._interrupted:
                logger.error("SmallestAI WS segment failed: %s", exc)
                self.emit("error", f"SmallestAI WS segment failed: {exc}")
            raise
        finally:
            self._active_requests.pop(request_id, None)
            logger.debug(
                "SmallestAI WS segment | total_ms=%.0f request_id=%s",
                (time.monotonic() - post_start) * 1000.0,
                request_id,
            )

    async def _ensure_ws_connection(self) -> None:
        """Open or refresh the WebSocket. Acquires ``_connection_lock`` so
        concurrent ``synthesize`` and ``prewarm`` calls don't race the open."""
        async with self._connection_lock:
            now = asyncio.get_event_loop().time()

            if self._ws_connection and not self._ws_connection.closed:
                age = now - self._ws_connect_time
                if age < self._max_connection_age_sec:
                    return
                logger.info(f"Refreshing SmallestAI WebSocket (age={age:.1f}s)")
                await self._close_ws_connection_locked()
            elif self._ws_connection or self._ws_session:
                await self._close_ws_connection_locked()

            try:
                self._ws_session = aiohttp.ClientSession()
                self._ws_connection = await asyncio.wait_for(
                    self._ws_session.ws_connect(
                        self._ws_endpoint,
                        headers=self._ws_auth_headers,
                        heartbeat=30.0,
                    ),
                    timeout=10.0,
                )
                self._ws_connect_time = now
                self._receive_task = asyncio.create_task(self._recv_loop())
                logger.info(
                    "SmallestAI WebSocket opened | endpoint=%s",
                    self._ws_endpoint,
                )
            except Exception as exc:
                logger.error(f"Failed to open SmallestAI WebSocket: {exc}")
                self.emit("error", f"SmallestAI WebSocket connect failed: {exc}")
                if self._ws_session and not self._ws_session.closed:
                    try:
                        await self._ws_session.close()
                    except Exception:
                        pass
                self._ws_session = None
                self._ws_connection = None
                raise

    async def _close_ws_connection_locked(self) -> None:
        """Tear down the WS + receive task. Caller must hold ``_connection_lock``."""
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass
        self._receive_task = None
        if self._ws_connection and not self._ws_connection.closed:
            try:
                await self._ws_connection.close()
            except Exception:
                pass
        self._ws_connection = None
        if self._ws_session and not self._ws_session.closed:
            try:
                await self._ws_session.close()
            except Exception:
                pass
        self._ws_session = None

    async def _recv_loop(self) -> None:
        """Long-running task: parse incoming server messages, route audio
        chunks, resolve done futures on ``status: complete``."""
        try:
            while self._ws_connection and not self._ws_connection.closed:
                msg = await self._ws_connection.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break
                if msg.type == aiohttp.WSMsgType.ERROR:
                    raise ConnectionError(
                        f"SmallestAI WebSocket error: {self._ws_connection.exception()}"
                    )
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.debug("SmallestAI WS: non-JSON message dropped: %r", msg.data[:200])
                    continue

                error_payload = data.get("error") or (
                    data.get("data", {}).get("error")
                    if isinstance(data.get("data"), dict)
                    else None
                )
                if error_payload or data.get("status") == "error":
                    err_msg = error_payload or data.get("message") or json.dumps(data)[:300]
                    logger.error("SmallestAI WS error frame: %s", err_msg)
                    self.emit("error", f"SmallestAI WS error: {err_msg}")
                    req_id = data.get("external_request_id")
                    state = self._active_requests.get(req_id) if req_id else None
                    if state is not None and not state.future.done():
                        state.future.set_exception(RuntimeError(str(err_msg)))
                        state.audio_event.set()
                    elif state is None:
                        for s in list(self._active_requests.values()):
                            if not s.future.done():
                                s.future.set_exception(RuntimeError(str(err_msg)))
                            s.audio_event.set()
                    continue

                req_id = data.get("external_request_id")
                state = self._active_requests.get(req_id) if req_id else None
                if state is None:
                    logger.debug(
                        "SmallestAI WS: dropping frame with no matching request_id "
                        "(external_request_id=%r, status=%r, keys=%r)",
                        req_id, data.get("status"), list(data.keys()),
                    )
                    continue
                if state.future.done():
                    continue

                status = data.get("status")
                if status == "chunk":
                    payload = data.get("data") or {}
                    audio_b64 = payload.get("audio")
                    if audio_b64:
                        try:
                            await self._stream_audio_chunks(base64.b64decode(audio_b64))
                            state.audio_event.set()
                        except Exception as exc:
                            logger.error(f"SmallestAI WS audio decode failed: {exc}")
                elif status == "complete":
                    if not state.future.done():
                        state.future.set_result(None)
                else:
                    logger.debug(
                        "SmallestAI WS: unrecognised frame status=%r keys=%r",
                        status, list(data.keys()),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._interrupted:
                logger.error(f"SmallestAI WS receive loop error: {exc}")
            for state in list(self._active_requests.values()):
                if not state.future.done():
                    state.future.set_exception(exc)
                state.audio_event.set()
        finally:
            for state in list(self._active_requests.values()):
                if not state.future.done():
                    state.future.set_exception(
                        RuntimeError("SmallestAI WebSocket closed")
                    )
                state.audio_event.set()

    async def interrupt(self) -> None:
        """Stop the in-flight synthesis as soon as possible."""
        self._interrupted = True
        if self.audio_track:
            self.audio_track.interrupt()

        for state in list(self._active_requests.values()):
            if not state.future.done():
                state.future.cancel()

            state.audio_event.set()

    async def aclose(self) -> None:
        """Release HTTP and WebSocket resources, then chain to ``TTS.aclose``."""
        self._interrupted = True
        if self._http_session is not None and not self._http_session.closed:
            await self._http_session.close()
            logger.info("SmallestAI HTTP session closed")
        async with self._connection_lock:
            await self._close_ws_connection_locked()
        await super().aclose()
