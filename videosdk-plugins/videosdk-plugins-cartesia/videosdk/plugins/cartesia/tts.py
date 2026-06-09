from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional, Union

import aiohttp

from videosdk.agents import TTS, FlushMarker

logger = logging.getLogger(__name__)

CARTESIA_SAMPLE_RATE = 24000
CARTESIA_CHANNELS = 1
DEFAULT_MODEL = "sonic-3.5"
DEFAULT_VOICE_ID = "f8f5f1b2-f02d-4d8e-a40d-fd850a487b3d"
API_VERSION = "2026-03-01"
DEFAULT_CONNECTION_MAX_AGE_SEC = 300.0
SONIC_3_CONFIG_MIN_API_VERSION = "2024-12-15"
CARTESIA_REQUEST_ID_HEADER = "x-cartesia-request-id"


@dataclass
class GenerationConfig:
    """Voice generation parameters. Only sent to Cartesia for sonic-3+ / sonic-latest
    models on a sufficiently recent API version, and only for fields explicitly set.

    Fields (see Cartesia "Volume, Speed, and Emotion" docs):
        speed: 0.6–1.5 (1.0 = default speed).
        volume: 0.5–2.0 (1.0 = default volume).
        emotion: one of "neutral", "calm", "angry", "content", "sad".
    """
    speed: float | None = None
    emotion: str | None = None
    volume: float | None = None


def _supports_generation_config(model: str, api_version: str) -> bool:
    supported_model = "sonic-3" in model or model.startswith("sonic-latest")
    return supported_model and api_version >= SONIC_3_CONFIG_MIN_API_VERSION


def _build_generation_config_payload(cfg: GenerationConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if cfg.speed is not None:
        out["speed"] = cfg.speed
    if cfg.emotion is not None:
        out["emotion"] = cfg.emotion
    if cfg.volume is not None:
        out["volume"] = cfg.volume
    return out


class CartesiaTTS(TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        voice_id: Union[str, List[float]] = DEFAULT_VOICE_ID,
        language: str = "en",
        base_url: str = "https://api.cartesia.ai",
        sample_rate: int = CARTESIA_SAMPLE_RATE,
        generation_config: GenerationConfig | None = None,
        pronunciation_dict_id: str | None = None,
        max_buffer_delay_ms: int | None = None,
        word_timestamps: bool = False,
        max_connection_age_sec: float = DEFAULT_CONNECTION_MAX_AGE_SEC,
    ) -> None:
        """Initialize the Cartesia TTS plugin.

        Args:
            api_key: Cartesia API key. Falls back to CARTESIA_API_KEY env var.
            model: Cartesia model id. Defaults to "sonic-3.5". Use a dated snapshot
                (e.g. "sonic-3.5-2026-05-04") to pin a version, or "sonic-latest" for
                the newest beta (not recommended for production).
            voice_id: Either a Cartesia voice id (str) or a voice embedding (list of floats).
            language: BCP-47 language tag (Sonic 3.5 supports 42 languages, e.g. "en",
                "hi", "es", "ja", "he"; use the same value for every chunk in a context).
            base_url: Cartesia base URL.
            generation_config: Voice generation params (sonic-3+ / sonic-latest only).
                Only fields you set are forwarded; defaults are not sent so they don't
                override Cartesia's model defaults on older models.
            pronunciation_dict_id: Custom pronunciation dictionary id.
            max_buffer_delay_ms: Deprecated. Sentence-paced flushing now drives buffer
                behavior; this value is no longer forwarded.
            word_timestamps: If True, request per-word timestamps for transcript sync.
            max_connection_age_sec: Refresh the WebSocket after this many seconds to
                avoid hitting Cartesia's idle/session limits.
        """
        super().__init__(
            sample_rate=sample_rate,
            num_channels=CARTESIA_CHANNELS,
            word_timestamps=word_timestamps,
        )

        self.model = model
        self.language = language
        self.base_url = base_url
        self._voice = voice_id
        self._first_chunk_sent = False
        self._interrupted = False
        self._generation_config = generation_config or GenerationConfig()
        self.pronunciation_dictionary_id = pronunciation_dict_id
        self.max_buffer_delay_ms = max_buffer_delay_ms
        self._max_connection_age_sec = max_connection_age_sec

        api_key = api_key or os.getenv("CARTESIA_API_KEY")
        if not api_key:
            raise ValueError(
                "Cartesia API key must be provided either through api_key parameter or CARTESIA_API_KEY environment variable"
            )
        self._api_key = api_key

        self._ws_session: aiohttp.ClientSession | None = None
        self._ws_connection: aiohttp.ClientWebSocketResponse | None = None
        self._ws_connect_time: float = 0.0
        self._ws_request_id: str | None = None
        self._connection_lock = asyncio.Lock()
        self._receive_task: asyncio.Task | None = None
        self._context_futures: dict[str, asyncio.Future[None]] = {}
        self._current_context_id: str | None = None

        self._audio_start_time: Optional[float] = None
        self._spoken_words: List[str] = []
        self._word_schedule_tasks: List[asyncio.Task] = []
        self._last_scheduled_start_sec: float = -1.0

    def reset_first_audio_tracking(self) -> None:
        self._first_chunk_sent = False
        self._audio_start_time = None
        self._spoken_words = []
        for task in self._word_schedule_tasks:
            if not task.done():
                task.cancel()
        self._word_schedule_tasks = []
        self._last_scheduled_start_sec = -1.0

    async def prewarm(self) -> None:
        """Pre-establish the Cartesia WebSocket so the first ``synthesize()`` call
        does not pay the TLS + auth + upgrade cost. Safe to call repeatedly."""
        try:
            await self._ensure_ws_connection()
        except Exception as e:
            logger.warning(f"Cartesia TTS prewarm failed: {e}")

    async def synthesize(
        self,
        text: AsyncIterator[Union[str, FlushMarker]] | str,
        voice_id: Optional[Union[str, List[float]]] = None,
        **kwargs: Any,
    ) -> None:
        """Synthesize text to speech using Cartesia's streaming WebSocket API."""
        context_id = ""
        try:
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or event loop not set")
                return

            if voice_id:
                self._voice = voice_id

            self._interrupted = False

            await self._ensure_ws_connection()
            if not self._ws_connection:
                raise RuntimeError("WebSocket connection is not available.")

            context_id = os.urandom(8).hex()
            self._current_context_id = context_id
            done_future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
            self._context_futures[context_id] = done_future

            async def _string_iterator(s: str) -> AsyncIterator[str]:
                yield s

            text_iterator = _string_iterator(text) if isinstance(text, str) else text
            send_task = asyncio.create_task(self._send_task(text_iterator, context_id))

            try:
                await done_future
            except asyncio.CancelledError:
                pass
            await send_task

        except Exception as e:
            logger.error(f"TTS synthesis failed (context_id={context_id}): {e}")
            self.emit("error", f"TTS synthesis failed: {str(e)}")
            raise
        finally:
            if context_id and context_id in self._context_futures:
                del self._context_futures[context_id]
            if self._current_context_id == context_id:
                self._current_context_id = None

    def _build_voice_payload(self) -> dict[str, Any]:
        if isinstance(self._voice, str):
            return {"mode": "id", "id": self._voice}
        return {"mode": "embedding", "embedding": self._voice}

    def _build_base_payload(self, context_id: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_id": self.model,
            "language": self.language,
            "voice": self._build_voice_payload(),
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
            "add_timestamps": self.supports_word_timestamps,
            "context_id": context_id,
        }
        if _supports_generation_config(self.model, API_VERSION):
            gen_cfg = _build_generation_config_payload(self._generation_config)
            if gen_cfg:
                payload["generation_config"] = gen_cfg
        if self.pronunciation_dictionary_id:
            payload["pronunciation_dict_id"] = self.pronunciation_dictionary_id
        return payload

    async def _send_text_packet(self, base_payload: dict[str, Any], text: str) -> bool:
        if not text or not text.strip():
            return False
        if not self._ws_connection or self._ws_connection.closed:
            return False
        payload = {
            **base_payload,
            "transcript": text + " ",
            "continue": True,
            "max_buffer_delay_ms": 0,
        }
        await self._ws_connection.send_str(json.dumps(payload))
        return True

    async def _send_task(self, text_iterator: AsyncIterator[Union[str, FlushMarker]], context_id: str) -> None:
        """Forward sentence-bounded chunks to Cartesia.

        Sentence segmentation is handled upstream by the framework
        (BasicSentenceTokenizer in pipeline_orchestrator) — each chunk
        arriving here is already one sentence, for both LLM streams and
        ``session.say()`` static text. Cartesia's server progressively
        streams audio with ``max_buffer_delay_ms: 0`` so we just forward
        each sentence verbatim. ``FlushMarker`` is a no-op since there
        is no client-side buffer to drain.
        """
        has_sent_transcript = False
        base_payload = self._build_base_payload(context_id)

        try:
            async for chunk in text_iterator:
                if self._interrupted:
                    break
                if isinstance(chunk, FlushMarker):
                    continue
                if not chunk:
                    continue
                if await self._send_text_packet(base_payload, chunk):
                    has_sent_transcript = True

        except Exception as e:
            logger.error(f"Error in send_task (context_id={context_id}): {e}")
            future = self._context_futures.get(context_id)
            if future and not future.done():
                future.set_exception(e)
        finally:
            if has_sent_transcript and not self._interrupted:
                final_payload = {**base_payload, "transcript": " ", "continue": False}
                if self._ws_connection and not self._ws_connection.closed:
                    try:
                        await self._ws_connection.send_str(json.dumps(final_payload))
                    except Exception as e:
                        logger.debug(
                            f"Error sending final continue:false (context_id={context_id}): {e}"
                        )

    async def _receive_loop(self) -> None:
        try:
            while self._ws_connection and not self._ws_connection.closed:
                msg = await self._ws_connection.receive()
                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(msg.data)
                context_id = data.get("context_id")
                future = self._context_futures.get(context_id)

                # Drop frames for cancelled / unknown contexts. A future that's
                # been cancelled (via interrupt()) has done()==True, so its
                # trailing audio bytes from Cartesia are silently dropped here
                # rather than bleeding into the next context's playback.
                if not future or future.done():
                    continue

                if data.get("type") == "error":
                    logger.error(
                        f"Cartesia API error (context_id={context_id}, request_id={self._ws_request_id}): {data}"
                    )
                    future.set_exception(RuntimeError(f"Cartesia API error: {json.dumps(data)}"))
                elif data.get("type") == "timestamps" and "word_timestamps" in data:
                    wt = data["word_timestamps"] or {}
                    words = wt.get("words", []) or []
                    starts = wt.get("start", []) or []
                    for word, start_sec in zip(words, starts):
                        start_sec_f = float(start_sec)
                        if start_sec_f <= self._last_scheduled_start_sec:
                            continue
                        self._last_scheduled_start_sec = start_sec_f
                        task = asyncio.create_task(
                            self._schedule_word_emit(word, start_sec_f)
                        )
                        self._word_schedule_tasks.append(task)
                elif "data" in data and data["data"]:
                    await self._stream_audio(base64.b64decode(data["data"]))
                elif data.get("done"):
                    future.set_result(None)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Cartesia receive loop error (request_id={self._ws_request_id}): {e}")
            for fut in self._context_futures.values():
                if not fut.done():
                    fut.set_exception(e)
        finally:
            for fut in self._context_futures.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("Cartesia WebSocket closed"))
            self._context_futures.clear()

    async def _ensure_ws_connection(self) -> None:
        async with self._connection_lock:
            now = asyncio.get_event_loop().time()

            if self._ws_connection and not self._ws_connection.closed:
                age = now - self._ws_connect_time
                if age < self._max_connection_age_sec:
                    return
                logger.info(f"Refreshing Cartesia WebSocket (age={age:.1f}s)")
                await self._close_connection_locked()
            elif self._ws_connection or self._ws_session:
                await self._close_connection_locked()

            try:
                self._ws_session = aiohttp.ClientSession()
                ws_url = self.base_url.replace("http", "ws", 1)
                full_ws_url = f"{ws_url}/tts/websocket?cartesia_version={API_VERSION}"

                self._ws_connection = await asyncio.wait_for(
                    self._ws_session.ws_connect(
                        full_ws_url,
                        headers={"X-API-Key": self._api_key},
                        heartbeat=30.0,
                    ),
                    timeout=5.0,
                )
                self._ws_connect_time = now

                try:
                    resp = getattr(self._ws_connection, "_response", None)
                    if resp is not None:
                        self._ws_request_id = resp.headers.get(CARTESIA_REQUEST_ID_HEADER)
                except Exception:
                    self._ws_request_id = None

                logger.debug(
                    f"Cartesia WebSocket established (request_id={self._ws_request_id})"
                )
                self._receive_task = asyncio.create_task(self._receive_loop())
            except Exception as e:
                logger.error(f"Failed to establish Cartesia WebSocket: {e}")
                self.emit("error", f"Failed to establish WebSocket connection: {e}")
                if self._ws_session and not self._ws_session.closed:
                    try:
                        await self._ws_session.close()
                    except Exception:
                        pass
                self._ws_session = None
                self._ws_connection = None
                raise

    async def _close_connection_locked(self) -> None:
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            except Exception:
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
        self._ws_request_id = None

    async def _stream_audio(self, audio_chunk: bytes) -> None:
        if self._interrupted or not audio_chunk:
            return

        if not self._first_chunk_sent:
            self._first_chunk_sent = True
            self._audio_start_time = asyncio.get_event_loop().time()
            if self._first_audio_callback:
                await self._first_audio_callback()

        if self.audio_track:
            await self.audio_track.add_new_bytes(audio_chunk)

    async def _schedule_word_emit(self, word: str, start_sec: float) -> None:
        while self._audio_start_time is None and not self._interrupted:
            await asyncio.sleep(0.01)
        if self._interrupted or self._audio_start_time is None:
            return

        target_time = self._audio_start_time + float(start_sec)
        now = asyncio.get_event_loop().time()
        delay = target_time - now
        if delay > 0:
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return
        if self._interrupted:
            return

        self._spoken_words.append(word)
        cumulative = " ".join(self._spoken_words)
        try:
            self.emit("word_spoken", {"word": word, "cumulative_text": cumulative})
        except Exception:
            pass

    async def interrupt(self) -> None:
        """Stop emitting audio for the current synthesis. Keeps the WebSocket
        open so the next turn does not pay reconnect cost; audio chunks for the
        cancelled context are dropped client-side via context_id filtering."""
        self._interrupted = True

        for task in self._word_schedule_tasks:
            if not task.done():
                task.cancel()
        self._word_schedule_tasks = []

        if self.audio_track:
            self.audio_track.interrupt()

        for fut in list(self._context_futures.values()):
            if not fut.done():
                fut.cancel()

    async def aclose(self) -> None:
        await super().aclose()
        self._interrupted = True
        async with self._connection_lock:
            await self._close_connection_locked()
