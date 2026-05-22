from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import AsyncIterator, Iterable, TYPE_CHECKING, Union

import av
from av.audio.resampler import AudioResampler

if TYPE_CHECKING:
    from .tts.tts import TTS

logger = logging.getLogger(__name__)


DEFAULT_SAMPLE_RATE = 24000
DEFAULT_NUM_CHANNELS = 1


def load_audio_file(
    path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_channels: int = DEFAULT_NUM_CHANNELS,
) -> bytes:
    """Load an audio file as raw PCM bytes ready for ``session.say(audio_data=...)``.

    Decodes via libav (PyAV), so any libav-decodable format works: WAV, MP3,
    Ogg/Vorbis, Ogg/Opus, FLAC, M4A/AAC — and any other codec ffmpeg supports.
    Resamples and downmixes/upmixes as needed so the output matches the agent
    audio track format (default 24 kHz, mono, 16-bit signed PCM).

    Args:
        path: Path to an audio file.
        sample_rate: Target sample rate in Hz. Must match the agent audio
            track. Default 24000.
        num_channels: Target channel count (1 = mono, 2 = stereo). Default 1.

    Returns:
        Raw little-endian 16-bit signed PCM bytes.
    """
    if num_channels not in (1, 2):
        raise ValueError(f"num_channels must be 1 or 2, got {num_channels}")

    layout = "mono" if num_channels == 1 else "stereo"
    container = av.open(path)
    try:
        stream = next(
            (s for s in container.streams if s.type == "audio"), None
        )
        if stream is None:
            raise ValueError(f"No audio stream found in {path}")

        resampler = AudioResampler(format="s16", layout=layout, rate=sample_rate)
        chunks: list[bytes] = []
        for src_frame in container.decode(stream):
            for out_frame in resampler.resample(src_frame):
                chunks.append(out_frame.to_ndarray().tobytes())

        for out_frame in resampler.resample(None):
            chunks.append(out_frame.to_ndarray().tobytes())
    finally:
        container.close()

    return b"".join(chunks)


async def _iter_audio_bytes(
    source: Union[bytes, bytearray, Iterable[bytes], AsyncIterator[bytes]],
) -> AsyncIterator[bytes]:
    """Normalize a bytes/iterable/async-iterable audio source into an async chunk stream."""
    if isinstance(source, (bytes, bytearray)):
        if source:
            yield bytes(source)
        return
    if hasattr(source, "__aiter__"):
        async for chunk in source:
            yield chunk
        return
    if hasattr(source, "__iter__"):
        for chunk in source:
            yield chunk
        return
    raise TypeError(
        "audio_data must be bytes, Iterable[bytes], or AsyncIterator[bytes]; "
        f"got {type(source).__name__}"
    )


class TTSAudioCache:
    """Reuse-by-key cache for TTS-synthesized audio.

    On the first call to :meth:`fetch` for a given phrase, the cache invokes
    the wrapped TTS instance, collects the PCM bytes, and stores them. All
    subsequent calls for the same phrase return the stored bytes without
    touching the TTS provider.

    Eviction is LRU — the least-recently-fetched entry is dropped when the
    cache exceeds ``max_entries``.

    Example::

        cache = TTSAudioCache(tts)
        await cache.preload(["Let me check that for you.", "One moment."])
        await session.say("One moment.", audio_data=await cache.fetch("One moment."))
    """

    def __init__(self, tts: "TTS", max_entries: int = 128) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._tts = tts
        self._max_entries = max_entries
        self._store: "OrderedDict[str, bytes]" = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, text: str) -> bool:
        return self._key(text, None) in self._store

    def clear(self) -> None:
        """Drop every entry from the cache."""
        self._store.clear()
        self._locks.clear()

    async def fetch(self, text: str, *, voice_id: str | None = None) -> bytes:
        """Return cached PCM bytes for ``text``, synthesizing on first call.

        Concurrent fetches for the same key share a single synthesis — the
        second caller awaits the first instead of triggering a duplicate
        TTS request.
        """
        key = self._key(text, voice_id)
        cached = self._store.get(key)
        if cached is not None:
            self._store.move_to_end(key)
            return cached

        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._store.get(key)
            if cached is not None:
                self._store.move_to_end(key)
                return cached

            audio = await self._synthesize(text, voice_id)
            self._store[key] = audio
            self._store.move_to_end(key)
            self._evict_overflow()
            return audio

    async def preload(
        self, texts: list[str], *, voice_id: str | None = None
    ) -> None:
        """Synthesize and cache a batch of phrases up front (e.g. at startup)."""
        for text in texts:
            await self.fetch(text, voice_id=voice_id)

    async def _synthesize(self, text: str, voice_id: str | None) -> bytes:
        async def _text_iter() -> AsyncIterator[str]:
            yield text

        kwargs: dict = {}
        if voice_id is not None:
            kwargs["voice_id"] = voice_id

        chunks: list[bytes] = []
        async for chunk in self._tts.stream_synthesize(_text_iter(), **kwargs):
            if chunk:
                chunks.append(chunk)
        return b"".join(chunks)

    def _evict_overflow(self) -> None:
        while len(self._store) > self._max_entries:
            evicted_key, _ = self._store.popitem(last=False)
            self._locks.pop(evicted_key, None)

    @staticmethod
    def _key(text: str, voice_id: str | None) -> str:
        payload = f"{voice_id or ''}::{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
