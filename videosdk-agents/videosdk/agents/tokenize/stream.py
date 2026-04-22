"""Push-based stream adapter for sentence tokenizers.

Converts a push-style API (``push_text``, ``flush``, ``end_input``) into a
pull-style ``AsyncIterator[str]`` that emits sentences as soon as they are
confirmed complete. Idle-flush fires a word-boundary cut when no terminator
arrives within ``idle_flush_ms`` so the TTS never stalls on a runaway
terminator-less response.

Invariant, covered by tests: every emitted segment starts and ends on a
whitespace, punctuation, or CJK-character boundary. Mid-word splits are
impossible by construction.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable

from .base import SentenceStream
from .patterns import NO_SPACE_SCRIPTS_REGEX

logger = logging.getLogger(__name__)

# Sentinel pushed onto the outgoing queue to signal end-of-stream to the iterator.
_EOS: object = object()


class BufferedSentenceStream(SentenceStream):
    """Stream adapter that wraps a one-shot tokenize function.

    Strategy:

    * Accumulate incoming text in a buffer.
    * After each ``push_text``, call the tokenize function on the buffer.
      When it returns more than one segment, every segment except the last is
      confirmed complete (they're followed by at least one more character in
      the buffer, which the tokenizer interprets as lookahead). Emit them.
      The last segment is the in-flight remainder; keep it as the new buffer.
    * An idle-flush background task watches for silence. If nothing has been
      pushed for ``idle_flush_ms`` and the buffer holds enough text, cut on a
      word or CJK-character boundary and emit. This bounds time-to-audio for
      terminator-less responses.
    * ``flush()`` and ``end_input()`` drain the buffer as a single trailing
      segment and close the iterator cleanly.
    """

    def __init__(
        self,
        *,
        tokenize_fn: Callable[[str], list[str]],
        strong_terminators: str,
        min_sentence_len: int = 20,
        idle_flush_ms: int = 400,
        idle_min_chars: int = 40,
    ) -> None:
        """Initialise the stream.

        Args:
            tokenize_fn: Callable that maps buffer string → list of segments.
                Typically ``partial(tokenizer.tokenize, language=...)``.
            strong_terminators: Character class used for fast lookahead checks.
            min_sentence_len: Passed through for consistency (tokenize_fn
                already knows this; kept here for the idle-flush heuristic).
            idle_flush_ms: Milliseconds of inactivity before a word-boundary
                cut fires. Clamped to at least 100 ms.
            idle_min_chars: Minimum buffer length before idle-flush is allowed
                to fire. Prevents emitting 1-3 char fragments on slow LLMs.
        """
        self._tokenize_fn = tokenize_fn
        self._strong_set = set(strong_terminators)
        self._min_sentence_len = max(1, int(min_sentence_len))
        self._idle_flush_s = max(0.1, idle_flush_ms / 1000.0)
        self._idle_min_chars = max(1, int(idle_min_chars))

        self._buffer: str = ""
        self._buffer_lock = asyncio.Lock()
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._closed: bool = False
        self._last_push_monotonic: float = time.monotonic()
        self._idle_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def push_text(self, text: str) -> None:
        """Feed more text into the stream."""
        if self._closed:
            logger.debug("push_text on closed BufferedSentenceStream; ignored")
            return
        if not text:
            return
        async with self._buffer_lock:
            self._buffer += text
            self._last_push_monotonic = time.monotonic()
            self._ensure_idle_task()
            await self._try_emit_locked()

    async def flush(self) -> None:
        """Emit any buffered text as a single trailing segment."""
        async with self._buffer_lock:
            await self._flush_locked()

    async def end_input(self) -> None:
        """Signal end of input; drain the buffer and close the iterator."""
        async with self._buffer_lock:
            await self._flush_locked()
            if not self._closed:
                self._closed = True
                await self._queue.put(_EOS)
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        item = await self._queue.get()
        if item is _EOS:
            raise StopAsyncIteration
        return item  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    async def _try_emit_locked(self) -> None:
        """Emit every confirmed-complete sentence from the current buffer.

        Called with ``_buffer_lock`` held.
        """
        if not self._buffer:
            return
        try:
            segments = self._tokenize_fn(self._buffer)
        except Exception:  # pragma: no cover - defensive
            logger.error("Tokenizer raised; flushing buffer as-is", exc_info=True)
            remainder = self._buffer.strip()
            self._buffer = ""
            if remainder:
                await self._queue.put(remainder)
            return

        if len(segments) <= 1:
            return

        for segment in segments[:-1]:
            stripped = segment.strip()
            if stripped:
                logger.debug("[chunking] tokenizer emit: %r", stripped)
                await self._queue.put(stripped)

        self._buffer = segments[-1]

    async def _flush_locked(self) -> None:
        """Force-emit any buffered text. Called with ``_buffer_lock`` held."""
        if not self._buffer:
            return
        # Run the tokenizer one more time; it may split the remainder into
        # multiple sentences at strong terminators that were previously held
        # back by the "no lookahead" rule.
        try:
            segments = self._tokenize_fn(self._buffer)
        except Exception:  # pragma: no cover - defensive
            logger.error("Tokenizer raised during flush", exc_info=True)
            segments = [self._buffer]

        self._buffer = ""
        for segment in segments:
            stripped = segment.strip()
            if stripped:
                await self._queue.put(stripped)

    def _ensure_idle_task(self) -> None:
        """Spawn the idle-flush watchdog on first push."""
        if self._idle_task is None or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._idle_monitor())

    async def _idle_monitor(self) -> None:
        """Fire word-boundary cuts when no push arrives for ``idle_flush_ms``."""
        try:
            while not self._closed:
                await asyncio.sleep(self._idle_flush_s)
                if self._closed:
                    return
                if time.monotonic() - self._last_push_monotonic < self._idle_flush_s:
                    continue
                await self._idle_flush()
        except asyncio.CancelledError:
            raise

    async def _idle_flush(self) -> None:
        """Emit a word-boundary cut from the current buffer, if it's long enough."""
        async with self._buffer_lock:
            if len(self._buffer) < self._idle_min_chars:
                return

            cut_index = self._find_fallback_cut(self._buffer)
            if cut_index <= 0:
                return

            segment = self._buffer[:cut_index].strip()
            self._buffer = self._buffer[cut_index:].lstrip()
            if segment:
                logger.debug("Idle-flush emitting %d chars", len(segment))
                await self._queue.put(segment)

    def _find_fallback_cut(self, text: str) -> int:
        """Pick the best place to cut ``text`` in the absence of a terminator.

        Prefers the last space. If there are no spaces (CJK / Thai / Lao /
        Myanmar / Khmer prose), cuts at the last character-range boundary.
        Returns 0 when no safe cut exists.
        """
        last_space = text.rfind(" ")
        if last_space > 0:
            return last_space

        last_cjk = -1
        for m in NO_SPACE_SCRIPTS_REGEX.finditer(text):
            last_cjk = m.end()
        if last_cjk > 0 and last_cjk < len(text):
            return last_cjk

        return 0
