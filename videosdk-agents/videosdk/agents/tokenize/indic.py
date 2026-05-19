from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from .base import SentenceChunkStream, SentenceChunker
from .stream import BufferedSentenceChunkStream

if TYPE_CHECKING:
    from indicnlp.tokenize.sentence_tokenize import sentence_split as _SentenceSplit

logger = logging.getLogger(__name__)

# Language codes supported by indic-nlp-library's sentence_split / normalisers.
# ISO 639-1 (or 639-3 for codes without a 1-letter form).
INDIC_LANGS: frozenset[str] = frozenset({
    "hi",   # Hindi
    "bn",   # Bengali
    "ta",   # Tamil
    "te",   # Telugu
    "kn",   # Kannada
    "ml",   # Malayalam
    "mr",   # Marathi
    "gu",   # Gujarati
    "pa",   # Punjabi (Gurmukhi)
    "or",   # Odia
    "as",   # Assamese
    "si",   # Sinhala
    "ur",   # Urdu
    "sa",   # Sanskrit
    "ne",   # Nepali
    "kok",  # Konkani
    "mai",  # Maithili
    "bgc",  # Haryanvi
})

# Back-compat alias for the name the plugin used to export.
SUPPORTED_LANGUAGES: frozenset[str] = INDIC_LANGS


def _load_sentence_split() -> "_SentenceSplit":
    """Lazy import of the upstream sentence splitter.

    ``indic-nlp-library`` is a direct dependency of ``videosdk-agents``, so this
    should never fail; the guard is defensive (e.g. a broken install).
    """
    try:
        from indicnlp.tokenize.sentence_tokenize import sentence_split

        return sentence_split
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "indic-nlp-library is missing — it ships as a dependency of "
            "videosdk-agents, so reinstall the package: uv pip install -U videosdk-agents"
        ) from exc


def pre_warm_tokenizer() -> None:
    """Eagerly import ``indic-nlp-library`` so the first ``.tokenize()`` call is cheap.

    The underlying ``indicnlp.tokenize.sentence_tokenize`` module performs its
    expensive initialisation on first import (~6s on a cold Python process).
    Calling this at worker start — alongside ``TurnDetector.pre_download_model()``
    — moves that cost out of the first conversational turn.
    """
    _load_sentence_split()


class IndicSentenceChunker(SentenceChunker):
    """Sentence chunker for Indic scripts using indic-nlp-library.

    Falls back gracefully for unsupported ``language`` values by returning the
    input text as a single segment — the ``BufferedSentenceChunkStream`` then
    relies on idle-flush for phrasing.
    """

    def __init__(
        self,
        *,
        language: str = "hi",
        min_sentence_len: int = 1,
        idle_flush_ms: int = 400,
    ) -> None:
        """Initialise the chunker.

        Args:
            language: Default ISO 639-1 code. Override per-turn via
                ``tokenize(..., language=...)`` or ``stream(language=...)``.
            min_sentence_len: Passed through to the ``BufferedSentenceChunkStream``
                idle-flush heuristic. Default 1 (Indic sentences can be very short).
            idle_flush_ms: Idle timeout before a word-boundary cut fires.
        """
        if language not in INDIC_LANGS:
            logger.warning(
                "IndicSentenceChunker language %r is not in the supported set "
                "%s; sentence splitting may degrade. Consider using "
                "BasicSentenceChunker for non-Indic languages.",
                language,
                sorted(INDIC_LANGS),
            )
        self._default_language = language
        self._min_sentence_len = int(min_sentence_len)
        self._idle_flush_ms = int(idle_flush_ms)
        self._split_fn: "_SentenceSplit | None" = None

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        if not text or not text.strip():
            return []
        lang = self._resolve_language(language)
        split_fn = self._get_split_fn()
        try:
            sentences = split_fn(text, lang=lang)
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "indic-nlp-library sentence_split raised for lang=%r; returning "
                "input as single segment",
                lang,
                exc_info=True,
            )
            return [text.strip()]
        return [s.strip() for s in sentences if s and s.strip()]

    def tokenize_raw(self, text: str, *, language: str | None = None) -> list[str]:
        """Return segments with original whitespace preserved.

        Used by the stream adapter: the ``BufferedSentenceChunkStream`` re-uses
        the last segment as the continuation buffer, so losing edge whitespace
        would concatenate unrelated words (e.g. ``व्यापार`` + ``से`` →
        ``व्यापारसे``) when the next chunk arrives.

        indic-nlp-library's ``sentence_split`` returns stripped strings, so we
        map each stripped segment back to its position in the original text
        and slice raw ranges, preserving the whitespace that originally sat
        between sentences.
        """
        if not text:
            return []
        lang = self._resolve_language(language)
        split_fn = self._get_split_fn()
        try:
            stripped_sentences = split_fn(text, lang=lang)
        except Exception:  # pragma: no cover - defensive
            return [text]

        raw_segments: list[str] = []
        cursor = 0
        for s in stripped_sentences:
            core = s.strip() if s else ""
            if not core:
                continue
            idx = text.find(core, cursor)
            if idx < 0:
                # Shouldn't happen — defensive fallback keeps the stripped text.
                raw_segments.append(core)
                continue
            end = idx + len(core)
            raw_segments.append(text[cursor:end])
            cursor = end

        if cursor < len(text):
            if raw_segments:
                raw_segments[-1] = raw_segments[-1] + text[cursor:]
            else:
                raw_segments.append(text[cursor:])

        return raw_segments

    def stream(self, *, language: str | None = None) -> SentenceChunkStream:
        lang = self._resolve_language(language)
        tokenize_fn: Callable[[str], list[str]] = partial(
            self.tokenize_raw, language=lang
        )
        return BufferedSentenceChunkStream(
            tokenize_fn=tokenize_fn,
            strong_terminators="।॥.!?",  # Devanagari danda + Latin terminators
            min_sentence_len=self._min_sentence_len,
            idle_flush_ms=self._idle_flush_ms,
        )

    def _resolve_language(self, language: str | None) -> str:
        """Pick the language code to pass to the upstream library."""
        lang = (language or self._default_language or "hi").lower()
        if lang == "auto":
            return self._default_language
        return lang

    def _get_split_fn(self) -> "_SentenceSplit":
        """Lazy-load the upstream function on first use."""
        if self._split_fn is None:
            self._split_fn = _load_sentence_split()
        return self._split_fn


class IndicScriptTransliterator:
    """Thin wrapper around ``UnicodeIndicTransliterator``.

    Useful as a ``TextFilter``-adjacent utility when your LLM emits Hindi but
    the TTS speaks Telugu (or similar cross-Indic scenarios).

    Example::

        from videosdk.agents.tokenize import IndicScriptTransliterator

        trans = IndicScriptTransliterator(source="hi", target="te")
        out = trans.convert("नमस्ते दुनिया")
        # out == "నమస్తే దునియా" (approximate phonetic conversion)
    """

    def __init__(self, *, source: str, target: str) -> None:
        self._source = source
        self._target = target

    def convert(self, text: str) -> str:
        try:
            from indicnlp.transliterate.unicode_transliterate import (
                UnicodeIndicTransliterator,
            )
        except ImportError as exc:
            raise ImportError(
                "indic-nlp-library is missing — it ships as a dependency of "
                "videosdk-agents; reinstall with: uv pip install -U videosdk-agents"
            ) from exc
        return UnicodeIndicTransliterator.transliterate(text, self._source, self._target)
