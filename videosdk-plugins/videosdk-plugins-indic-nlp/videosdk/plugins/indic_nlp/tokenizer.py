"""``SentenceTokenizer`` implementation backed by indic-nlp-library.

Wraps ``indicnlp.tokenize.sentence_tokenize.sentence_split`` so any cascade
``Pipeline`` can plug in richer Indic sentence segmentation without touching
core code.

Usage::

    from videosdk.agents import Pipeline
    from videosdk.plugins.indic_nlp import IndicSentenceTokenizer

    pipeline = Pipeline(
        stt=..., llm=..., tts=..., vad=..., turn_detector=...,
        tokenizer=IndicSentenceTokenizer(language="hi"),
        chunking_language="hi",
    )

Why this plugin exists — differences vs. the built-in
``BasicSentenceTokenizer``:

* Honours Indic honorifics (``डॉ.``, ``श्री.``, ``प्रो.``) so a period after
  them doesn't cut a sentence.
* Per-language heuristics tuned for Hindi, Tamil, Telugu, Kannada, Bengali,
  Marathi, Gujarati, Punjabi, Malayalam, Odia, Assamese, Sinhala, Urdu.
* Optional script transliteration between Indic scripts (``IndicScriptTransliterator``).

When to keep using ``BasicSentenceTokenizer`` instead:

* Mixed-script agents (Latin + Indic + CJK in one response).
* English-dominant deployments where a pure-stdlib core matters.
* Cold-start-sensitive workers — ``indic-nlp-library`` adds ~5 MB of imports.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from videosdk.agents.tokenize import (
    BufferedSentenceStream,
    SentenceStream,
    SentenceTokenizer,
)

if TYPE_CHECKING:
    from indicnlp.tokenize.sentence_tokenize import sentence_split as _SentenceSplit

logger = logging.getLogger(__name__)

# Language codes supported by indic-nlp-library's sentence_split / normalisers.
# Keys: ISO 639-1 codes; values: library internal codes (same, but aliased for
# clarity — a small few differ).
SUPPORTED_LANGUAGES: frozenset[str] = frozenset({
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


def _load_sentence_split() -> "_SentenceSplit":
    """Lazy import of the upstream function with a friendly error message."""
    try:
        from indicnlp.tokenize.sentence_tokenize import sentence_split

        return sentence_split
    except ImportError as exc:
        raise ImportError(
            "videosdk-plugins-indic-nlp requires the `indic-nlp-library` package. "
            "Install it with: uv pip install indic-nlp-library"
        ) from exc


def pre_warm_tokenizer() -> None:
    """Eagerly import ``indic-nlp-library`` so the first ``.tokenize()`` call is cheap.

    The underlying ``indicnlp.tokenize.sentence_tokenize`` module performs its
    expensive initialisation on first import (~6s on a cold Python process).
    Calling this at worker start — alongside ``TurnDetector.pre_download_model()``
    — moves that cost out of the first conversational turn.
    """
    _load_sentence_split()


class IndicSentenceTokenizer(SentenceTokenizer):
    """Sentence tokenizer for Indic scripts using indic-nlp-library.

    Falls back gracefully for unsupported ``language`` values by returning the
    input text as a single segment — the ``BufferedSentenceStream`` will then
    rely on idle-flush for phrasing. Consider using the core
    ``BasicSentenceTokenizer`` if your deployment is not Indic-heavy.
    """

    def __init__(
        self,
        *,
        language: str = "hi",
        min_sentence_len: int = 1,
        idle_flush_ms: int = 400,
    ) -> None:
        """Initialise the tokenizer.

        Args:
            language: Default ISO 639-1 code. Override per-turn via
                ``tokenize(..., language=...)`` or ``stream(language=...)``.
            min_sentence_len: Passed through to the ``BufferedSentenceStream``
                idle-flush heuristic. Default 1 (Indic sentences can be very short).
            idle_flush_ms: Idle timeout before a word-boundary cut fires.
        """
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(
                "IndicSentenceTokenizer language %r is not in the supported set "
                "%s; sentence splitting may degrade. Consider using "
                "BasicSentenceTokenizer for non-Indic languages.",
                language,
                sorted(SUPPORTED_LANGUAGES),
            )
        self._default_language = language
        self._min_sentence_len = int(min_sentence_len)
        self._idle_flush_ms = int(idle_flush_ms)
        self._split_fn: "_SentenceSplit | None" = None

    # ------------------------------------------------------------------ #
    # SentenceTokenizer interface
    # ------------------------------------------------------------------ #

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

        Used by the stream adapter: the ``BufferedSentenceStream`` re-uses the
        last segment as the continuation buffer, so losing edge whitespace
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

        # Whatever trails after the last sentence (typically whitespace) must
        # be attached to the last segment so the stream adapter's continuation
        # buffer preserves the separator before the next chunk.
        if cursor < len(text):
            if raw_segments:
                raw_segments[-1] = raw_segments[-1] + text[cursor:]
            else:
                raw_segments.append(text[cursor:])

        return raw_segments

    def stream(self, *, language: str | None = None) -> SentenceStream:
        lang = self._resolve_language(language)
        # Stream uses the whitespace-preserving variant so the continuation
        # buffer keeps the space that separates sentences across chunks.
        tokenize_fn: Callable[[str], list[str]] = partial(
            self.tokenize_raw, language=lang
        )
        return BufferedSentenceStream(
            tokenize_fn=tokenize_fn,
            strong_terminators="।॥.!?",  # Devanagari danda + Latin terminators
            min_sentence_len=self._min_sentence_len,
            idle_flush_ms=self._idle_flush_ms,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

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

    Useful as a custom ``TextFilter``-adjacent utility when your LLM emits
    Hindi but the TTS speaks Telugu (or similar cross-Indic scenarios).

    Example::

        from videosdk.plugins.indic_nlp import IndicScriptTransliterator

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
                "videosdk-plugins-indic-nlp requires the `indic-nlp-library` "
                "package. Install it with: uv pip install indic-nlp-library"
            ) from exc
        return UnicodeIndicTransliterator.transliterate(text, self._source, self._target)
