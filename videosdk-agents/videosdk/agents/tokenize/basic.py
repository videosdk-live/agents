"""Multilingual, Unicode-aware default sentence tokenizer.

Zero external dependencies — only ``re`` and the stdlib.

Algorithm:

1. **Structural protection.** URLs, emails, version strings, file paths,
   decimals, acronyms, and dotted identifiers are matched by regex and each
   match is replaced with a single Private Use Area placeholder. Terminator
   characters inside these tokens therefore never trigger a sentence cut.
2. **Abbreviation protection.** For each known abbreviation in the resolved
   language set, ``Abbrev.`` is rewritten to ``Abbrev<placeholder>``, again
   hiding the dot from the terminator scan.
3. **Lookahead-before-commit scan.** Walk the protected text; at each strong
   terminator, look ahead to the next non-whitespace character. If one exists,
   cut after the terminator (inclusive); otherwise leave the remainder as the
   trailing segment so the caller can continue buffering.
4. **Restore.** Replace every placeholder with its original text before
   returning.

This gives correct behaviour across Latin, Indic, CJK, Arabic/Urdu, Armenian,
Ethiopic, Tibetan, Myanmar, Khmer, Greek, Cyrillic, Hebrew, Georgian, Thai,
Lao, and any other script whose strong terminators are listed in
``patterns.STRONG_TERMINATORS``.

Future plugin tokenizers (e.g. Blingfire, spaCy, ICU) plug into the same
``SentenceTokenizer`` interface without touching this file.
"""

from __future__ import annotations

import logging
import re
from functools import partial

from .base import SentenceStream, SentenceTokenizer
from .patterns import (
    ABBREVIATIONS_BY_LANG,
    ABBREVIATIONS_EN,
    ACRONYM_REGEX,
    CLOSING_QUOTES,
    DECIMAL_REGEX,
    EMAIL_REGEX,
    IDENTIFIER_REGEX,
    PATH_REGEX,
    PLACEHOLDER_BASE,
    STRONG_TERMINATORS,
    URL_REGEX,
    VERSION_REGEX,
    WEAK_TERMINATORS,
    detect_script,
)

logger = logging.getLogger(__name__)


# Order matters: protect the most specific patterns first so a broad pattern
# cannot eat a substring of a more specific one (e.g. the decimal regex must
# not fire inside an already-protected URL).
_STRUCTURAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    URL_REGEX,
    EMAIL_REGEX,
    PATH_REGEX,
    VERSION_REGEX,
    ACRONYM_REGEX,
    IDENTIFIER_REGEX,
    DECIMAL_REGEX,
)


def _placeholder_char(idx: int) -> str:
    """Return a single Private Use Area char for placeholder index ``idx``."""
    return chr(PLACEHOLDER_BASE + idx)


class BasicSentenceTokenizer(SentenceTokenizer):
    """Default multilingual, Unicode-aware sentence tokenizer.

    Works correctly without a ``language`` hint for every major world script.
    An explicit hint sharpens behaviour (abbreviation set, Greek ``;`` upgrade)
    but is not required.

    Args:
        language: ISO 639-1 code or ``"auto"`` (default). ``"auto"`` triggers
            script detection on the first ~200 characters of each tokenize call.
        min_sentence_len: Minimum character length before a weak terminator
            (``, ; :`` etc.) is treated as a cut boundary. Strong terminators
            (``. ! ? 。 ！ ？ । 。 ۔ ։ ።`` …) always cut regardless of length.
        strong_terminators: Override the default strong-terminator character class.
        weak_terminators: Override the default weak-terminator character class.
        abbreviations: Override the abbreviation set. When ``None`` (default),
            the set is chosen from ``ABBREVIATIONS_BY_LANG`` based on the
            resolved language.
    """

    def __init__(
        self,
        *,
        language: str = "auto",
        min_sentence_len: int = 20,
        strong_terminators: str = STRONG_TERMINATORS,
        weak_terminators: str = WEAK_TERMINATORS,
        abbreviations: frozenset[str] | None = None,
    ) -> None:
        self._language = language
        self._min_sentence_len = max(1, int(min_sentence_len))
        self._strong = strong_terminators
        self._weak = weak_terminators
        self._override_abbreviations = abbreviations

    # ------------------------------------------------------------------ #
    # SentenceTokenizer interface
    # ------------------------------------------------------------------ #

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        """Split ``text`` into sentence-sized strings.

        Args:
            text: Input text. May be a full response or a partial buffer.
            language: Optional ISO 639-1 override. When omitted, uses the
                instance's default.

        Returns:
            Sentence strings in input order, leading/trailing whitespace
            stripped, empty segments filtered out. When the input ends without
            a confirmed sentence boundary, the final element contains the
            unterminated remainder (the stream adapter uses this to retain
            buffer state).
        """
        if not text:
            return []

        lang = language or self._language
        if lang == "auto":
            lang = detect_script(text[:200])

        strong = self._strong
        weak = self._weak
        if lang == "el":
            # Greek uses ``;`` as question mark and ``·`` as semicolon.
            if ";" not in strong:
                strong = strong + ";"
            weak = weak.replace(";", "")

        if self._override_abbreviations is not None:
            abbrevs = self._override_abbreviations
        else:
            abbrevs = ABBREVIATIONS_BY_LANG.get(lang, ABBREVIATIONS_EN)

        protected, restore_map = self._protect(text, abbrevs)
        segments = self._split(protected, strong=strong, weak=weak)
        return [self._restore(s, restore_map) for s in segments if s.strip()]

    def tokenize_raw(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> list[str]:
        """Return raw (un-stripped) segment substrings.

        Same boundary decisions as :py:meth:`tokenize`, but each segment is
        returned as it appears in the input text — leading / trailing
        whitespace intact. Used by the stream adapter so it can re-use the
        final segment as the continuation buffer without losing the space
        that separates it from the upcoming chunk.
        """
        if not text:
            return []

        lang = language or self._language
        if lang == "auto":
            lang = detect_script(text[:200])

        strong = self._strong
        weak = self._weak
        if lang == "el":
            if ";" not in strong:
                strong = strong + ";"
            weak = weak.replace(";", "")

        if self._override_abbreviations is not None:
            abbrevs = self._override_abbreviations
        else:
            abbrevs = ABBREVIATIONS_BY_LANG.get(lang, ABBREVIATIONS_EN)

        protected, restore_map = self._protect(text, abbrevs)
        segments = self._split(protected, strong=strong, weak=weak)
        return [self._restore_raw(s, restore_map) for s in segments]

    @staticmethod
    def _restore_raw(text: str, restore_map: dict[str, str]) -> str:
        """Same as :py:meth:`_restore` but preserves edge whitespace."""
        if not restore_map:
            return text
        for key, value in restore_map.items():
            text = text.replace(key, value)
        return text

    def stream(self, *, language: str | None = None) -> SentenceStream:
        """Open a push-based stream adapter bound to this tokenizer."""
        # Local import to avoid a cycle between base/basic/stream at module load.
        from .stream import BufferedSentenceStream

        return BufferedSentenceStream(
            tokenize_fn=partial(self.tokenize_raw, language=language),
            strong_terminators=self._resolve_strong_for_stream(language),
            min_sentence_len=self._min_sentence_len,
        )

    # ------------------------------------------------------------------ #
    # Protection / restoration helpers
    # ------------------------------------------------------------------ #

    def _protect(
        self,
        text: str,
        abbreviations: frozenset[str],
    ) -> tuple[str, dict[str, str]]:
        """Replace structural tokens and abbreviation dots with placeholders.

        Each protected region becomes a single PUA character that the split
        scan cannot interpret as a terminator.
        """
        restore_map: dict[str, str] = {}
        counter = [0]

        def _substitute(match: re.Match[str]) -> str:
            key = _placeholder_char(counter[0])
            counter[0] += 1
            restore_map[key] = match.group(0)
            return key

        for regex in _STRUCTURAL_PATTERNS:
            text = regex.sub(_substitute, text)

        if abbreviations:
            # Build an alternation matching `<abbrev>.` exactly.
            # Sort longest-first so "Ph.D" wins over "Ph".
            escaped = sorted(
                (re.escape(a) for a in abbreviations),
                key=len,
                reverse=True,
            )
            abbrev_regex = re.compile(
                r"\b(?:" + "|".join(escaped) + r")\.",
                re.IGNORECASE,
            )
            text = abbrev_regex.sub(_substitute, text)

        return text, restore_map

    @staticmethod
    def _restore(text: str, restore_map: dict[str, str]) -> str:
        """Replace every placeholder with its original text and strip edges."""
        if not restore_map:
            return text.strip()
        for key, value in restore_map.items():
            text = text.replace(key, value)
        return text.strip()

    # ------------------------------------------------------------------ #
    # Core split scan
    # ------------------------------------------------------------------ #

    def _split(
        self,
        text: str,
        *,
        strong: str,
        weak: str,
    ) -> list[str]:
        """Walk ``text`` and return a list of sentence segments.

        Strong terminators always cut (with lookahead-before-commit). Weak
        terminators cut only after the accumulated buffer reaches
        ``min_sentence_len`` chars so we don't emit fragments from every comma
        at the start of a response.
        """
        strong_set = set(strong)
        weak_set = set(weak)
        closing_quote_set = set(CLOSING_QUOTES)

        segments: list[str] = []
        start = 0
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]
            if ch not in strong_set and ch not in weak_set:
                i += 1
                continue

            # Some terminators cluster (e.g. "?!" or "..."). Consume the whole
            # run so we emit one logical boundary, not one per character.
            j = i + 1
            cur_set = strong_set if ch in strong_set else weak_set
            while j < n and text[j] in cur_set:
                j += 1

            # If the cluster is immediately followed by a closing quote,
            # extend the cut to include the quote so `"Hello."` becomes one
            # segment rather than `"Hello.` + `"`.
            if j < n and text[j] in closing_quote_set:
                j += 1

            # Lookahead: scan past horizontal whitespace to find the next
            # non-whitespace character. A newline breaks the lookahead wait —
            # it's a hard paragraph boundary.
            look = j
            saw_newline = ch == "\n"
            while look < n and text[look] in " \t":
                look += 1
            if look < n and text[look] == "\n":
                saw_newline = True
                look += 1
                while look < n and text[look] in " \t":
                    look += 1

            if ch in strong_set:
                if look < n or saw_newline:
                    # Confirmed strong boundary.
                    segments.append(text[start:j])
                    start = j
                    i = look
                    continue
                # End of text at a strong terminator with no lookahead —
                # don't commit; let the caller decide (stream adapter will
                # wait for more text or flush on end_input).
                break

            # Weak terminator: cut only when the running buffer is long enough.
            cluster_end = j
            if cluster_end - start >= self._min_sentence_len and look < n:
                segments.append(text[start:cluster_end])
                start = cluster_end
                i = look
                continue

            i = j

        if start < n:
            segments.append(text[start:n])

        return segments

    # ------------------------------------------------------------------ #
    # Stream-side strong-terminator resolution
    # ------------------------------------------------------------------ #

    def _resolve_strong_for_stream(self, language: str | None) -> str:
        """Return the strong-terminator set the stream adapter should use for fast-path checks."""
        lang = language or self._language
        if lang == "el" and ";" not in self._strong:
            return self._strong + ";"
        return self._strong
