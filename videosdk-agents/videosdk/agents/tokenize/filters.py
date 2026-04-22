"""Pre-tokenization text filter.

Handles the English-prosody issues enumerated in
``LLM_TTS_CHUNKING_BEHAVIOR.md`` §6 — Markdown leakage, symbol read-aloud,
punctuation-cluster splitting, quoted-speech handling, mid-path fragmentation,
parenthetical loss.

Runs *before* the sentence tokenizer in the cascade flow:

    LLM deltas -> BasicTextFilter -> BasicSentenceTokenizer -> user hook -> TTS
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator

from .base import TextFilter
from .patterns import (
    EMAIL_REGEX,
    MD_BLOCKQUOTE_REGEX,
    MD_BOLD_STAR_REGEX,
    MD_BOLD_UNDER_REGEX,
    MD_FENCED_CODE_REGEX,
    MD_HEADING_REGEX,
    MD_HR_REGEX,
    MD_IMAGE_REGEX,
    MD_INLINE_CODE_REGEX,
    MD_ITALIC_STAR_REGEX,
    MD_ITALIC_UNDER_REGEX,
    MD_LINK_REGEX,
    MD_LIST_MARKER_REGEX,
    MD_TABLE_PIPE_REGEX,
    MD_TABLE_SEP_REGEX,
    METADATA_PARENS_REGEX,
    METADATA_PREFIX_REGEX,
    SCRIPT_MIXED_PAREN_REGEX,
    PATH_REGEX,
    PLACEHOLDER_BASE,
    PUNCT_ELLIPSIS_REGEX,
    PUNCT_SPACED_DASH_REGEX,
    RANGE_REGEX,
    RANGE_REGEX_WIDE,
    RANGE_SEPARATOR_BY_LANG,
    SPELL_LONG_DIGITS_REGEX,
    SPELL_PHONE_REGEX,
    SYMBOL_EXPANSIONS_EN,
    URL_REGEX,
    VERSION_REGEX,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Buffering strategy
# ---------------------------------------------------------------------------
# A chunk that contains a potentially-open pattern (an opening "**" with no
# matching "**" yet, a single backtick, a "[" with no matching "]", a "```"
# with no matching close) must be held back until the pattern either closes
# or the buffer grows past ``_MAX_BUFFER`` chars. The latter is a safety
# valve so a broken response can't stall synthesis forever.
_MAX_BUFFER: int = 2000

# Shift-in offset for filter-local placeholders so they don't collide with
# the tokenizer's placeholders (which start at PLACEHOLDER_BASE = U+E000).
_FILTER_PLACEHOLDER_BASE: int = PLACEHOLDER_BASE + 0x1000


class BasicTextFilter(TextFilter):
    """Default text filter with six independently-toggleable rules.

    All rules are on by default. Symbol expansion is suppressed for non-English
    languages; the TTS provider handles symbol readings for those.
    """

    def __init__(
        self,
        *,
        language: str = "auto",
        strip_markdown: bool = True,
        strip_llm_metadata: bool = True,
        collapse_script_parens: bool = True,
        normalise_punctuation: bool = True,
        expand_symbols: bool = True,
        expand_ranges: bool = True,
        protect_structural: bool = True,
        respect_quotes: bool = True,
        respect_parens: bool = True,
        ssml_flavor: str = "none",
    ) -> None:
        """Initialise the filter.

        Args:
            language: ISO 639-1 code or ``"auto"``. Drives symbol expansion
                (English only) and numeric-range separator word ("to", "से",
                "から", etc.).
            strip_markdown: Strip Markdown syntax before TTS. Default ``True``.
            strip_llm_metadata: Strip common LLM state-leak shapes such as
                ``(SESSION STATE: intro_delivered=true)`` or
                ``STATE: language=en``. Safety net — the real fix is a
                clearer prompt. Default ``True``.
            collapse_script_parens: When the LLM writes both an English word
                and its non-Latin gloss (``Tally (टैली)``, ``one click
                (एक क्लिक)``), collapse to just the gloss so the TTS doesn't
                speak the phrase twice. Word-count aware: ``Extra manpower
                (मैनपावर)`` → ``Extra मैनपावर``. Default ``True``.
            normalise_punctuation: Collapse ``"..."`` → ``"…"`` etc. Default ``True``.
            expand_symbols: Rewrite ``&``, ``%``, ``#N`` → words (English only).
                Default ``True``.
            expand_ranges: Rewrite short digit ranges ``2-3`` → ``2 to 3``
                using the locale's separator word. Default ``True``.
            protect_structural: Preserve URLs / paths / versions / emails
                through the Markdown and symbol stages. Default ``True``.
            respect_quotes: (Reserved — currently a no-op; preserved state
                is tracked per turn so future logic can use it.)
            respect_parens: (Reserved — same as above.)
            ssml_flavor: TTS-specific SSML injection. ``"none"`` (default)
                emits plain text. ``"cartesia"`` wraps phone numbers and
                long digit runs in ``<spell>...</spell>`` tags so Cartesia
                Sonic-3 reads them digit-by-digit (e.g. account IDs, OTPs).
                Other TTS providers would read the tags literally — keep
                ``"none"`` unless the target TTS is Cartesia Sonic-3.
        """
        self._language = language
        self._strip_markdown = strip_markdown
        self._strip_llm_metadata = strip_llm_metadata
        self._collapse_script_parens = collapse_script_parens
        self._normalise_punctuation = normalise_punctuation
        self._expand_symbols = expand_symbols
        self._expand_ranges = expand_ranges
        self._protect_structural = protect_structural
        self._respect_quotes = respect_quotes
        self._respect_parens = respect_parens
        self._ssml_flavor = (ssml_flavor or "none").lower()

        self._buffer: str = ""
        self._in_code_fence: bool = False
        self._placeholder_counter: int = 0
        self._placeholder_map: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # TextFilter interface
    # ------------------------------------------------------------------ #

    async def filter(self, chunks: AsyncIterator[str]) -> AsyncIterator[str]:
        """Transform an incoming chunk stream."""
        await self.reset()
        try:
            async for chunk in chunks:
                if not chunk:
                    continue
                logger.debug("[chunking] filter ← raw: %r", chunk)
                self._buffer += chunk

                # Resolve any fence state before emitting. Loop because a
                # fence may open, close, and then a second fence may begin
                # in the same chunk.
                emitted_before_fence: list[str] = []
                while True:
                    if self._in_code_fence:
                        close_idx = self._buffer.find("```")
                        if close_idx == -1:
                            # Whole buffer is inside the fence — discard it.
                            self._buffer = ""
                            break
                        # Exit fence; consume through the closing marker.
                        self._buffer = self._buffer[close_idx + 3 :]
                        self._in_code_fence = False
                        continue  # look for another fence

                    open_idx = self._buffer.find("```")
                    if open_idx == -1:
                        break  # no fence in buffer, normal emit logic
                    close_idx = self._buffer.find("```", open_idx + 3)
                    if close_idx != -1:
                        break  # complete fence — MD_FENCED_CODE_REGEX removes it
                    # Unclosed open fence: emit the clean prefix, discard the
                    # opening marker, enter fence mode.
                    prefix = self._buffer[:open_idx]
                    self._buffer = ""
                    self._in_code_fence = True
                    if prefix:
                        processed = self._process(prefix)
                        if processed:
                            emitted_before_fence.append(processed)
                    break

                for piece in emitted_before_fence:
                    logger.debug("[chunking] filter → tokenizer: %r", piece)
                    yield piece

                if self._in_code_fence or not self._buffer:
                    continue

                # Normal emission logic.
                safe_cut = self._find_emit_boundary(self._buffer)
                if safe_cut <= 0:
                    if len(self._buffer) > _MAX_BUFFER:
                        processed = self._process(self._buffer)
                        self._buffer = ""
                        if processed:
                            logger.debug("[chunking] filter → tokenizer: %r", processed)
                            yield processed
                    continue
                processed = self._process(self._buffer[:safe_cut])
                self._buffer = self._buffer[safe_cut:]
                if processed:
                    logger.debug("[chunking] filter → tokenizer: %r", processed)
                    yield processed
            # Drain on end of stream.
            if self._in_code_fence:
                # Unclosed fence at end of stream — discard remainder.
                self._buffer = ""
                self._in_code_fence = False
            if self._buffer:
                processed = self._process(self._buffer)
                self._buffer = ""
                if processed:
                    logger.debug("[chunking] filter → tokenizer (drain): %r", processed)
                    yield processed
        except Exception:
            logger.error("BasicTextFilter errored; yielding buffer raw", exc_info=True)
            if self._buffer:
                yield self._buffer
                self._buffer = ""

    async def reset(self) -> None:
        """Reset per-turn state."""
        self._buffer = ""
        self._in_code_fence = False
        self._placeholder_counter = 0
        self._placeholder_map = {}

    # ------------------------------------------------------------------ #
    # Buffer management
    # ------------------------------------------------------------------ #

    def _find_emit_boundary(self, text: str) -> int:
        """Return the length of the prefix that can be safely emitted.

        Assumes fence handling has already happened: this only worries about
        balanced inline Markdown markers (``**``, ``__``, backticks, brackets).

        Returns 0 when no safe whitespace boundary exists yet.
        """
        if not text:
            return 0
        for idx in range(len(text) - 1, -1, -1):
            if not text[idx].isspace():
                continue
            candidate = text[: idx + 1]
            if self._has_balanced_markers(candidate):
                return idx + 1
        return 0

    @staticmethod
    def _has_balanced_markers(text: str) -> bool:
        """Cheap balance check for common Markdown / grouping markers.

        Ensures we never emit a prefix that leaves an opener dangling —
        needed so that:
        - inline ``**bold**`` / ``_italic_`` / ``` `code` ``` all close,
        - Markdown links ``[text](url)`` both brackets close,
        - and, critically, parentheses balance. The last matters for the
          metadata stripper: an LLM-leaked ``(SESSION STATE: …)`` must
          reach the processor as one intact block so the regex can delete it.
        """
        if text.count("**") % 2 != 0:
            return False
        if text.count("__") % 2 != 0:
            return False
        bt_count = sum(1 for c in text if c == "`")
        if bt_count % 2 != 0:
            return False
        if text.count("[") != text.count("]"):
            return False
        if text.count("(") != text.count(")"):
            return False
        return True

    # ------------------------------------------------------------------ #
    # Core processing pipeline — deterministic stage order
    # ------------------------------------------------------------------ #

    def _process(self, text: str) -> str:
        if not text:
            return ""

        if self._protect_structural:
            text = self._protect(text)

        if self._strip_llm_metadata:
            text = self._strip_metadata(text)

        if self._strip_markdown:
            text = self._strip_md(text)

        if self._collapse_script_parens:
            text = self._collapse_script_parens_fn(text)

        if self._normalise_punctuation:
            text = self._normalise_punct(text)

        # Spell-wrap MUST run before range expansion: once phones are wrapped
        # in ``<spell>…</spell>``, the range regex can safely use a wider
        # digit allowance without mistaking a phone for a range.
        if self._ssml_flavor == "cartesia":
            text = self._inject_cartesia_ssml(text)

        if self._expand_ranges:
            text = self._expand_numeric_ranges(text)

        if self._expand_symbols and self._language_is_english():
            text = self._expand(text)

        if self._protect_structural:
            text = self._restore(text)

        return text

    def _language_is_english(self) -> bool:
        lang = (self._language or "").lower()
        return lang in ("en", "auto", "")

    # ------------------------------------------------------------------ #
    # Structural protection
    # ------------------------------------------------------------------ #

    def _protect(self, text: str) -> str:
        def _substitute(match: re.Match[str]) -> str:
            key = chr(_FILTER_PLACEHOLDER_BASE + self._placeholder_counter)
            self._placeholder_counter += 1
            self._placeholder_map[key] = match.group(0)
            return key

        for regex in (URL_REGEX, EMAIL_REGEX, PATH_REGEX, VERSION_REGEX):
            text = regex.sub(_substitute, text)
        return text

    def _restore(self, text: str) -> str:
        if not self._placeholder_map:
            return text
        for key, value in self._placeholder_map.items():
            if key in text:
                text = text.replace(key, value)
        return text

    # ------------------------------------------------------------------ #
    # LLM metadata stripping — safety net for state-leak bugs
    # ------------------------------------------------------------------ #

    @staticmethod
    def _strip_metadata(text: str) -> str:
        """Remove common LLM state-leak shapes.

        Targets patterns the model produces when it misinterprets a
        ``SESSION STATE`` / ``INTERNAL`` block in the system prompt as
        something to announce: parenthesised ALL-CAPS key/value dumps, and
        bare ``KEYWORD: key=value`` lines. Designed to never match normal
        parentheticals — the key detector is ``= inside all-caps-prefixed``.
        """
        text = METADATA_PARENS_REGEX.sub("", text)
        text = METADATA_PREFIX_REGEX.sub("", text)
        return text

    # ------------------------------------------------------------------ #
    # Script-mixed parenthetical collapse
    # ------------------------------------------------------------------ #

    @staticmethod
    def _collapse_script_parens_fn(text: str) -> str:
        """Collapse ``<Latin word(s)> (<non-Latin gloss>)`` → ``<gloss>``.

        Walks each ``(…)`` containing a non-ASCII character, counts its
        words, and removes exactly that many Latin-script words from the
        immediately-preceding text. Preserves whitespace and non-Latin
        context around the match.
        """
        matches = list(SCRIPT_MIXED_PAREN_REGEX.finditer(text))
        if not matches:
            return text

        result: list[str] = []
        pos = 0
        for m in matches:
            paren_start = m.start()
            paren_end = m.end()
            content = m.group(1).strip()
            if not content:
                result.append(text[pos:paren_end])
                pos = paren_end
                continue

            n_words = len(content.split())

            # Try to peel off n_words Latin words from the text that sits
            # between the previous processed position and this paren opening.
            before = text[pos:paren_start]
            idx = len(before)
            removed = 0
            while removed < n_words:
                # Skip trailing whitespace
                while idx > 0 and before[idx - 1].isspace():
                    idx -= 1
                end = idx
                # Walk back one Latin word (ASCII alphanumerics only)
                while (
                    idx > 0
                    and before[idx - 1].isascii()
                    and before[idx - 1].isalnum()
                ):
                    idx -= 1
                if end == idx:
                    break  # no Latin word to peel
                removed += 1

            if removed == n_words:
                # Full collapse: emit text before the peeled words, then the
                # gloss content. Add a single separating space when needed.
                kept = before[:idx]
                if kept and not kept[-1:].isspace():
                    result.append(kept + " ")
                else:
                    result.append(kept)
                result.append(content)
            else:
                # Not enough Latin words to match; leave the parens as-is.
                result.append(before)
                result.append(text[paren_start:paren_end])
            pos = paren_end

        result.append(text[pos:])
        return "".join(result)

    # ------------------------------------------------------------------ #
    # Markdown stripping
    # ------------------------------------------------------------------ #

    @staticmethod
    def _strip_md(text: str) -> str:
        # Fenced code blocks — drop entirely.
        text = MD_FENCED_CODE_REGEX.sub("", text)
        # Inline code — drop content (it rarely reads well).
        text = MD_INLINE_CODE_REGEX.sub("", text)
        # Images — drop entirely (alt text is usually redundant).
        text = MD_IMAGE_REGEX.sub("", text)
        # Links — keep anchor text, drop URL.
        text = MD_LINK_REGEX.sub(r"\1", text)
        # Headings — drop the leading "#" markers, keep the text.
        text = MD_HEADING_REGEX.sub("", text)
        # List markers — drop the bullet / number, keep the item text.
        text = MD_LIST_MARKER_REGEX.sub("", text)
        # Blockquote markers — drop.
        text = MD_BLOCKQUOTE_REGEX.sub("", text)
        # Horizontal rules — drop.
        text = MD_HR_REGEX.sub("", text)
        # Table separators — drop.
        text = MD_TABLE_SEP_REGEX.sub("", text)
        # Table pipes — replace with a single space so cells read as a list.
        text = MD_TABLE_PIPE_REGEX.sub(" ", text)
        # Bold / italic — keep inner text.
        text = MD_BOLD_STAR_REGEX.sub(r"\1", text)
        text = MD_BOLD_UNDER_REGEX.sub(r"\1", text)
        text = MD_ITALIC_STAR_REGEX.sub(r"\1", text)
        text = MD_ITALIC_UNDER_REGEX.sub(r"\1", text)
        return text

    # ------------------------------------------------------------------ #
    # Punctuation normalisation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise_punct(text: str) -> str:
        text = PUNCT_ELLIPSIS_REGEX.sub("…", text)
        text = PUNCT_SPACED_DASH_REGEX.sub(" — ", text)
        return text

    # ------------------------------------------------------------------ #
    # Symbol expansion
    # ------------------------------------------------------------------ #

    @staticmethod
    def _expand(text: str) -> str:
        for regex, replacement in SYMBOL_EXPANSIONS_EN:
            text = regex.sub(replacement, text)
        return text

    # ------------------------------------------------------------------ #
    # Cartesia SSML injection
    # ------------------------------------------------------------------ #

    @staticmethod
    def _inject_cartesia_ssml(text: str) -> str:
        """Wrap phone numbers and long digit runs in ``<spell>…</spell>``.

        Cartesia Sonic-3 reads unwrapped digit strings as natural-language
        numbers ("one thousand two hundred thirty four"). For IDs, account
        numbers, OTPs, and phone numbers you almost always want digit-by-digit.

        The order matters: phone regex runs first (more specific), then the
        long-digit fallback for any standalone 7+ digit run.
        """
        text = SPELL_PHONE_REGEX.sub(lambda m: f"<spell>{m.group(0)}</spell>", text)
        text = SPELL_LONG_DIGITS_REGEX.sub(
            lambda m: f"<spell>{m.group(0)}</spell>", text
        )
        return text

    # ------------------------------------------------------------------ #
    # Numeric-range expansion
    # ------------------------------------------------------------------ #

    def _expand_numeric_ranges(self, text: str) -> str:
        """Rewrite ``N-M`` as ``N <sep> M`` using the locale's separator word.

        Regex width is adaptive:

        * With ``ssml_flavor="cartesia"`` (phones already wrapped in
          ``<spell>…</spell>``) — uses the WIDE regex (up to 4 digits per
          side), so ``500-1000 entries`` becomes ``500 से 1000 entries``.
        * Otherwise — uses the narrow 3-digit regex so an English phone
          like ``555-1234`` is not mistaken for a range.
        * For non-Latin agents (Hindi, Tamil, etc.), we also use the wide
          regex because phone numbers are rare in those prose contexts,
          while range patterns (दाम पांच सौ से हज़ार, etc.) are common.
        """
        lang = (self._language or "").lower()
        separator = RANGE_SEPARATOR_BY_LANG.get(lang)
        if separator is None:
            return text
        regex = (
            RANGE_REGEX_WIDE
            if self._ssml_flavor == "cartesia" or not self._language_is_english()
            else RANGE_REGEX
        )
        return regex.sub(
            lambda m: f"{m.group(1)} {separator} {m.group(2)}",
            text,
        )
