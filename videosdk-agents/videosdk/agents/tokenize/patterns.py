"""Multilingual delimiter catalogue and regex patterns.

Centralises every regex, character class, and abbreviation set used by the
tokenizer and filter. Keeping these in one module makes it easy to add a new
script family (one dict entry) and makes tests straightforward to parameterise.

All sets are frozen / immutable at module load time; no runtime mutation.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Strong terminators: end-of-sentence across world scripts.
# ---------------------------------------------------------------------------
# A strong terminator flushes the tokenizer's buffer as soon as the next
# non-whitespace character is seen (lookahead-before-commit). Sets from
# different scripts are disjoint by Unicode construction, so the full union
# can always be active simultaneously without false positives.
STRONG_TERMINATORS: str = (
    # Latin / ASCII
    ".!?"
    # Unicode ellipsis + interrobangs + double punctuation
    "…"           # horizontal ellipsis
    "‼⁇⁈⁉"  # double-exclamation, double-question, !?, ?!
    # CJK fullwidth (Chinese, Japanese, Korean)
    "。！？"
    # Indic danda & double danda (Hindi, Bengali, Marathi, Sanskrit, Gujarati,
    # Punjabi, Odia, Tamil, Telugu, Kannada, Malayalam, Sinhala)
    "।॥"
    # Arabic / Urdu / Persian
    "؟"           # Arabic question mark
    "۔"           # Arabic/Urdu full stop
    "؞"           # Arabic triple dot punctuation
    # Armenian full stop only. `՞` (question) and `՜` (exclamation) are
    # placed before the stressed syllable of the questioned / exclaimed word,
    # so they may appear mid-word and are unreliable sentence terminators.
    "։"
    # Ethiopic (Amharic, Tigrinya)
    "።፧፨"
    # Tibetan
    "།༎༏༐༑"
    # Myanmar
    "။"
    # Khmer
    "។៕"
    # Hard line break (paragraph split)
    "\n"
)

# ---------------------------------------------------------------------------
# Weak terminators: clause-level pauses.
# ---------------------------------------------------------------------------
# These flush only after the buffer has reached ``min_chars`` to avoid chopping
# on every comma at the start of a response.
WEAK_TERMINATORS: str = (
    # Latin / ASCII
    ",;:"
    # CJK fullwidth
    "、，；："
    # Arabic / Urdu
    "،؛"
    # Armenian
    "՝՛"
    # Ethiopic
    "፣፤፥"
    # Tibetan tsheg
    "་"
    # Myanmar little section
    "၊"
    # Khmer khan
    "៖"
    # Em-dash / en-dash — weak pause in every Latin-script language
    "—–"
)

# Characters that terminate a sentence when followed by the closing form of a
# paired quote (handles `"Hello." She left.` → after the `."`).
CLOSING_QUOTES: str = "\"'”’»"

# ---------------------------------------------------------------------------
# Script ranges for no-whitespace languages.
# ---------------------------------------------------------------------------
# When idle-flushing a long buffer without a terminator, we need a word-boundary
# to cut on. In whitespace languages we use spaces; in CJK / Thai / Lao /
# Myanmar / Khmer we cut on any character in these ranges (each character is
# its own "word" for fallback-cut purposes).
NO_SPACE_SCRIPTS_REGEX: re.Pattern[str] = re.compile(
    "["
    "一-鿿"      # CJK Unified Ideographs
    "㐀-䶿"      # CJK Extension A
    "豈-﫿"      # CJK Compatibility Ideographs
    "぀-ゟ"      # Hiragana
    "゠-ヿ"      # Katakana
    "가-힯"      # Hangul syllables
    "฀-๿"      # Thai
    "຀-໿"      # Lao
    "က-႟"      # Myanmar
    "ក-៿"      # Khmer
    "]"
)

# ---------------------------------------------------------------------------
# Abbreviation guards — per language.
# ---------------------------------------------------------------------------
# When the tokenizer sees `<word>.`, it looks up ``<word>`` in the language's
# set; a match suppresses the terminator so `Dr. Smith` stays joined.
#
# Non-Latin scripts rarely use period-abbreviation, so their sets are empty.

ABBREVIATIONS_EN: frozenset[str] = frozenset({
    "Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Jr", "Sr", "Capt", "Lt",
    "Sgt", "Gen", "Col", "Rev", "Hon", "Inc", "Ltd", "Co", "Corp",
    "vs", "etc", "cf", "al", "No", "Vol", "pp", "Ph", "Mt", "Fr",
    "e.g", "i.e", "a.m", "p.m", "A.M", "P.M", "U.S", "U.K",
    # Academic / professional multi-dot abbreviations
    "Ph.D", "M.D", "B.A", "B.S", "M.A", "M.S", "B.Sc", "M.Sc",
    "B.Tech", "M.Tech", "LL.B", "LL.M", "J.D", "Ed.D",
})
ABBREVIATIONS_DE: frozenset[str] = frozenset({
    "Hr", "Fr", "Dr", "Prof", "Nr", "St", "bzw", "ggf", "inkl",
    "z.B", "u.a", "d.h", "ca", "bzw",
})
ABBREVIATIONS_FR: frozenset[str] = frozenset({
    "M", "Mme", "Mlle", "Dr", "Pr", "St", "Ste", "etc", "cf",
    "p.ex", "c.-à-d", "av", "J.-C",
})
ABBREVIATIONS_ES: frozenset[str] = frozenset({
    "Sr", "Sra", "Srta", "Dr", "Dra", "Ud", "Uds", "etc", "p.ej",
})
ABBREVIATIONS_PT: frozenset[str] = ABBREVIATIONS_ES
ABBREVIATIONS_IT: frozenset[str] = frozenset({
    "Sig", "Sigra", "Dr", "Dott", "Prof", "ecc", "sig", "dott",
})

_EMPTY_ABBR: frozenset[str] = frozenset()

ABBREVIATIONS_BY_LANG: dict[str, frozenset[str]] = {
    "en": ABBREVIATIONS_EN,
    "de": ABBREVIATIONS_DE,
    "fr": ABBREVIATIONS_FR,
    "es": ABBREVIATIONS_ES,
    "pt": ABBREVIATIONS_PT,
    "it": ABBREVIATIONS_IT,
    # Non-Latin scripts: no period-based abbreviations
    "hi": _EMPTY_ABBR, "bn": _EMPTY_ABBR, "mr": _EMPTY_ABBR, "gu": _EMPTY_ABBR,
    "pa": _EMPTY_ABBR, "ta": _EMPTY_ABBR, "te": _EMPTY_ABBR, "kn": _EMPTY_ABBR,
    "ml": _EMPTY_ABBR, "or": _EMPTY_ABBR, "si": _EMPTY_ABBR,
    "zh": _EMPTY_ABBR, "ja": _EMPTY_ABBR, "ko": _EMPTY_ABBR,
    "ar": _EMPTY_ABBR, "fa": _EMPTY_ABBR, "ur": _EMPTY_ABBR, "he": _EMPTY_ABBR,
    "th": _EMPTY_ABBR, "lo": _EMPTY_ABBR, "my": _EMPTY_ABBR, "km": _EMPTY_ABBR,
    "am": _EMPTY_ABBR, "ti": _EMPTY_ABBR, "bo": _EMPTY_ABBR,
    "hy": _EMPTY_ABBR, "ka": _EMPTY_ABBR, "el": _EMPTY_ABBR,
    "ru": ABBREVIATIONS_EN,  # Russian uses many of the same Latin-style abbreviations
    "uk": ABBREVIATIONS_EN,
}

# ---------------------------------------------------------------------------
# Structural patterns that must never be split mid-token.
# ---------------------------------------------------------------------------
# URL — must not gobble the closing paren of a surrounding Markdown link.
# Excludes trailing punctuation that commonly terminates URLs in prose.
URL_REGEX: re.Pattern[str] = re.compile(
    r"\b(?:https?://|www\.|ftp://)[^\s<>\"'`()\[\]]+[^\s<>\"'`()\[\].,;:!?]",
    re.IGNORECASE,
)
EMAIL_REGEX: re.Pattern[str] = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
# v1.2.3, 1.2.3, 1.2.3-rc1, 1.2.3+build
VERSION_REGEX: re.Pattern[str] = re.compile(
    r"\bv?\d+\.\d+(?:\.\d+)+(?:[-+][A-Za-z0-9.]+)?\b"
)
# Posix and Windows paths. Components must not end in punctuation that would
# typically trail a path in prose (period, comma, semicolon, etc.).
PATH_REGEX: re.Pattern[str] = re.compile(
    r"(?:/[^\s/.,;:!?]+(?:/[^\s/.,;:!?]+)*)"
    r"|(?:[A-Za-z]:\\[^\s\\.,;:!?]+(?:\\[^\s\\.,;:!?]+)*)"
)
# 1.5, 3.14 — US-style decimal
DECIMAL_REGEX: re.Pattern[str] = re.compile(r"\d\.\d")
# U.S.A., F.B.I. — letters separated by dots
ACRONYM_REGEX: re.Pattern[str] = re.compile(r"\b(?:[A-Z]\.){2,}(?:[A-Z]\b)?")
# Dotted identifiers: foo.bar.baz, request.body.json. Matches lowercase-first
# tokens only so it does not swallow capitalised abbreviations like `Ph.D` or
# phrases like `Dr. Smith` (letter-dot-letter) that have different meanings.
IDENTIFIER_REGEX: re.Pattern[str] = re.compile(
    r"\b[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*){1,}\b"
)

# ---------------------------------------------------------------------------
# Markdown patterns — used by BasicTextFilter.
# ---------------------------------------------------------------------------
MD_FENCED_CODE_REGEX: re.Pattern[str] = re.compile(
    r"```[^\n]*\n.*?\n?```",
    re.DOTALL,
)
MD_INLINE_CODE_REGEX: re.Pattern[str] = re.compile(r"(?<!`)`([^`\n]+?)`(?!`)")
MD_BOLD_STAR_REGEX: re.Pattern[str] = re.compile(r"\*\*([^*\n]+?)\*\*")
MD_BOLD_UNDER_REGEX: re.Pattern[str] = re.compile(r"__([^_\n]+?)__")
MD_ITALIC_STAR_REGEX: re.Pattern[str] = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
MD_ITALIC_UNDER_REGEX: re.Pattern[str] = re.compile(r"(?<![_\w])_([^_\n]+?)_(?![_\w])")
MD_LINK_REGEX: re.Pattern[str] = re.compile(r"\[([^\]]+)\]\([^)]+\)")
MD_IMAGE_REGEX: re.Pattern[str] = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
MD_HEADING_REGEX: re.Pattern[str] = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
MD_LIST_MARKER_REGEX: re.Pattern[str] = re.compile(
    r"^\s*(?:[-*+]|\d+\.)\s+",
    re.MULTILINE,
)
MD_BLOCKQUOTE_REGEX: re.Pattern[str] = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
MD_TABLE_PIPE_REGEX: re.Pattern[str] = re.compile(r"\s*\|\s*")
MD_TABLE_SEP_REGEX: re.Pattern[str] = re.compile(
    r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?\s*$",
    re.MULTILINE,
)
MD_HR_REGEX: re.Pattern[str] = re.compile(r"^\s*(?:-\s*){3,}$|^\s*(?:\*\s*){3,}$", re.MULTILINE)

# ---------------------------------------------------------------------------
# Symbol expansion (en default).
# ---------------------------------------------------------------------------
# Each entry is (regex, replacement). Applied in order; later rules see the
# output of earlier ones. Chosen conservatively — only substitute when the
# result is unambiguous in spoken English.
SYMBOL_EXPANSIONS_EN: list[tuple[re.Pattern[str], str]] = [
    # "&" between words or bounded by spaces → "and"
    (re.compile(r"(?<=\w)\s*&\s*(?=\w)"), " and "),
    # "50%" → "50 percent"
    (re.compile(r"(\d)\s*%"), r"\1 percent"),
    # "#1", "#42" → "number 1", "number 42"
    (re.compile(r"#(\d)"), r"number \1"),
    # "~50" → "approximately 50"
    (re.compile(r"~(\d)"), r"approximately \1"),
    # "x ≈ y" → "x approximately y"
    (re.compile(r"\s*≈\s*"), " approximately "),
    # "x × y" → "x times y"
    (re.compile(r"(?<=\w)\s*×\s*(?=\w)"), " times "),
    # "x ÷ y" → "x divided by y"
    (re.compile(r"(?<=\w)\s*÷\s*(?=\w)"), " divided by "),
]

# ---------------------------------------------------------------------------
# Punctuation normalisation.
# ---------------------------------------------------------------------------
# Three-or-more dots → single Unicode ellipsis; spaced-hyphen between words →
# em-dash (TTS prosody is better with em-dash than with a lone hyphen).
PUNCT_ELLIPSIS_REGEX: re.Pattern[str] = re.compile(r"\.{3,}")
PUNCT_SPACED_DASH_REGEX: re.Pattern[str] = re.compile(r"(?<=\w)\s+-\s+(?=\w)")

# ---------------------------------------------------------------------------
# LLM metadata leakage — defensive stripping.
# ---------------------------------------------------------------------------
# LLMs sometimes echo internal state bookkeeping into their reply. Observed
# shapes across real sessions:
#
#   (SESSION STATE: intro_delivered=true, language=hinglish)
#   (connection_clarity_retries = 1)
#   (ambiguous_input_detected = true)
#   STATE: demo_booked=false
#
# The unifying signal is the ``=`` sign inside a parenthesised block or on a
# bare KEYWORD-prefixed line. Natural spoken text never uses ``=`` in parens,
# so deleting any ``(… = …)`` is safe for voice output.
METADATA_PARENS_REGEX: re.Pattern[str] = re.compile(
    # Any parenthesised block containing an ``=`` sign. No nested parens, no
    # line breaks inside (to avoid eating multi-paragraph prose). Matches:
    #   "(SESSION STATE: foo=bar)"
    #   "(connection_clarity_retries = 1)"
    #   "(intro_delivered=false, language=hinglish)"
    # Does NOT match natural parentheticals like "(Tax One)", "(व्यापार)", or
    # "(e.g., apples)" — they have no ``=``.
    r"\([^()\n=]*=[^()\n]*\)",
)
METADATA_PREFIX_REGEX: re.Pattern[str] = re.compile(
    # Bare ``KEYWORD: foo=bar`` line (no surrounding parens), e.g.
    # "SESSION STATE: intro_delivered=true". Followed by either a newline or
    # end of string. Keeps the ALL-CAPS restriction here because a
    # lowercase ``key=value`` at the start of a line could appear in natural
    # prose (rare, but possible).
    r"(?:^|\n)\s*[A-Z][A-Z_ ]{2,}:\s*[^\n]*=[^\n]*(?=\n|$)",
)

# ---------------------------------------------------------------------------
# Cartesia-flavoured SSML: <spell> wrapping for digit sequences.
# ---------------------------------------------------------------------------
# Cartesia Sonic-3 supports `<spell>1234</spell>` to force digit-by-digit
# reading (vs. "one thousand two hundred thirty four"). This matters for
# phone numbers, account IDs, OTPs, credit cards, etc. in a voice agent.
#
# Conservative pattern: 7+ consecutive digits OR a phone-number-shaped run
# (groups separated by hyphen/space). Protected tokens (URLs, versions,
# dates) already get placeholders upstream and therefore never match.
SPELL_PHONE_REGEX: re.Pattern[str] = re.compile(
    # An unambiguous phone-number shape. Must have at least ONE of these
    # distinguishing features, otherwise a two-group ``N-M`` pattern is
    # treated as a range by ``RANGE_REGEX_WIDE`` (``500-1000 dollars`` →
    # ``500 to 1000 dollars``):
    #
    #   1. Parenthesised area code:      (555) 123-4567
    #   2. Country-code prefix:          +1 555 123 4567  /  +91 98765 43210
    #   3. Three-or-more digit groups:   555-123-4567  /  555 123 4567
    #
    # Matched together in one alternation so any of the three kicks in.
    r"(?<!\d)(?:"
        # (1) Parens around area code: (555) 123-4567
        r"\(\d{3,4}\)[\s\-]?\d{3,4}(?:[\s\-]?\d{3,5})?"
        r"|"
        # (2) Explicit country code (+, with separator). Allows up to 5-digit
        # groups to cover formats like Indian mobiles: +91 98765 43210
        r"\+\d{1,3}[\s\-]\d{3,5}[\s\-]?\d{3,5}(?:[\s\-]\d{3,5})?"
        r"|"
        # (3) Three digit groups, separators required: 555-123-4567
        r"\d{3,4}[\s\-]\d{3,4}[\s\-]\d{3,5}"
    r")(?!\d)"
)
# Long digit runs (6+) that aren't already wrapped. OTPs are commonly 6
# digits, so we lower the threshold below our previous 7. The negative
# lookbehind/lookahead prevents double-wrapping after the phone regex ran.
SPELL_LONG_DIGITS_REGEX: re.Pattern[str] = re.compile(
    r"(?<!<spell>)(?<!\d)\d{6,}(?!\d)(?!</spell>)"
)

# ---------------------------------------------------------------------------
# Script-mixed parenthetical gloss collapse.
# ---------------------------------------------------------------------------
# When the LLM is given pronunciation rules like "Tally → टैली", it often
# emits both forms side by side: ``Tally (टैली)``. The TTS then reads both,
# producing duplicated audio ("tally tally"). Detect and collapse to just
# the parenthetical gloss.
#
# Trigger: parens that contain at least one non-ASCII character, preceded by
# Latin-script word(s). The number of Latin words removed equals the word
# count inside the parens, so:
#
#   "Tally (टैली)"                   → "टैली"
#   "one click (एक क्लिक)"           → "एक क्लिक"
#   "Extra manpower (मैनपावर)"      → "Extra मैनपावर"
#   "(Tax One)"                      → "(Tax One)"       (no non-ASCII)
#   "(व्यापार)"                      → "(व्यापार)"        (no Latin prefix)
#   "hello (is fine)"                → "hello (is fine)"  (ASCII parens, left alone)
SCRIPT_MIXED_PAREN_REGEX: re.Pattern[str] = re.compile(
    r"\(([^()\n]*[^\x00-\x7f][^()\n]*)\)"
)

# ---------------------------------------------------------------------------
# Numeric range normalisation.
# ---------------------------------------------------------------------------
# Matches short digit-hyphen-digit ranges like ``2-3`` or ``5 - 10`` but not
# phone numbers (``555-1234`` has a 4-digit second group), ISO dates
# (``2026-04-22`` is a 3-segment sequence), versions (protected earlier as
# placeholders), or numbers embedded in longer codes. The lookbehind /
# lookahead for ``[-\d.]`` ensures we're not inside a larger hyphen/dot run.
# Using character-class lookarounds (not ``\b``) so CJK scripts like
# ``5-10分`` are handled correctly — Python's ``\b`` treats CJK chars as
# word characters, which would otherwise prevent the match.
# Narrow 3-digit range regex — the safe default when no SSML spell-wrap has
# pre-protected phone numbers. Catches 2-3, 5-10, 100-200, 50-100, etc.
RANGE_REGEX: re.Pattern[str] = re.compile(
    r"(?<![-\d.])(\d{1,3})\s*-\s*(\d{1,3})(?![-\d.])"
)

# Wider 4-digit range regex — safe when phones are already wrapped in
# ``<spell>…</spell>`` tags ahead of this pass. Tag-aware negative
# lookbehind/lookahead (``>`` / ``<``) prevents matching digits inside
# existing SSML wrappers. Catches 500-1000, 2000-5000, etc.
RANGE_REGEX_WIDE: re.Pattern[str] = re.compile(
    r"(?<![->\d.])(\d{1,4})\s*-\s*(\d{1,4})(?![-<\d.])"
)

# Separator word used when expanding ``N-M`` into ``N <sep> M``. Picked by the
# resolved language; ``None`` (default) means "don't expand".
RANGE_SEPARATOR_BY_LANG: dict[str, str] = {
    "en": "to",
    "de": "bis",
    "fr": "à",
    "es": "a",
    "pt": "a",
    "it": "a",
    # Indic / Indo-Aryan
    "hi": "से",
    "bn": "থেকে",
    "mr": "ते",
    "gu": "થી",
    "pa": "ਤੋਂ",
    "ta": "முதல்",
    "te": "నుండి",
    "kn": "ರಿಂದ",
    "ml": "മുതൽ",
    # East Asian
    "zh": "到",
    "ja": "から",
    "ko": "에서",
    # Semitic / Middle Eastern
    "ar": "إلى",
    "fa": "تا",
    "ur": "سے",
    "he": "עד",
    # South-East Asian
    "th": "ถึง",
    # Slavic / Cyrillic
    "ru": "до",
    "uk": "до",
    # Greek / others
    "el": "έως",
}

# ---------------------------------------------------------------------------
# Placeholder code points for structural protection (PUA, unassigned).
# ---------------------------------------------------------------------------
# Unicode guarantees U+E000..U+F8FF are never assigned to a character, so we
# can safely use them as single-character placeholders without risk of
# colliding with real content.
PLACEHOLDER_BASE: int = 0xE000
PLACEHOLDER_MAX: int = 0xF8FF


# ---------------------------------------------------------------------------
# Script detection.
# ---------------------------------------------------------------------------
# Map a sample of text to an ISO 639-1 hint. Returns "en" as a default for
# Latin-script text; callers can override with an explicit language kwarg.

_SCRIPT_RANGES: list[tuple[str, re.Pattern[str]]] = [
    ("hi", re.compile(r"[ऀ-ॿ]")),       # Devanagari
    ("bn", re.compile(r"[ঀ-৿]")),       # Bengali
    ("pa", re.compile(r"[਀-੿]")),       # Gurmukhi
    ("gu", re.compile(r"[઀-૿]")),       # Gujarati
    ("or", re.compile(r"[଀-୿]")),       # Odia
    ("ta", re.compile(r"[஀-௿]")),       # Tamil
    ("te", re.compile(r"[ఀ-౿]")),       # Telugu
    ("kn", re.compile(r"[ಀ-೿]")),       # Kannada
    ("ml", re.compile(r"[ഀ-ൿ]")),       # Malayalam
    ("si", re.compile(r"[඀-෿]")),       # Sinhala
    ("th", re.compile(r"[฀-๿]")),       # Thai
    ("lo", re.compile(r"[຀-໿]")),       # Lao
    ("bo", re.compile(r"[ༀ-࿿]")),       # Tibetan
    ("my", re.compile(r"[က-႟]")),       # Myanmar
    ("km", re.compile(r"[ក-៿]")),       # Khmer
    ("am", re.compile(r"[ሀ-፿]")),       # Ethiopic
    ("hy", re.compile(r"[԰-֏]")),       # Armenian
    ("ka", re.compile(r"[Ⴀ-ჿ]")),       # Georgian
    ("el", re.compile(r"[Ͱ-Ͽ]")),       # Greek
    ("he", re.compile(r"[֐-׿]")),       # Hebrew
    ("ar", re.compile(r"[؀-ۿ]")),       # Arabic
    ("ru", re.compile(r"[Ѐ-ӿ]")),       # Cyrillic (default Russian)
    ("zh", re.compile(r"[一-鿿㐀-䶿]")),  # CJK Unified
    ("ja", re.compile(r"[぀-ヿ]")),       # Hiragana/Katakana
    ("ko", re.compile(r"[가-힯]")),       # Hangul
]


def detect_script(sample: str, *, default: str = "en") -> str:
    """Return an ISO 639-1 hint based on the first non-Latin script found.

    Looks at the sample string for characters from each supported script in
    priority order. Falls back to ``default`` when only Latin characters
    (or whitespace / digits / punctuation) are present.

    Args:
        sample: Text to inspect. The first ~200 characters are sufficient.
        default: Language code returned when no non-Latin script is detected.

    Returns:
        An ISO 639-1 code. The hint is used for abbreviation selection and
        locale-specific disambiguation (e.g. Greek ``;`` handling); it does
        not gate which strong terminators are recognised.
    """
    if not sample:
        return default
    # Prefer the longest-continuous non-Latin run
    scores: dict[str, int] = {}
    for code, pattern in _SCRIPT_RANGES:
        matches = pattern.findall(sample)
        if matches:
            scores[code] = len(matches)
    if not scores:
        return default
    # For CJK, disambiguate Chinese vs Japanese vs Korean:
    # if Hiragana/Katakana present, it's Japanese regardless of Han count.
    if "ja" in scores:
        return "ja"
    if "ko" in scores:
        return "ko"
    return max(scores.items(), key=lambda kv: kv[1])[0]
