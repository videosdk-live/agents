from __future__ import annotations

import re

STRONG_TERMINATORS: str = (
    ".!?"
    "…"          
    "‼⁇⁈⁉" 
    "。！？"
    "।॥"
    "؟"          
    "۔"           
    "؞"        
    "։"
    "።፧፨"
    "།༎༏༐༑"
    "။"
    "។៕"
    "\n"
)

WEAK_TERMINATORS: str = (
    ",;:"
    "、，；："
    "،؛"
    "՝՛"
    "፣፤፥"
    "་"
    "၊"
    "៖"
    "—–"
)

CLOSING_QUOTES: str = "\"'”’»"

NO_SPACE_SCRIPTS_REGEX: re.Pattern[str] = re.compile(
    "["
    "一-鿿"      
    "㐀-䶿"    
    "豈-﫿"     
    "぀-ゟ"      
    "゠-ヿ"      
    "가-힯"      
    "฀-๿"      
    "຀-໿"     
    "က-႟"     
    "ក-៿"     
    "]"
)

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

ABBREVIATIONS_HI: frozenset[str] = frozenset({
    "डॉ", "श्री", "श्रीमती", "कु", "प्रो", "पं", "डा",
})
ABBREVIATIONS_BN: frozenset[str] = frozenset({"ড", "শ্রী"})
ABBREVIATIONS_MR: frozenset[str] = ABBREVIATIONS_HI
ABBREVIATIONS_GU: frozenset[str] = frozenset({"ડૉ", "શ્રી", "પ્રો"})

_EMPTY_ABBR: frozenset[str] = frozenset()

ABBREVIATIONS_BY_LANG: dict[str, frozenset[str]] = {
    "en": ABBREVIATIONS_EN,
    "de": ABBREVIATIONS_DE,
    "fr": ABBREVIATIONS_FR,
    "es": ABBREVIATIONS_ES,
    "pt": ABBREVIATIONS_PT,
    "it": ABBREVIATIONS_IT,
    "hi": ABBREVIATIONS_HI,
    "bn": ABBREVIATIONS_BN,
    "mr": ABBREVIATIONS_MR,
    "gu": ABBREVIATIONS_GU,
    "pa": _EMPTY_ABBR, "ta": _EMPTY_ABBR, "te": _EMPTY_ABBR, "kn": _EMPTY_ABBR,
    "ml": _EMPTY_ABBR, "or": _EMPTY_ABBR, "si": _EMPTY_ABBR,
    "zh": _EMPTY_ABBR, "ja": _EMPTY_ABBR, "ko": _EMPTY_ABBR,
    "ar": _EMPTY_ABBR, "fa": _EMPTY_ABBR, "ur": _EMPTY_ABBR, "he": _EMPTY_ABBR,
    "th": _EMPTY_ABBR, "lo": _EMPTY_ABBR, "my": _EMPTY_ABBR, "km": _EMPTY_ABBR,
    "am": _EMPTY_ABBR, "ti": _EMPTY_ABBR, "bo": _EMPTY_ABBR,
    "hy": _EMPTY_ABBR, "ka": _EMPTY_ABBR, "el": _EMPTY_ABBR,
    "ru": ABBREVIATIONS_EN, 
    "uk": ABBREVIATIONS_EN,
}

URL_REGEX: re.Pattern[str] = re.compile(
    r"\b(?:https?://|www\.|ftp://)[^\s<>\"'`()\[\]]+[^\s<>\"'`()\[\].,;:!?]",
    re.IGNORECASE,
)
EMAIL_REGEX: re.Pattern[str] = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
VERSION_REGEX: re.Pattern[str] = re.compile(
    r"\bv?\d+\.\d+(?:\.\d+)+(?:[-+][A-Za-z0-9.]+)?\b"
)
PATH_REGEX: re.Pattern[str] = re.compile(
    r"(?:/[^\s/.,;:!?]+(?:/[^\s/.,;:!?]+)*)"
    r"|(?:[A-Za-z]:\\[^\s\\.,;:!?]+(?:\\[^\s\\.,;:!?]+)*)"
)
DECIMAL_REGEX: re.Pattern[str] = re.compile(r"\d\.\d")
ACRONYM_REGEX: re.Pattern[str] = re.compile(r"\b(?:[A-Z]\.){2,}(?:[A-Z]\b)?")
IDENTIFIER_REGEX: re.Pattern[str] = re.compile(
    r"\b[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*){1,}\b"
)

NUMBER_GROUPING_REGEX: re.Pattern[str] = re.compile(r"\d{1,3}(?:,\d{2,3})+")

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

SYMBOL_EXPANSIONS_EN: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<=\w)\s*&\s*(?=\w)"), " and "),
    (re.compile(r"(\d)\s*%"), r"\1 percent"),
    (re.compile(r"#(\d)"), r"number \1"),
    (re.compile(r"~(\d)"), r"approximately \1"),
    (re.compile(r"\s*≈\s*"), " approximately "),
    (re.compile(r"(?<=\w)\s*×\s*(?=\w)"), " times "),
    (re.compile(r"(?<=\w)\s*÷\s*(?=\w)"), " divided by "),
]


PUNCT_ELLIPSIS_REGEX: re.Pattern[str] = re.compile(r"\.{3,}")
PUNCT_SPACED_DASH_REGEX: re.Pattern[str] = re.compile(r"(?<=\w)\s+-\s+(?=\w)")

METADATA_PARENS_REGEX: re.Pattern[str] = re.compile(
    r"\([^()\n=]*=[^()\n]*\)",
)
METADATA_PREFIX_REGEX: re.Pattern[str] = re.compile(
    r"(?:^|\n)\s*[A-Z][A-Z_ ]{2,}:\s*[^\n]*=[^\n]*(?=\n|$)",
)

SPELL_PHONE_REGEX: re.Pattern[str] = re.compile(
    r"(?<!\d)(?:"
        r"\(\d{3,4}\)[\s\-]?\d{3,4}(?:[\s\-]?\d{3,5})?"
        r"|"
        r"\+\d{1,3}[\s\-]\d{3,5}[\s\-]?\d{3,5}(?:[\s\-]\d{3,5})?"
        r"|"
        r"\d{3,4}[\s\-]\d{3,4}[\s\-]\d{3,5}"
    r")(?!\d)"
)

SPELL_LONG_DIGITS_REGEX: re.Pattern[str] = re.compile(
    r"(?<!<spell>)(?<!\d)\d{6,}(?!\d)(?!</spell>)"
)

SCRIPT_MIXED_PAREN_REGEX: re.Pattern[str] = re.compile(
    r"\(([^()\n]*[^\x00-\x7f][^()\n]*)\)"
)

RANGE_REGEX: re.Pattern[str] = re.compile(
    r"(?<![-\d.])(\d{1,3})\s*-\s*(\d{1,3})(?![-\d.])"
)

RANGE_REGEX_WIDE: re.Pattern[str] = re.compile(
    r"(?<![->\d.])(\d{1,4})\s*-\s*(\d{1,4})(?![-<\d.])"
)

RANGE_SEPARATOR_BY_LANG: dict[str, str] = {
    "en": "to",
    "de": "bis",
    "fr": "à",
    "es": "a",
    "pt": "a",
    "it": "a",
    "hi": "से",
    "bn": "থেকে",
    "mr": "ते",
    "gu": "થી",
    "pa": "ਤੋਂ",
    "ta": "முதல்",
    "te": "నుండి",
    "kn": "ರಿಂದ",
    "ml": "മുതൽ",
    "zh": "到",
    "ja": "から",
    "ko": "에서",
    "ar": "إلى",
    "fa": "تا",
    "ur": "سے",
    "he": "עד",
    "th": "ถึง",
    "ru": "до",
    "uk": "до",
    "el": "έως",
}

PLACEHOLDER_BASE: int = 0xE000
PLACEHOLDER_MAX: int = 0xF8FF


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
    scores: dict[str, int] = {}
    for code, pattern in _SCRIPT_RANGES:
        matches = pattern.findall(sample)
        if matches:
            scores[code] = len(matches)
    if not scores:
        return default

    if "ja" in scores:
        return "ja"
    if "ko" in scores:
        return "ko"
    return max(scores.items(), key=lambda kv: kv[1])[0]


def normalize_lang_code(code: str | None) -> str | None:
    """Reduce an STT/TTS language tag to a bare ISO 639-1 code, or ``None``.
    """
    if not code:
        return None
    base = code.split("-", 1)[0].split("_", 1)[0].strip().lower()
    if not base or base == "auto":
        return None
    return base
