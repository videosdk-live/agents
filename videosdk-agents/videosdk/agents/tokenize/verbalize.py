from __future__ import annotations

import re
from typing import Callable

_EN_ONES: tuple[str, ...] = (
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
)
_EN_TEENS: tuple[str, ...] = (
    "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
)
_EN_TENS: tuple[str, ...] = (
    "", "", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety",
)
_EN_SCALES: tuple[str, ...] = (
    "", "thousand", "million", "billion", "trillion",
)


def _en_under_1000(n: int) -> str:
    """Return English words for 0..999 (empty string for 0)."""
    if n == 0:
        return ""
    parts: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        parts.append(f"{_EN_ONES[hundreds]} hundred")
    if rest:
        if rest < 10:
            parts.append(_EN_ONES[rest])
        elif rest < 20:
            parts.append(_EN_TEENS[rest - 10])
        else:
            tens, ones = divmod(rest, 10)
            if ones:
                parts.append(f"{_EN_TENS[tens]}-{_EN_ONES[ones]}")
            else:
                parts.append(_EN_TENS[tens])
    return " ".join(parts)


def verbalize_en(n: int) -> str:
    """Return English cardinal words for an integer."""
    if n == 0:
        return "zero"
    sign = ""
    if n < 0:
        sign = "minus "
        n = -n
    groups: list[int] = []
    while n > 0:
        n, r = divmod(n, 1000)
        groups.append(r)
    parts: list[str] = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        words = _en_under_1000(g)
        if i < len(_EN_SCALES) and _EN_SCALES[i]:
            parts.append(f"{words} {_EN_SCALES[i]}")
        else:
            parts.append(words)
    return sign + " ".join(parts)


_HI_0_99: tuple[str, ...] = (
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चौंतालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरेपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरेसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उन्यासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी",
    "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
)

_HI_HUNDRED: str = "सौ"
_HI_THOUSAND: str = "हज़ार"
_HI_LAKH: str = "लाख"
_HI_CRORE: str = "करोड़"
_HI_ARAB: str = "अरब"
_HI_POINT: str = "दशमलव"


_MR_0_99: tuple[str, ...] = (
    "शून्य", "एक", "दोन", "तीन", "चार", "पाच", "सहा", "सात", "आठ", "नऊ",
    "दहा", "अकरा", "बारा", "तेरा", "चौदा", "पंधरा", "सोळा", "सतरा", "अठरा", "एकोणीस",
    "वीस", "एकवीस", "बावीस", "तेवीस", "चोवीस", "पंचवीस", "सव्वीस", "सत्तावीस", "अठ्ठावीस", "एकोणतीस",
    "तीस", "एकतीस", "बत्तीस", "तेहतीस", "चौतीस", "पस्तीस", "छत्तीस", "सदतीस", "अडतीस", "एकोणचाळीस",
    "चाळीस", "एक्केचाळीस", "बेचाळीस", "त्रेचाळीस", "चव्वेचाळीस", "पंचेचाळीस", "सेहेचाळीस", "सत्तेचाळीस", "अठ्ठेचाळीस", "एकोणपन्नास",
    "पन्नास", "एक्कावन्न", "बावन्न", "त्रेपन्न", "चोपन्न", "पंचावन्न", "छप्पन्न", "सत्तावन्न", "अठ्ठावन्न", "एकोणसाठ",
    "साठ", "एकसष्ट", "बासष्ट", "त्रेसष्ट", "चौसष्ट", "पासष्ट", "सहासष्ट", "सदुसष्ट", "अडुसष्ट", "एकोणसत्तर",
    "सत्तर", "एक्काहत्तर", "बाहत्तर", "त्र्याहत्तर", "चौऱ्याहत्तर", "पंच्याहत्तर", "शहात्तर", "सत्त्याहत्तर", "अठ्ठ्याहत्तर", "एकोणऐंशी",
    "ऐंशी", "एक्क्याऐंशी", "ब्याऐंशी", "त्र्याऐंशी", "चौऱ्याऐंशी", "पंच्याऐंशी", "शहाऐंशी", "सत्त्याऐंशी", "अठ्ठ्याऐंशी", "एकोणनव्वद",
    "नव्वद", "एक्क्याण्णव", "ब्याण्णव", "त्र्याण्णव", "चौऱ्याण्णव", "पंच्याण्णव", "शहाण्णव", "सत्त्याण्णव", "अठ्ठ्याण्णव", "नव्व्याण्णव",
)

_MR_HUNDRED: str = "शंभर"
_MR_THOUSAND: str = "हजार"
_MR_LAKH: str = "लाख"
_MR_CRORE: str = "कोटी"
_MR_ARAB: str = "अब्ज"
_MR_POINT: str = "दशांश"


_GU_0_99: tuple[str, ...] = (
    "શૂન્ય", "એક", "બે", "ત્રણ", "ચાર", "પાંચ", "છ", "સાત", "આઠ", "નવ",
    "દસ", "અગિયાર", "બાર", "તેર", "ચૌદ", "પંદર", "સોળ", "સત્તર", "અઢાર", "ઓગણીસ",
    "વીસ", "એકવીસ", "બાવીસ", "ત્રેવીસ", "ચોવીસ", "પચીસ", "છવ્વીસ", "સત્તાવીસ", "અઠ્ઠાવીસ", "ઓગણત્રીસ",
    "ત્રીસ", "એકત્રીસ", "બત્રીસ", "તેત્રીસ", "ચોત્રીસ", "પાંત્રીસ", "છત્રીસ", "સાડત્રીસ", "અડત્રીસ", "ઓગણચાળીસ",
    "ચાળીસ", "એકતાળીસ", "બેતાળીસ", "ત્રેતાળીસ", "ચુમ્માળીસ", "પિસ્તાળીસ", "છેતાળીસ", "સુડતાળીસ", "અડતાળીસ", "ઓગણપચાસ",
    "પચાસ", "એકાવન", "બાવન", "ત્રેપન", "ચોપન", "પંચાવન", "છપ્પન", "સત્તાવન", "અઠ્ઠાવન", "ઓગણસાઠ",
    "સાઠ", "એકસઠ", "બાસઠ", "ત્રેસઠ", "ચોસઠ", "પાંસઠ", "છાસઠ", "સડસઠ", "અડસઠ", "ઓગણસિત્તેર",
    "સિત્તેર", "એકોતેર", "બોતેર", "ત્રોતેર", "ચુમોતેર", "પંચોતેર", "છોતેર", "સિત્યોતેર", "ઇઠ્ઠોતેર", "ઓગણ્યાએંસી",
    "એંસી", "એક્યાસી", "બ્યાસી", "ત્યાસી", "ચોર્યાસી", "પંચાસી", "છ્યાસી", "સત્યાસી", "ઈઠ્યાસી", "નેવ્યાસી",
    "નેવું", "એકાણું", "બાણું", "ત્રાણું", "ચોરાણું", "પંચાણું", "છન્નું", "સત્તાણું", "અઠ્ઠાણું", "નવ્વાણું",
)

_GU_HUNDRED: str = "સો"
_GU_THOUSAND: str = "હજાર"
_GU_LAKH: str = "લાખ"
_GU_CRORE: str = "કરોડ"
_GU_ARAB: str = "અબજ"
_GU_POINT: str = "દશાંશ"


def _indic_verbalize(
    n: int,
    *,
    zero_to_99: tuple[str, ...],
    hundred: str,
    thousand: str,
    lakh: str,
    crore: str,
    arab: str,
    neg_prefix: str,
) -> str:
    """Generic Indian-grouping verbalizer parameterised by per-language tables.

    All Indic languages share the same grouping (हज़ार / लाख / करोड़ / अरब),
    only the words and 0-99 table differ — so one implementation fits Hindi,
    Marathi, Gujarati, Bengali, Tamil, etc. once the tables are supplied.
    """
    if n == 0:
        return zero_to_99[0]
    sign = ""
    if n < 0:
        sign = neg_prefix + " "
        n = -n
    parts: list[str] = []

    if n >= 1_00_00_00_000:
        q, n = divmod(n, 1_00_00_00_000)
        if q <= 99:
            parts.append(f"{zero_to_99[q]} {arab}")
        else:
            parts.append(f"{_indic_verbalize(q, zero_to_99=zero_to_99, hundred=hundred, thousand=thousand, lakh=lakh, crore=crore, arab=arab, neg_prefix=neg_prefix)} {arab}")
    if n >= 1_00_00_000:  
        q, n = divmod(n, 1_00_00_000)
        parts.append(f"{zero_to_99[q]} {crore}")
    if n >= 1_00_000: 
        q, n = divmod(n, 1_00_000)
        parts.append(f"{zero_to_99[q]} {lakh}")
    if n >= 1_000: 
        q, n = divmod(n, 1_000)
        parts.append(f"{zero_to_99[q]} {thousand}")
    if n >= 100:
        q, n = divmod(n, 100)
        parts.append(f"{zero_to_99[q]} {hundred}")
    if n > 0:
        parts.append(zero_to_99[n])
    return sign + " ".join(parts)


def verbalize_hi(n: int) -> str:
    """Hindi cardinal words via Indian grouping (हज़ार / लाख / करोड़)."""
    return _indic_verbalize(
        n,
        zero_to_99=_HI_0_99,
        hundred=_HI_HUNDRED, thousand=_HI_THOUSAND,
        lakh=_HI_LAKH, crore=_HI_CRORE, arab=_HI_ARAB,
        neg_prefix="ऋण",
    )


def verbalize_mr(n: int) -> str:
    """Marathi cardinal words via Indian grouping (हजार / लाख / कोटी)."""
    return _indic_verbalize(
        n,
        zero_to_99=_MR_0_99,
        hundred=_MR_HUNDRED, thousand=_MR_THOUSAND,
        lakh=_MR_LAKH, crore=_MR_CRORE, arab=_MR_ARAB,
        neg_prefix="उणे",
    )


def verbalize_gu(n: int) -> str:
    """Gujarati cardinal words via Indian grouping (હજાર / લાખ / કરોડ)."""
    return _indic_verbalize(
        n,
        zero_to_99=_GU_0_99,
        hundred=_GU_HUNDRED, thousand=_GU_THOUSAND,
        lakh=_GU_LAKH, crore=_GU_CRORE, arab=_GU_ARAB,
        neg_prefix="ઓછું",
    )

_INDIC_LANGS: frozenset[str] = frozenset({
    "hi", "bn", "mr", "gu", "pa", "or", "as",
    "ta", "te", "kn", "ml", "si", "ne", "ur",
})


def _pick_verbalizer(lang: str) -> Callable[[int], str]:
    """Return the integer-to-words function for a language.

    Native tables exist for Hindi, Marathi, and Gujarati. Other Indic
    languages (bn, ta, te, kn, ml, pa, or, …) fall back to Hindi words
    in Devanagari script — acceptable for Cartesia Sonic-3 Devanagari
    models but NOT for languages using a distinct script (Gujarati has
    its own table precisely because Cartesia ``language="gu"`` expects
    Gujarati script).
    """
    code = (lang or "en").lower().split("-")[0]
    if code == "mr":
        return verbalize_mr
    if code == "gu":
        return verbalize_gu
    if code == "hi" or code in _INDIC_LANGS:
        return verbalize_hi
    return verbalize_en


def _point_word(lang: str) -> str:
    code = (lang or "en").lower().split("-")[0]
    if code == "mr":
        return _MR_POINT
    if code == "gu":
        return _GU_POINT
    if code == "hi" or code in _INDIC_LANGS:
        return _HI_POINT
    return "point"


def _digit_words(digits: str, lang: str) -> str:
    """Convert a run of digits into space-separated digit words."""
    verb = _pick_verbalizer(lang)
    return " ".join(verb(int(d)) for d in digits)


def verbalize_decimal(int_part: str, frac_part: str, lang: str) -> str:
    """Verbalize a decimal like ``3.5`` → ``three point five``.

    The fractional part is read digit-by-digit (so ``3.14`` → ``three point
    one four``) which is the natural spoken form for non-money decimals.
    """
    int_val = int(int_part.replace(",", "")) if int_part else 0
    verb = _pick_verbalizer(lang)
    head = verb(int_val)
    if frac_part:
        return f"{head} {_point_word(lang)} {_digit_words(frac_part, lang)}"
    return head

_CURRENCY_TABLE: tuple[tuple[str, str, str, str, str], ...] = (
    ("$",  "dollars", "डॉलर", "डॉलर",   "ડૉલર"),
    ("₹",  "rupees",  "रुपये", "रुपये",  "રૂપિયા"),
    ("€",  "euros",   "यूरो",  "यूरो",   "યુરો"),
    ("£",  "pounds",  "पाउंड", "पाउंड",  "પાઉન્ડ"),
    ("¥",  "yen",     "येन",   "येन",    "યેન"),
    ("₽",  "rubles",  "रूबल",  "रूबल",   "રૂબલ"),
    ("₩",  "won",     "वॉन",   "वॉन",    "વોન"),
)


_HINT_TO_WORDS: dict[str, tuple[str, str, str, str]] = {
    "usd": ("dollars", "डॉलर",  "डॉलर",   "ડૉલર"),
    "inr": ("rupees",  "रुपये", "रुपये",  "રૂપિયા"),
    "eur": ("euros",   "यूरो",  "यूरो",   "યુરો"),
    "gbp": ("pounds",  "पाउंड", "पाउंड",  "પાઉન્ડ"),
    "jpy": ("yen",     "येन",   "येन",    "યેન"),
    "cad": ("Canadian dollars", "कनाडाई डॉलर", "कॅनेडियन डॉलर", "કેનેડિયન ડૉલર"),
    "aud": ("Australian dollars", "ऑस्ट्रेलियाई डॉलर", "ऑस्ट्रेलियन डॉलर", "ઑસ્ટ્રેલિયન ડૉલર"),
}


_SCALE_MULT: dict[str, int] = {
    # Latin abbreviations
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
    "t": 1_000_000_000_000,
    "l": 1_00_000,         
    "cr": 1_00_00_000,    
    "lakh": 1_00_000,
    "crore": 1_00_00_000,
    "lac": 1_00_000,

    "हज़ार": 1_000,
    "हजार": 1_000,
    "लाख": 1_00_000,
    "करोड़": 1_00_00_000,
    "कोटी": 1_00_00_000,  
    "अरब": 1_00_00_00_000,
    "अब्ज": 1_00_00_00_000,  
    
    "હજાર": 1_000,
    "લાખ": 1_00_000,
    "કરોડ": 1_00_00_000,
    "અબજ": 1_00_00_00_000,
}

_SUFFIX_ALT: str = (
    r"(?:"
    r"Lakh|Crore|Lac|Cr|K|M|B|T|L"                         
    r"|लाख|करोड़|कोटी|हज़ार|हजार|अरब|अब्ज"                     
    r"|લાખ|કરોડ|હજાર|અબજ"                                 
    r")(?![A-Za-z0-9ऀ-ॿ઀-૿])"
)

_CURRENCY_AMOUNT_RE: re.Pattern[str] = re.compile(
    r"([$₹€£¥₽₩])\s*"
    r"(\d{1,3}(?:[, ]\d{2,3})*|\d+)"
    r"(?:\.(\d+))?"
    r"(?:\s*(" + _SUFFIX_ALT + r"))?",
    re.IGNORECASE,
)

_CARDINAL_RE: re.Pattern[str] = re.compile(
    r"(?<!\d)(?<!\.)"
    r"("
        r"\d{1,3}(?:,\d{2,3})+"
        r"|\d{3,}"
        r"|\d{1,2}(?=\.\d)"
        r"|\d{1,2}(?=\s*" + _SUFFIX_ALT + r")"
    r")"
    r"(?:\.(\d+))?"
    r"(?:\s*(" + _SUFFIX_ALT + r"))?"
    r"(?!\d)",
    re.IGNORECASE,
)


def _resolve_currency_word(symbol: str, hint: str | None, lang: str) -> str:
    """Return the spoken currency word for (symbol, hint, language)."""
    code = (lang or "en").lower().split("-")[0]
    if code == "mr":
        idx = 2
    elif code == "gu":
        idx = 3
    elif code == "hi" or code in _INDIC_LANGS:
        idx = 1
    else:
        idx = 0

    if hint:
        row = _HINT_TO_WORDS.get(hint.lower())
        if row:
            return row[idx]
    for entry in _CURRENCY_TABLE:
        if entry[0] == symbol:
            return entry[idx + 1]  
    return "" 


def _expand_scale(n_str: str, frac: str | None, suffix: str | None) -> tuple[int, str | None]:
    """Combine integer+fraction+suffix into a numeric value + leftover words.

    Returns (numeric_value, scale_phrase). When suffix is K/M/B/T, the
    suffix is multiplied into the value and ``scale_phrase`` is None.
    When suffix is L/Cr/Lakh/Crore we preserve the word (so Hindi output
    reads ``दस लाख`` rather than ``दस लाख लाख``).
    """
    int_val = int(n_str.replace(",", "").replace(" ", ""))
    if not suffix:
        if frac:
            return int_val, None
        return int_val, None
    key = suffix.lower()
    mult = _SCALE_MULT.get(key, 1)
    if frac:
        whole = int(f"{int_val}{frac}")
        scale = mult // (10 ** len(frac))
        return whole * scale, None
    return int_val * mult, None


def expand_currency(text: str, *, language: str, hint: str | None) -> str:
    """Replace ``$500,000`` / ``₹10,00,000`` with spoken form.

    All-zero fractional parts (``₹50,00,000.00``, ``$1,000.00``) are dropped:
    voice agents should not read "fifty lakh point zero zero rupees" when the
    intent is just "fifty lakh rupees". LLMs (especially Gemini) routinely
    append ``.00`` to round rupee/dollar amounts — treat it as display
    formatting, not spoken content.
    """
    lang = (language or "en").lower()
    verb = _pick_verbalizer(lang)

    def _sub(m: re.Match[str]) -> str:
        symbol, int_part, frac, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
        currency_word = _resolve_currency_word(symbol, hint, lang)
        if frac and all(d == "0" for d in frac):
            frac = None
        if frac and not suffix:
            number_words = verbalize_decimal(int_part, frac, lang)
        else:
            value, _ = _expand_scale(int_part, frac, suffix)
            number_words = verb(value)
        if currency_word:
            return f"{number_words} {currency_word}"
        return number_words

    return _apply_outside_spell(text, lambda t: _CURRENCY_AMOUNT_RE.sub(_sub, t))


def expand_cardinals(text: str, *, language: str) -> str:
    """Verbalize standalone cardinal numbers (≥100 or comma-grouped).

    Skips 4-digit years (1700–2099) and numbers inside ``<spell>…</spell>``.
    Decimals are read as "integer point digit-digit".
    """
    lang = (language or "en").lower()
    verb = _pick_verbalizer(lang)

    def _sub(m: re.Match[str]) -> str:
        int_part, frac, suffix = m.group(1), m.group(2), m.group(3)
        digits_only = int_part.replace(",", "")
        if (
            not frac
            and not suffix
            and "," not in int_part
            and len(digits_only) == 4
            and 1700 <= int(digits_only) <= 2099
        ):
            return m.group(0)
        if frac and all(d == "0" for d in frac):
            frac = None
        if frac and not suffix:
            return verbalize_decimal(int_part, frac, lang)
        value, _ = _expand_scale(int_part, frac, suffix)
        return verb(value)

    return _apply_outside_spell(text, lambda t: _CARDINAL_RE.sub(_sub, t))


_SPELL_SPLIT_RE: re.Pattern[str] = re.compile(r"(<spell>.*?</spell>)", re.DOTALL)


def _apply_outside_spell(text: str, fn: Callable[[str], str]) -> str:
    """Apply ``fn`` to every substring that is NOT inside a ``<spell>`` tag."""
    if "<spell>" not in text:
        return fn(text)
    parts = _SPELL_SPLIT_RE.split(text)
    return "".join(p if p.startswith("<spell>") else fn(p) for p in parts)
