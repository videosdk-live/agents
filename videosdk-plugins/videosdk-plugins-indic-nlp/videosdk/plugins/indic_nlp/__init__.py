"""VideoSDK tokenizer plugin backed by the indic-nlp-library.

Provides a drop-in ``SentenceTokenizer`` implementation for Indic languages
(Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati,
Punjabi, Odia, Assamese, Sinhala, Urdu) with honorific-aware sentence
boundaries that the core ``BasicSentenceTokenizer`` doesn't have.

Optionally wraps ``UnicodeIndicTransliterator`` for cross-script conversion,
e.g. a Telugu TTS speaking text authored in Hindi.
"""

from .tokenizer import (
    IndicSentenceTokenizer,
    IndicScriptTransliterator,
    SUPPORTED_LANGUAGES,
    pre_warm_tokenizer,
)
from .version import __version__

__all__ = [
    "IndicSentenceTokenizer",
    "IndicScriptTransliterator",
    "SUPPORTED_LANGUAGES",
    "pre_warm_tokenizer",
    "__version__",
]
