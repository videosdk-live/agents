"""Text chunking and filtering for the cascade pipeline.

The public surface exposes the abstract interfaces and the default
implementations. External chunker plugins can plug into ``SentenceChunker``
without touching core code.

Naming note: these classes are intentionally distinct from other frameworks'
``*Tokenizer`` types — VideoSDK chunks LLM output into sentence-sized
segments for streaming TTS, hence the ``Chunker`` family.
"""

from .base import SentenceChunkStream, SentenceChunker, TextFilter
from .basic import BasicSentenceChunker
from .filters import BasicTextFilter
from .hyphenate import EnglishHyphenator, hyphenate_english
from .indic import (
    INDIC_LANGS,
    IndicScriptTransliterator,
    IndicSentenceChunker,
    SUPPORTED_LANGUAGES,
    pre_warm_tokenizer,
)
from .patterns import (
    ABBREVIATIONS_BY_LANG,
    STRONG_TERMINATORS,
    WEAK_TERMINATORS,
    detect_script,
    normalize_lang_code,
)
from .stream import BufferedSentenceChunkStream

__all__ = [
    "ABBREVIATIONS_BY_LANG",
    "BasicSentenceChunker",
    "BasicTextFilter",
    "BufferedSentenceChunkStream",
    "EnglishHyphenator",
    "INDIC_LANGS",
    "IndicScriptTransliterator",
    "IndicSentenceChunker",
    "STRONG_TERMINATORS",
    "SUPPORTED_LANGUAGES",
    "SentenceChunkStream",
    "SentenceChunker",
    "TextFilter",
    "WEAK_TERMINATORS",
    "detect_script",
    "hyphenate_english",
    "normalize_lang_code",
    "pre_warm_tokenizer",
]
