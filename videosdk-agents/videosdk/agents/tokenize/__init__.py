"""Text tokenization and filtering for the cascade pipeline.

The public surface exposes the abstract interfaces and the default
implementations. External tokenizer plugins (e.g. a future
``videosdk-plugins-blingfire``) can plug into ``SentenceTokenizer`` without
touching core code.
"""

from .base import SentenceStream, SentenceTokenizer, TextFilter
from .basic import BasicSentenceTokenizer
from .filters import BasicTextFilter
from .patterns import (
    ABBREVIATIONS_BY_LANG,
    STRONG_TERMINATORS,
    WEAK_TERMINATORS,
    detect_script,
)
from .stream import BufferedSentenceStream

__all__ = [
    "ABBREVIATIONS_BY_LANG",
    "BasicSentenceTokenizer",
    "BasicTextFilter",
    "BufferedSentenceStream",
    "STRONG_TERMINATORS",
    "SentenceStream",
    "SentenceTokenizer",
    "TextFilter",
    "WEAK_TERMINATORS",
    "detect_script",
]
