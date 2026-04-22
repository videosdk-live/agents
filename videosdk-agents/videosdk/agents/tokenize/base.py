"""Abstract interfaces for text tokenization and filtering.

Cascade pipelines use these two stages between the LLM and the TTS:

    LLM deltas -> TextFilter -> SentenceTokenizer -> user hook -> TTS

Both stages consume and produce ``AsyncIterator[str]`` so they compose cleanly
with the existing ``tts_stream_gen`` in the orchestrator and with the optional
``@pipeline.on("llm")`` user hook.

Implementations live in ``basic.py`` (regex-based, zero-dep) and ``filters.py``
(Markdown / symbol normalisation). External plugins (e.g. a future
``videosdk-plugins-blingfire``) can provide alternative ``SentenceTokenizer``
implementations without touching core code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class SentenceStream(ABC):
    """Push-based stream adapter for sentence tokenizers.

    A ``SentenceStream`` is single-use: open it with ``SentenceTokenizer.stream()``,
    push text as deltas arrive, call ``end_input()`` when the upstream source
    closes, and iterate over it to receive sentence-sized strings.
    """

    @abstractmethod
    async def push_text(self, text: str) -> None:
        """Feed more text into the stream.

        Args:
            text: A chunk of text. May be a single character or a multi-word
                fragment. Need not align with word or sentence boundaries.
        """

    @abstractmethod
    async def flush(self) -> None:
        """Force-emit any buffered text as a single trailing sentence."""

    @abstractmethod
    async def end_input(self) -> None:
        """Signal that no more text will arrive. Drains the buffer and closes the stream."""

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[str]:
        ...


class SentenceTokenizer(ABC):
    """Abstract tokenizer that splits text into sentence-sized segments."""

    @abstractmethod
    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        """Split the given text into sentences in one shot.

        Args:
            text: Full text to split.
            language: Optional ISO 639-1 language hint. When omitted, the
                tokenizer uses its internal heuristic (usually script detection).

        Returns:
            A list of sentence-sized strings with leading/trailing whitespace stripped.
        """

    @abstractmethod
    def stream(self, *, language: str | None = None) -> SentenceStream:
        """Open a push-based stream for incremental tokenization.

        Args:
            language: Optional ISO 639-1 language hint forwarded to ``tokenize``.

        Returns:
            A fresh ``SentenceStream`` instance.
        """


class TextFilter(ABC):
    """Pre-tokenization text transformation.

    Filters sit *before* the tokenizer. They may be stateful across a turn
    (e.g. tracking whether the stream is currently inside a Markdown code
    fence) and are reset between turns via ``reset()``.
    """

    @abstractmethod
    def filter(self, chunks: AsyncIterator[str]) -> AsyncIterator[str]:
        """Transform an input stream of text chunks.

        Args:
            chunks: Async iterator of raw LLM deltas.

        Yields:
            Filtered text chunks ready to be consumed by a ``SentenceTokenizer``.
        """

    @abstractmethod
    async def reset(self) -> None:
        """Reset internal state between turns."""
