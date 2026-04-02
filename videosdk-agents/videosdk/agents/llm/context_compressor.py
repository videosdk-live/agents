from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .chat_context import (
    ChatContext,
    ChatMessage,
    ChatRole,
    FunctionCall,
    FunctionCallOutput,
)

if TYPE_CHECKING:
    from .llm import LLM

logger = logging.getLogger(__name__)


class ContextCompressor:
    """
    Generates and maintains a running summary of old conversation history.
    Uses the agent's existing LLM by default — no extra model configuration needed.

    When the context exceeds a token budget, older turns are compressed into a
    single summary message while recent turns are kept verbatim. The summary is
    tagged so it won't be re-summarized redundantly.

    Usage::

        Pipeline(
            ...,
            max_context_tokens=8000,
            context_compressor=ContextCompressor(keep_recent_turns=3),
        )
    """

    SUMMARIZE_PROMPT = (
        "Summarize the following conversation excerpt. Preserve:\n"
        "- Key facts, names, and numbers mentioned\n"
        "- Decisions made and their reasoning\n"
        "- Tool/function call results and their outcomes\n"
        "- Any commitments or promises the assistant made\n"
        "- Emphasize user objectives, limitations, decisions made, important details, preferences, relevant items, and any outstanding or unresolved tasks.\n"
        "- Avoid greetings, extra wording, and casual remarks.\n"
        "Keep it concise but complete. Output ONLY the summary, nothing else."
    )

    def __init__(
        self,
        *,
        keep_recent_turns: int = 3,
        llm: LLM | None = None,
    ) -> None:
        """
        Args:
            keep_recent_turns: Number of recent user-assistant exchanges to keep
                              verbatim. Everything older gets compressed.
            llm: Optional dedicated LLM for summarization. If None, the agent's
                 main LLM is used (passed at call time by ContentGeneration).
        """
        self._keep_recent = keep_recent_turns
        self._dedicated_llm = llm

    def needs_compression(self, ctx: ChatContext, max_tokens: int) -> bool:
        """Check if context exceeds budget and has enough old turns to compress."""
        est = ctx.estimated_tokens()
        turns = ctx.turn_count()
        needed = est > max_tokens and turns > self._keep_recent + 1
        logger.debug(f"Compression check: {est}/{max_tokens} tokens, {turns} turns, needed={needed}")
        return needed

    async def compress(self, ctx: ChatContext, llm: LLM) -> None:
        """
        Compress old conversation turns into a summary, in-place.

        Args:
            ctx: The chat context to compress.
            llm: The LLM to use for summarization (typically the agent's main LLM).
                 Ignored if a dedicated LLM was provided at init.
        """
        active_llm = self._dedicated_llm or llm
        if not active_llm:
            logger.warning("No LLM available for context compression")
            return

        old_items, recent_items = self._split_items(ctx)
        if not old_items:
            logger.debug("No old items to compress")
            return

        conversation_text = self._render_items(old_items)
        if not conversation_text.strip():
            return

        logger.info("Generating context summary...")
        summary_text = await self._generate_summary(active_llm, conversation_text)
        if not summary_text:
            logger.warning("Compression produced empty summary")
            return
        logger.info(f"Summary generated ({len(summary_text)} chars)")

        system_msg = next(
            (item for item in ctx.items
             if isinstance(item, ChatMessage) and item.role == ChatRole.SYSTEM),
            None,
        )

        new_items = []
        if system_msg:
            new_items.append(system_msg)

        summary_msg = ChatMessage(
            role=ChatRole.ASSISTANT,
            content=[f"[Conversation Summary]\n{summary_text}"],
            extra={"summary": True},
        )
        new_items.append(summary_msg)
        new_items.extend(recent_items)

        ctx._items = new_items
        logger.info(f"Compressed {len(old_items)} items into summary. Context: {len(new_items)} items")

    def _split_items(self, ctx: ChatContext) -> tuple[list, list]:
        """Split items into old and recent, keeping the last N user turns in recent."""
        items = ctx.items

        user_indices = [
            i for i, item in enumerate(items)
            if isinstance(item, ChatMessage) and item.role == ChatRole.USER
        ]

        if len(user_indices) <= self._keep_recent:
            return [], items

        split_idx = user_indices[-self._keep_recent]

        old_items = [
            item for item in items[:split_idx]
            if not (
                isinstance(item, ChatMessage)
                and (item.role == ChatRole.SYSTEM or item.extra.get("summary"))
            )
        ]
        recent_items = [
            item for item in items[split_idx:]
        ]

        return old_items, recent_items

    def _render_items(self, items: list) -> str:
        """Render chat items as human-readable text for the summarization prompt."""
        lines = []

        for item in items:
            if isinstance(item, ChatMessage) and item.extra.get("summary"):
                text = self._extract_text(item)
                lines.append(f"[Previous Summary]: {text}")
                continue

            if isinstance(item, ChatMessage):
                role_label = item.role.value.capitalize()
                text = self._extract_text(item)
                if item.interrupted:
                    text += " [interrupted]"
                lines.append(f"{role_label}: {text}")
            elif isinstance(item, FunctionCall):
                lines.append(f"[Tool Call: {item.name}] args={item.arguments}")
            elif isinstance(item, FunctionCallOutput):
                output_preview = item.output[:200] if item.output else ""
                if item.is_error:
                    lines.append(f"[Tool Error: {item.name}] {output_preview}")
                else:
                    lines.append(f"[Tool Result: {item.name}] {output_preview}")

        return "\n".join(lines)

    @staticmethod
    def _extract_text(msg: ChatMessage) -> str:
        """Extract plain text content from a ChatMessage."""
        if isinstance(msg.content, str):
            return msg.content
        if isinstance(msg.content, list):
            parts = [p for p in msg.content if isinstance(p, str)]
            return " ".join(parts) if parts else "[non-text content]"
        return ""

    async def _generate_summary(self, llm: LLM, conversation_text: str) -> str:
        """Call the LLM to generate a summary of the conversation excerpt."""
        summary_ctx = ChatContext()
        summary_ctx.add_message(
            role=ChatRole.SYSTEM,
            content=self.SUMMARIZE_PROMPT,
        )
        summary_ctx.add_message(
            role=ChatRole.USER,
            content=conversation_text,
        )

        result_parts = []
        try:
            async for chunk in llm.chat(summary_ctx):
                if chunk and chunk.content:
                    result_parts.append(chunk.content)
        except Exception as e:
            logger.error(f"Error during summary generation: {e}")
            return ""

        return "".join(result_parts)