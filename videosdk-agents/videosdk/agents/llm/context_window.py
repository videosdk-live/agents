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


class ContextWindow:
    """
    Manages the conversation context window for LLM interactions.

    Handles two responsibilities:
    1. **Compression** — summarizes old conversation turns using an LLM so the
       agent retains long-term memory instead of losing it to truncation.
    2. **Truncation** — removes oldest items when the context exceeds the
       token or item budget, preserving system messages, summaries, and the
       last user message.

    Args:
        max_tokens: Maximum estimated token budget for the context.
            When exceeded, compression (if enough turns) then truncation kicks in.
        max_context_items: Maximum number of items in the context.
            When exceeded, same behavior as max_tokens.
        keep_recent_turns: Number of recent user-assistant exchanges to keep
            verbatim during compression. Everything older gets summarized.
        summary_llm: Optional dedicated LLM for generating summaries.
            If None, the agent's main LLM is used automatically.
    """

    SUMMARIZE_PROMPT = (
        "Summarize the following conversation excerpt. Preserve:\n"
        "- Key facts, names, and numbers mentioned\n"
        "- Decisions made and their reasoning\n"
        "- Tool/function call results and their outcomes\n"
        "- Any commitments or promises the assistant made\n"
        "- Emphasize user objectives, limitations, decisions made, important details, "
        "preferences, relevant items, and any outstanding or unresolved tasks.\n"
        "- Avoid greetings, extra wording, and casual remarks.\n"
        "Keep it concise but complete. Output ONLY the summary, nothing else."
    )

    def __init__(
        self,
        *,
        max_tokens: int | None = None,
        max_context_items: int | None = None,
        keep_recent_turns: int = 3,
        max_tool_calls_per_turn: int = 10,
        summary_llm: LLM | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.max_context_items = max_context_items
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self._keep_recent = keep_recent_turns
        self._summary_llm = summary_llm

    async def manage(self, ctx: ChatContext, llm: LLM) -> None:
        """
        Run the full context management cycle: compress then truncate.

        Called automatically before each LLM call by ContentGeneration.

        Args:
            ctx: The chat context to manage (modified in-place).
            llm: The agent's main LLM (used for summarization if no summary_llm set).
        """
        # Step 1: Compress old turns into a summary if budget exceeded
        if self._needs_compression(ctx):
            try:
                logger.info("Compressing context via summary generation")
                await self._compress(ctx, llm)
                logger.info(f"Context compression complete. Size: {len(ctx.items)} items")
            except Exception as e:
                logger.error(f"Error during context compression: {e}", exc_info=True)

        # Step 2: Truncate remaining items if still over budget
        if self.max_tokens is not None or self.max_context_items is not None:
            before = len(ctx.items)
            ctx.truncate(max_items=self.max_context_items, max_tokens=self.max_tokens)
            after = len(ctx.items)
            if after < before:
                logger.info(f"Truncated context from {before} to {after} items")

    # ── Compression ───────────────────────────────────────────────────

    def _needs_compression(self, ctx: ChatContext) -> bool:
        """Check if context exceeds budget and has enough old turns to compress."""
        est = ctx.estimated_tokens()
        items = len(ctx.items)
        turns = ctx.turn_count()
        exceeds_tokens = self.max_tokens is not None and est > self.max_tokens
        exceeds_items = self.max_context_items is not None and items > self.max_context_items
        enough_turns = turns > self._keep_recent + 1
        needed = (exceeds_tokens or exceeds_items) and enough_turns
        logger.debug(
            f"Compression check: {est} tokens (max={self.max_tokens}), "
            f"{items} items (max={self.max_context_items}), {turns} turns, needed={needed}"
        )
        return needed

    async def _compress(self, ctx: ChatContext, llm: LLM) -> None:
        """Compress old conversation turns into a summary, in-place."""
        active_llm = self._summary_llm or llm
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
        recent_items = list(items[split_idx:])

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