from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .context import ChatContext
from .items import ChatMessage, ChatRole, FunctionCall, FunctionCallOutput

if TYPE_CHECKING:
    from ..llm import LLM

logger = logging.getLogger(__name__)

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


def _extract_text(msg: ChatMessage) -> str:
    """Extract plain text content from a ChatMessage."""
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        parts = [p for p in msg.content if isinstance(p, str)]
        return " ".join(parts) if parts else "[non-text content]"
    return ""


def render_items(items: list) -> str:
    """Render chat items as human-readable text for a summarization prompt."""
    lines = []
    for item in items:
        if isinstance(item, ChatMessage) and item.extra.get("summary"):
            lines.append(f"[Previous Summary]: {_extract_text(item)}")
        elif isinstance(item, ChatMessage):
            role_label = item.role.value.capitalize()
            lines.append(f"{role_label}: {_extract_text(item)}")
        elif isinstance(item, FunctionCall):
            lines.append(f"[Tool Call: {item.name}] args={item.arguments}")
        elif isinstance(item, FunctionCallOutput):
            output_preview = item.output[:200] if item.output else ""
            label = "Tool Error" if item.is_error else "Tool Result"
            lines.append(f"[{label}: {item.name}] {output_preview}")
    return "\n".join(lines)


async def generate_summary(llm: "LLM", conversation_text: str) -> str:
    """Call the LLM to summarize a conversation excerpt."""
    summary_ctx = ChatContext()
    summary_ctx.add_message(role=ChatRole.SYSTEM, content=SUMMARIZE_PROMPT)
    summary_ctx.add_message(role=ChatRole.USER, content=conversation_text)

    result_parts = []
    try:
        async for chunk in llm.chat(summary_ctx):
            if chunk and chunk.content:
                result_parts.append(chunk.content)
    except Exception as e:
        logger.error(f"Error during summary generation: {e}")
        return ""
    return "".join(result_parts)


class ContextWindow:
    """Policy layer that decides *when* to compress/truncate a ChatContext.

    Compression and truncation primitives live on ChatContext; this class
    holds the budgets and delegates to them.

    Args:
        max_tokens: Maximum estimated token budget.
        max_context_items: Maximum number of items.
        keep_recent_turns: Recent user-assistant exchanges kept verbatim
            during compression.
        max_tool_calls_per_turn: Retained for compatibility with callers.
        summary_llm: Optional dedicated LLM for summaries; falls back to the
            agent's main LLM.
    """

    def __init__(
        self,
        *,
        max_tokens: int | None = None,
        max_context_items: int | None = None,
        keep_recent_turns: int = 3,
        max_tool_calls_per_turn: int = 10,
        summary_llm: "LLM | None" = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.max_context_items = max_context_items
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self._keep_recent = keep_recent_turns
        self._summary_llm = summary_llm

    async def manage(self, ctx: ChatContext, llm: "LLM") -> None:
        """Run the full context management cycle: compress then truncate."""
        if self._needs_compression(ctx):
            try:
                active_llm = self._summary_llm or llm
                if active_llm is not None:
                    logger.info("Compressing context via summary generation")
                    await ctx.summarize(active_llm, keep_recent_turns=self._keep_recent)
                    logger.info(
                        f"Context compression complete. Size: {len(ctx.items)} items"
                    )
                else:
                    logger.warning("No LLM available for context compression")
            except Exception as e:
                logger.error(f"Error during context compression: {e}", exc_info=True)

        if self.max_tokens is not None or self.max_context_items is not None:
            before = len(ctx.items)
            ctx.truncate(max_items=self.max_context_items, max_tokens=self.max_tokens)
            after = len(ctx.items)
            if after < before:
                logger.info(f"Truncated context from {before} to {after} items")

    def _needs_compression(self, ctx: ChatContext) -> bool:
        """Check if context exceeds budget and has enough old turns to compress."""
        est = ctx.estimated_tokens()
        items = len(ctx.items)
        turns = ctx.turn_count()
        exceeds_tokens = self.max_tokens is not None and est > self.max_tokens
        exceeds_items = (
            self.max_context_items is not None and items > self.max_context_items
        )
        enough_turns = turns > self._keep_recent + 1
        needed = (exceeds_tokens or exceeds_items) and enough_turns
        logger.debug(
            f"Compression check: {est} tokens (max={self.max_tokens}), "
            f"{items} items (max={self.max_context_items}), {turns} turns, needed={needed}"
        )
        return needed
