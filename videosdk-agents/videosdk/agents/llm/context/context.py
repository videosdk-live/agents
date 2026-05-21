from __future__ import annotations

import logging
import time
import uuid
from typing import List, Optional, Union, TYPE_CHECKING

from ...utils import FunctionTool, is_function_tool, get_tool_info
from .items import (
    ChatContent,
    ChatItem,
    ChatMessage,
    ChatRole,
    FunctionCall,
    FunctionCallOutput,
    ImageContent,
    AgentHandoff,
    AgentConfigUpdate,
)

if TYPE_CHECKING:
    from ..llm import LLM

logger = logging.getLogger(__name__)


class ChatContext:
    """
    Manages a conversation context for LLM interactions.
    """

    def __init__(self, items: Optional[List[ChatItem]] = None):
        """
        Initialize the chat context.

        Args:
            items (Optional[List[ChatItem]]): Initial list of chat items. If None, starts with empty context.
        """
        self._items: List[ChatItem] = items or []

    @classmethod
    def empty(cls) -> ChatContext:
        """
        Create an empty chat context.

        Returns:
            ChatContext: A new empty chat context instance.
        """
        return cls([])

    @property
    def items(self) -> List[ChatItem]:
        """
        Get all items in the context.

        Returns:
            List[ChatItem]: List of all conversation items (messages, function calls, outputs).
        """
        return self._items

    def messages(self) -> List[ChatMessage]:
        """
        Return only ChatMessage items, filtering out function calls and outputs.

        Returns:
            List[ChatMessage]: List of all chat messages in the context.
        """
        return [item for item in self._items if isinstance(item, ChatMessage)]

    def turn_count(self) -> int:
        """
        Count the number of user turns (user-assistant exchange pairs).

        Returns:
            int: Number of user messages in the context.
        """
        return sum(
            1 for item in self._items
            if isinstance(item, ChatMessage) and item.role == ChatRole.USER
        )

    def estimated_tokens(self) -> int:
        """
        Rough token estimate for the current context using a ~4 chars per token heuristic.
        Good enough for budget decisions — not a replacement for provider-reported usage.

        Returns:
            int: Estimated token count.
        """
        total = 0
        for item in self._items:
            total += self._estimate_item_tokens(item)
        return total

    def _estimate_item_tokens(self, item: ChatItem) -> int:
        """Estimate tokens for a single chat item."""
        tokens = 4
        if isinstance(item, ChatMessage):
            parts = item.content if isinstance(item.content, list) else [item.content]
            for part in parts:
                if part is None:
                    continue
                if isinstance(part, str):
                    tokens += len(part) // 4
                elif isinstance(part, ImageContent):
                    tokens += 300
        elif isinstance(item, FunctionCall):
            tokens += len(item.name) // 4 + 5
            if item.arguments:
                tokens += len(item.arguments) // 4
        elif isinstance(item, FunctionCallOutput):
            tokens += len(item.name) // 4 + 5
            if item.output:
                tokens += len(item.output) // 4
        return tokens

    def add_message(
        self,
        role: ChatRole,
        content: Union[str, List[ChatContent]],
        message_id: Optional[str] = None,
        created_at: Optional[float] = None,
        replace: bool = False,
        agent_id: Optional[str] = None,
    ) -> ChatMessage:
        """Add a new message to the context."""
        if replace and role == ChatRole.SYSTEM:
            self._items = [
                item for item in self._items
                if not (isinstance(item, ChatMessage) and item.role == ChatRole.SYSTEM)
            ]

        if isinstance(content, str):
            content = [content]

        message = ChatMessage(
            role=role,
            content=content,
            id=message_id or f"msg_{uuid.uuid4().hex[:12]}",
            created_at=created_at or time.time(),
            agent_id=agent_id,
        )
        self._items.append(message)
        return message

    def add_function_call(
        self,
        name: str,
        arguments: str,
        call_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> FunctionCall:
        """Add a function call to the context.

        ``metadata`` carries provider-specific per-call data — notably the
        Gemini ``thought_signature`` — which must travel with this exact call
        when the context is later converted for the provider.
        """
        call = FunctionCall(
            name=name,
            arguments=arguments,
            call_id=call_id or f"call_{uuid.uuid4().hex[:12]}",
            agent_id=agent_id,
            metadata=metadata,
        )
        self._items.append(call)
        return call

    def add_function_output(
        self,
        name: str,
        output: str,
        call_id: str,
        is_error: bool = False,
        agent_id: Optional[str] = None,
    ) -> FunctionCallOutput:
        """Add a function output to the context."""
        function_output = FunctionCallOutput(
            name=name,
            output=output,
            call_id=call_id,
            is_error=is_error,
            agent_id=agent_id,
        )
        self._items.append(function_output)
        return function_output

    def add_handoff(
        self,
        to_agent: str,
        from_agent: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> AgentHandoff:
        """Record a transfer of control between agents."""
        handoff = AgentHandoff(from_agent=from_agent, to_agent=to_agent, reason=reason)
        self._items.append(handoff)
        return handoff

    def add_config_update(
        self,
        instructions: Optional[str] = None,
        tools: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
    ) -> AgentConfigUpdate:
        """Record a mid-conversation change to instructions or tools."""
        update = AgentConfigUpdate(
            instructions=instructions, tools=tools, agent_id=agent_id
        )
        self._items.append(update)
        return update

    def get_by_id(self, item_id: str) -> Optional[ChatItem]:
        """
        Find an item by its ID.

        Args:
            item_id (str): The ID of the item to find.

        Returns:
            Optional[ChatItem]: The found item or None if not found.
        """
        return next(
            (item for item in self._items if item.id == item_id),
            None
        )

    def insert(self, item: ChatItem) -> ChatItem:
        """Insert an item at the position determined by its ``created_at``."""
        pos = len(self._items)
        for i, existing in enumerate(self._items):
            if existing.created_at > item.created_at:
                pos = i
                break
        self._items.insert(pos, item)
        return item

    def insert_many(self, items: List[ChatItem]) -> None:
        """Batch-insert items, each placed in timestamp order."""
        for item in sorted(items, key=lambda it: it.created_at):
            self.insert(item)

    def active_config_at(
        self, target: Union[str, int, None] = None
    ) -> tuple[Optional[str], Optional[List[str]]]:
        """Resolve the effective (instructions, valid_tools) at a point in the context.

        Walks SYSTEM/DEVELOPER messages and AgentConfigUpdate items from the
        start up to and including ``target``.

        Args:
            target: An item id, a list index, or None for the end of the context.

        Returns:
            tuple: (instructions, valid_tools). Either element may be None.
        """
        if target is None:
            end = len(self._items)
        elif isinstance(target, int):
            end = target + 1
        else:
            end = next(
                (i + 1 for i, item in enumerate(self._items) if item.id == target),
                len(self._items),
            )

        instructions: Optional[str] = None
        tools: Optional[List[str]] = None
        for item in self._items[:end]:
            if isinstance(item, ChatMessage) and item.role in (
                ChatRole.SYSTEM,
                ChatRole.DEVELOPER,
            ):
                if isinstance(item.content, str):
                    instructions = item.content
                else:
                    instructions = " ".join(
                        p for p in item.content if isinstance(p, str)
                    )
            elif isinstance(item, AgentConfigUpdate):
                if item.instructions is not None:
                    instructions = item.instructions
                if item.tools is not None:
                    tools = item.tools
        return instructions, tools

    def copy(
        self,
        *,
        exclude_system_messages: bool = False,
        exclude_instructions: bool = False,
        exclude_empty_messages: bool = False,
        exclude_handoffs: bool = False,
        exclude_config_updates: bool = False,
        tools: Optional[List[FunctionTool]] = None,
        filter_agent_id: Optional[str] = None,
    ) -> ChatContext:
        """Create a filtered copy of the chat context.

        Args:
            exclude_system_messages: Drop SYSTEM-role messages.
            exclude_instructions: Drop SYSTEM- and DEVELOPER-role messages.
            exclude_empty_messages: Drop messages with no meaningful content.
            exclude_handoffs: Drop AgentHandoff items.
            exclude_config_updates: Drop AgentConfigUpdate items.
            tools: Tool-scoping for function calls/outputs. ``None`` (the
                default) keeps every function call/output; an empty list drops
                them all; a non-empty list keeps only calls/outputs whose tool
                is in the list.
            filter_agent_id: When given, keep only items with this agent_id.

        Returns:
            ChatContext: A new ChatContext with the filtered items.
        """
        items: List[ChatItem] = []
        valid_tool_names = {
            get_tool_info(tool).name
            for tool in (tools or [])
            if is_function_tool(tool)
        }

        for item in self._items:
            if isinstance(item, ChatMessage):
                if exclude_system_messages and item.role == ChatRole.SYSTEM:
                    continue
                if exclude_instructions and item.role in (
                    ChatRole.SYSTEM,
                    ChatRole.DEVELOPER,
                ):
                    continue
                if exclude_empty_messages and not self._has_content(item):
                    continue

            if exclude_handoffs and isinstance(item, AgentHandoff):
                continue
            if exclude_config_updates and isinstance(item, AgentConfigUpdate):
                continue

            if tools is not None and isinstance(
                item, (FunctionCall, FunctionCallOutput)
            ):
                if item.name not in valid_tool_names:
                    continue

            if filter_agent_id is not None:
                _structural = (
                    isinstance(item, (AgentHandoff, AgentConfigUpdate))
                    or (
                        isinstance(item, ChatMessage)
                        and item.role in (ChatRole.SYSTEM, ChatRole.DEVELOPER)
                    )
                )
                if not _structural and item.agent_id != filter_agent_id:
                    continue

            items.append(item)

        return ChatContext(items)

    @staticmethod
    def _has_content(msg: ChatMessage) -> bool:
        """Return True if the message has any non-empty content."""
        if isinstance(msg.content, str):
            return bool(msg.content.strip())
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, str) and part.strip():
                    return True
                if part is not None and not isinstance(part, str):
                    return True
        return False

    def fork(self) -> ChatContext:
        """Fork a complete, independent deep copy for a sub-agent.

        Returns:
            ChatContext: A new context; later mutations never touch this one.
        """
        return ChatContext([item.model_copy(deep=True) for item in self._items])

    def fork_filtered(
        self,
        recent_turns: int = 3,
        tools: Optional[List[FunctionTool]] = None,
    ) -> ChatContext:
        """Fork a context scoped to instructions + the most recent turns.

        Args:
            recent_turns: Number of recent user turns to keep (must be >= 1).
            tools: When given, function calls/outputs are limited to these tools.

        Returns:
            ChatContext: A new, independent context.
        """
        if recent_turns < 1:
            raise ValueError("recent_turns must be >= 1")

        scoped = self.copy(tools=tools)
        instruction_items = [
            item for item in scoped.items
            if isinstance(item, ChatMessage)
            and item.role in (ChatRole.SYSTEM, ChatRole.DEVELOPER)
        ]
        instruction_ids = {item.id for item in instruction_items}

        user_indices = [
            i for i, item in enumerate(scoped.items)
            if isinstance(item, ChatMessage) and item.role == ChatRole.USER
        ]
        if len(user_indices) > recent_turns:
            split_idx = user_indices[-recent_turns]
            tail = scoped.items[split_idx:]
        else:
            tail = scoped.items

        new_items = [item.model_copy(deep=True) for item in instruction_items]
        new_items += [
            item.model_copy(deep=True)
            for item in tail
            if item.id not in instruction_ids
        ]
        return ChatContext(new_items)

    def fork_brief(
        self,
        instructions: str,
        task_brief: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> ChatContext:
        """Fork a fresh context: sub-agent instructions + an optional task brief.

        Args:
            instructions: System instructions for the sub-agent (required,
                non-empty).
            task_brief: Optional task-description message.
            agent_id: Attribution stamped on the created items.

        Returns:
            ChatContext: A new context with no conversation history.
        """
        if not instructions:
            raise ValueError("fork_brief() requires non-empty instructions")

        items: List[ChatItem] = [
            ChatMessage(
                role=ChatRole.SYSTEM, content=[instructions], agent_id=agent_id
            )
        ]
        if task_brief:
            items.append(
                ChatMessage(
                    role=ChatRole.USER, content=[task_brief], agent_id=agent_id
                )
            )
        return ChatContext(items)

    async def merge(self, other: ChatContext) -> ChatContext:
        """Merge a sub-agent's full transcript into this context, in-place.

        Every item from ``other`` is merged, timestamp-ordered and
        de-duplicated by id. This is the most complete merge-back; the
        ``merge_result`` and ``merge_summary`` variants merge less.

        Args:
            other: The sub-agent's context.

        Returns:
            ChatContext: This context instance (modified in-place).
        """
        existing_ids = {item.id for item in self._items}
        incoming = [
            item.model_copy(deep=True)
            for item in other.items
            if item.id not in existing_ids
        ]
        combined = self._items + incoming
        combined.sort(key=lambda item: item.created_at)
        self._items = combined
        return self

    async def merge_result(
        self, other: ChatContext, *, agent_id: Optional[str] = None
    ) -> ChatContext:
        """Merge only a sub-agent's final assistant message, in-place.

        Args:
            other: The sub-agent's context.
            agent_id: Attribution stamped on the merged-in message.

        Returns:
            ChatContext: This context instance (modified in-place).
        """
        final = next(
            (
                item for item in reversed(other.items)
                if isinstance(item, ChatMessage)
                and item.role == ChatRole.ASSISTANT
            ),
            None,
        )
        if final is not None:
            merged = final.model_copy(deep=True)
            if agent_id is not None:
                merged.agent_id = agent_id
            self._items.append(merged)
        return self

    async def merge_summary(
        self,
        other: ChatContext,
        *,
        llm: "LLM",
        agent_id: Optional[str] = None,
    ) -> ChatContext:
        """Merge an LLM-generated summary of a sub-agent's work, in-place.

        Args:
            other: The sub-agent's context.
            llm: LLM used to generate the summary (required keyword argument).
            agent_id: Attribution stamped on the summary message.

        Returns:
            ChatContext: This context instance (modified in-place).
        """
        from .window import render_items, generate_summary

        text = render_items(other.items)
        summary_text = await generate_summary(llm, text) if text.strip() else ""
        if summary_text:
            self._items.append(
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    content=[f"[Sub-agent Summary]\n{summary_text}"],
                    agent_id=agent_id,
                    extra={"summary": True},
                )
            )
        return self

    def truncate(
        self,
        max_items: int | None = None,
        max_tokens: int | None = None,
    ) -> ChatContext:
        """
        Truncate the context while preserving system message and summary messages.

        Removes oldest non-system items until both constraints are satisfied.
        Keeps function call/output pairs together to avoid orphaned tool calls.

        Args:
            max_items: Maximum number of items to keep. None means no item limit.
            max_tokens: Maximum estimated token budget. None means no token limit.

        Returns:
            ChatContext: The current context instance after truncation.
        """
        if max_items is None and max_tokens is None:
            return self

        logger.debug(f"Truncating context: {len(self._items)} items, {self.estimated_tokens()} tokens")

        # Identify protected items that must never be removed:
        # - System message (agent instructions)
        # - Summary message (compressed history)
        # - Last user message (LLMs require conversation to end with user turn)
        system_msg = next(
            (item for item in self._items
             if isinstance(item, ChatMessage)
             and item.role in (ChatRole.SYSTEM, ChatRole.DEVELOPER)),
            None
        )
        summary_msg = next(
            (item for item in self._items
             if isinstance(item, ChatMessage) and item.extra.get("summary")),
            None
        )
        last_user_msg = next(
            (item for item in reversed(self._items)
             if isinstance(item, ChatMessage) and item.role == ChatRole.USER),
            None
        )
        structural_items = [
            item for item in self._items
            if isinstance(item, (AgentHandoff, AgentConfigUpdate))
        ]
        protected = {
            id(m)
            for m in (system_msg, summary_msg, last_user_msg, *structural_items)
            if m is not None
        }

        # Start with all items; remove oldest non-protected until constraints met
        new_items = list(self._items)

        def _needs_trim() -> bool:
            if max_items is not None and len(new_items) > max_items:
                return True
            if max_tokens is not None:
                token_est = sum(self._estimate_item_tokens(it) for it in new_items)
                if token_est > max_tokens:
                    return True
            return False

        while _needs_trim():
            removed = False
            for i, item in enumerate(new_items):
                # Skip protected items
                if id(item) in protected:
                    continue
                # Don't orphan function call pairs — remove them together
                if isinstance(item, FunctionCall):
                    output_idx = next(
                        (j for j in range(i + 1, len(new_items))
                         if isinstance(new_items[j], FunctionCallOutput) and new_items[j].call_id == item.call_id),
                        None
                    )
                    if output_idx is not None:
                        new_items.pop(output_idx)
                        new_items.pop(i)
                    else:
                        new_items.pop(i)
                    removed = True
                    break
                elif isinstance(item, FunctionCallOutput):
                    new_items.pop(i)
                    removed = True
                    break
                else:
                    new_items.pop(i)
                    removed = True
                    break
            if not removed:
                break  # Only protected items remain — stop even if over budget

        # Clean up ALL orphaned function items (call without output, or output without call)
        call_ids_in_list = {item.call_id for item in new_items if isinstance(item, FunctionCall)}
        output_ids_in_list = {item.call_id for item in new_items if isinstance(item, FunctionCallOutput)}
        new_items = [
            item for item in new_items
            if not (
                (isinstance(item, FunctionCall) and item.call_id not in output_ids_in_list)
                or
                (isinstance(item, FunctionCallOutput) and item.call_id not in call_ids_in_list)
            )
        ]

        # Re-insert protected items if they were accidentally removed by orphan cleanup
        if system_msg and system_msg not in new_items:
            new_items.insert(0, system_msg)
        if summary_msg and summary_msg not in new_items:
            insert_pos = 1 if system_msg in new_items else 0
            new_items.insert(insert_pos, summary_msg)
        if last_user_msg and last_user_msg not in new_items:
            new_items.append(last_user_msg)

        self._items = new_items
        logger.debug(f"Truncation complete: {len(self._items)} items, {self.estimated_tokens()} tokens")
        return self

    async def summarize(
        self, llm: "LLM", *, keep_recent_turns: int = 3
    ) -> ChatContext:
        """Compress old conversation turns into a single summary message, in-place.

        Splits the context into head (older) and tail (recent). The head is
        rendered and summarized by ``llm``; structural items (system/developer
        messages, prior summaries, handoffs) are preserved.

        Args:
            llm: LLM used to generate the summary.
            keep_recent_turns: Number of recent user turns kept verbatim.

        Returns:
            ChatContext: This context instance (modified in-place).
        """
        from .window import render_items, generate_summary

        user_indices = [
            i for i, item in enumerate(self._items)
            if isinstance(item, ChatMessage) and item.role == ChatRole.USER
        ]
        if len(user_indices) <= keep_recent_turns:
            return self

        split_idx = user_indices[-keep_recent_turns]
        head = self._items[:split_idx]
        recent_items = list(self._items[split_idx:])

        def _is_structural(item: ChatItem) -> bool:
            if isinstance(item, (AgentHandoff, AgentConfigUpdate)):
                return True
            if isinstance(item, ChatMessage):
                return (
                    item.role in (ChatRole.SYSTEM, ChatRole.DEVELOPER)
                    or bool(item.extra.get("summary"))
                )
            return False

        structural = [item for item in head if _is_structural(item)]
        summarizable = [item for item in head if not _is_structural(item)]
        if not summarizable:
            return self

        conversation_text = render_items(summarizable)
        if not conversation_text.strip():
            return self

        summary_text = await generate_summary(llm, conversation_text)
        if not summary_text:
            logger.warning("Compression produced empty summary")
            return self

        summary_msg = ChatMessage(
            role=ChatRole.ASSISTANT,
            content=[f"[Conversation Summary]\n{summary_text}"],
            extra={"summary": True},
        )
        self._items = structural + [summary_msg] + recent_items
        logger.info(
            f"Compressed {len(summarizable)} items into summary. "
            f"Context: {len(self._items)} items"
        )
        return self

    # ── Provider format conversions ────────────────────────────────────
    # Actual logic lives in llm/format_converters.py. These methods
    # delegate to keep the public API on ChatContext unchanged.

    def to_openai_messages(self, *, reasoning_model: bool = False) -> list[dict]:
        """Convert context to OpenAI chat completion messages format."""
        from ..format_converters import to_openai_messages
        return to_openai_messages(self, reasoning_model=reasoning_model)

    def to_anthropic_messages(self, *, caching: bool = False) -> tuple[list[dict], Optional[str]]:
        """Convert context to Anthropic messages format with role alternation enforced."""
        from ..format_converters import to_anthropic_messages
        return to_anthropic_messages(self, caching=caching)

    async def to_google_contents(self, *, thought_signatures: dict | None = None) -> tuple[list, Optional[str]]:
        """Convert context to Google Gemini contents format."""
        from ..format_converters import to_google_contents
        return await to_google_contents(self, thought_signatures=thought_signatures)

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Convert the context to a dictionary representation."""
        items = []
        for item in self._items:
            base = {
                "type": item.type,
                "id": item.id,
                "created_at": item.created_at,
                "agent_id": item.agent_id,
            }
            if isinstance(item, ChatMessage):
                base.update({
                    "role": item.role.value,
                    "content": item.content,
                    "interrupted": item.interrupted,
                    "extra": item.extra,
                    "confidence": item.confidence,
                    "metrics": item.metrics,
                    "audio_instructions": item.audio_instructions,
                    "text_instructions": item.text_instructions,
                })
            elif isinstance(item, FunctionCall):
                base.update({
                    "name": item.name,
                    "arguments": item.arguments,
                    "call_id": item.call_id,
                    "metadata": item.metadata,
                })
            elif isinstance(item, FunctionCallOutput):
                base.update({
                    "name": item.name,
                    "output": item.output,
                    "call_id": item.call_id,
                    "is_error": item.is_error,
                })
            elif isinstance(item, AgentHandoff):
                base.update({
                    "from_agent": item.from_agent,
                    "to_agent": item.to_agent,
                    "reason": item.reason,
                })
            elif isinstance(item, AgentConfigUpdate):
                base.update({
                    "instructions": item.instructions,
                    "tools": item.tools,
                })
            items.append(base)
        return {"items": items}

    @classmethod
    def from_dict(cls, data: dict) -> ChatContext:
        """Reconstruct a ChatContext from a dictionary representation."""
        items: List[ChatItem] = []
        for d in data["items"]:
            common = {"id": d["id"]}
            if d.get("created_at") is not None:
                common["created_at"] = d["created_at"]
            if "agent_id" in d:
                common["agent_id"] = d.get("agent_id")

            item_type = d["type"]
            if item_type == "message":
                items.append(ChatMessage(
                    role=ChatRole(d["role"]),
                    content=d["content"],
                    interrupted=d.get("interrupted", False),
                    extra=d.get("extra", {}) or {},
                    confidence=d.get("confidence"),
                    metrics=d.get("metrics"),
                    audio_instructions=d.get("audio_instructions"),
                    text_instructions=d.get("text_instructions"),
                    **common,
                ))
            elif item_type == "function_call":
                items.append(FunctionCall(
                    name=d["name"],
                    arguments=d["arguments"],
                    call_id=d["call_id"],
                    metadata=d.get("metadata"),
                    **common,
                ))
            elif item_type == "function_call_output":
                items.append(FunctionCallOutput(
                    name=d["name"],
                    output=d["output"],
                    call_id=d["call_id"],
                    is_error=d.get("is_error", False),
                    **common,
                ))
            elif item_type == "agent_handoff":
                items.append(AgentHandoff(
                    from_agent=d.get("from_agent"),
                    to_agent=d["to_agent"],
                    reason=d.get("reason"),
                    **common,
                ))
            elif item_type == "agent_config_update":
                items.append(AgentConfigUpdate(
                    instructions=d.get("instructions"),
                    tools=d.get("tools"),
                    **common,
                ))
        return cls(items)

    def cleanup(self) -> None:
        """
        Clear all chat context items and references to free memory.
        """
        logger.info(f"Cleaning up ChatContext with {len(self._items)} items")
        for item in self._items:
            if isinstance(item, ChatMessage):
                if isinstance(item.content, list):
                    for content_item in item.content:
                        if isinstance(content_item, ImageContent):
                            content_item.image = None
                item.content = None
            elif isinstance(item, FunctionCall):
                item.arguments = None
            elif isinstance(item, FunctionCallOutput):
                item.output = None
        self._items.clear()
        try:
            import gc
            gc.collect()
            logger.info("ChatContext garbage collection completed")
        except Exception as e:
            logger.error(f"Error during ChatContext garbage collection: {e}")

        logger.info("ChatContext cleanup completed")
