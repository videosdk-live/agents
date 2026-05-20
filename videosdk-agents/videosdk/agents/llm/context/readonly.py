from __future__ import annotations

from typing import List

from .context import ChatContext
from .items import ChatItem

_MUTATION_ERROR = (
    "This ChatContext is read-only. To modify it, create a mutable copy with "
    "ctx.copy() or ctx.fork() and modify that instead."
)


class ReadOnlyChatContext(ChatContext):
    """A read-only view over a ChatContext's items.

    All read operations (items, messages, copy, fork, get_by_id,
    active_config_at, serialization) work normally. Every mutating operation
    raises ``RuntimeError``. Used when a shared context is handed to an agent
    that must not mutate the parent.
    """

    def __init__(self, items: List[ChatItem]):
        # Hold a reference to the live items list — this is a view, not a copy.
        self._items = items

    def _readonly(self, *args, **kwargs):
        raise RuntimeError(_MUTATION_ERROR)

    async def _readonly_async(self, *args, **kwargs):
        raise RuntimeError(_MUTATION_ERROR)

    # Sync mutators
    add_message = _readonly
    add_function_call = _readonly
    add_function_output = _readonly
    add_handoff = _readonly
    add_config_update = _readonly
    insert = _readonly
    insert_many = _readonly
    truncate = _readonly
    cleanup = _readonly

    # Async mutators
    summarize = _readonly_async
    merge = _readonly_async
    merge_result = _readonly_async
    merge_summary = _readonly_async
