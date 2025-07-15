from typing import Any, Callable, Dict, Set, TypeVar, Generic
import asyncio
import inspect
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T", contravariant=True)

class EventEmitter(Generic[T]):
    def __init__(self) -> None:
        self._handlers: Dict[T, Set[Callable[..., Any]]] = {}

    def on(self, event: T, callback: Callable[..., Any] | None = None) -> Callable[..., Any]:
        def register(handler: Callable[..., Any]) -> Callable[..., Any]:
            if asyncio.iscoroutinefunction(handler):
                raise ValueError("Async handlers are not supported. Use a sync wrapper.")
            self._handlers.setdefault(event, set()).add(handler)
            return handler

        return register if callback is None else register(callback)

    def off(self, event: T, callback: Callable[..., Any]) -> None:
        if event in self._handlers:
            self._handlers[event].discard(callback)
            if not self._handlers[event]:
                del self._handlers[event]

    def emit(self, event: T, *args: Any) -> None:
        callbacks = self._handlers.get(event)
        if not callbacks:
            return

        arguments = args if args else ({},)
        for cb in list(callbacks):
            try:
                self._invoke(cb, arguments)
            except Exception as ex:
                logger.error(f"Handler raised exception on event '{event}': {ex}")

    def _invoke(self, func: Callable[..., Any], args: tuple[Any, ...]) -> None:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
            func(*args)
        else:
            max_args = sum(
                1 for p in params
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            )
            func(*args[:max_args])
