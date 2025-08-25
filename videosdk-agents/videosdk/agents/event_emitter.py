from typing import Any, Callable, Dict, List, TypeVar, Generic
import asyncio
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T", contravariant=True)


class EventEmitter(Generic[T]):
    def __init__(self) -> None:
        self._handlers: Dict[T, List[Callable[..., Any]]] = {}

    def on(
        self, event: T, callback: Callable[..., Any] | None = None
    ) -> Callable[..., Any]:
        def register(handler: Callable[..., Any]) -> Callable[..., Any]:
            if asyncio.iscoroutinefunction(handler):
                raise ValueError(
                    "Async handlers are not supported. Use a sync wrapper."
                )
            handlers = self._handlers.setdefault(event, [])
            if handler not in handlers:
                handlers.append(handler)
            return handler

        return register if callback is None else register(callback)

    def off(self, event: T, callback: Callable[..., Any]) -> None:
        if event in self._handlers:
            try:
                self._handlers[event].remove(callback)
            except ValueError:
                pass
            if not self._handlers[event]:
                del self._handlers[event]

    def emit(self, event: T, *args: Any) -> None:
        callbacks = self._handlers.get(event)
        if not callbacks:
            return

        arguments = args if args else ({},)
        for cb in callbacks[:]:
            try:
                self._invoke(cb, arguments)
            except Exception as ex:
                logger.error(f"Handler raised exception on event '{event}': {ex}")

    def _invoke(self, func: Callable[..., Any], args: tuple[Any, ...]) -> None:
        code = func.__code__
        argcount = code.co_argcount
        flags = code.co_flags
        has_varargs = flags & 0x04 != 0

        # If the function expects no arguments (only self), don't pass any
        if argcount == 1:  # Only self parameter
            func()
        elif has_varargs:
            func(*args)
        else:
            func(*args[:argcount])
