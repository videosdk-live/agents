from typing import Any, Callable, Dict, Set, TypeVar, Generic
import inspect
import asyncio
import logging

logger = logging.getLogger(__name__)

T_contra = TypeVar("T_contra", contravariant=True)

class EventEmitter(Generic[T_contra]):
    _instance = None
    _events: Dict[T_contra, Set[Callable]] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs) -> None:
        """Initialize event emitter with empty event handlers dictionary"""
        pass

    def emit(self, event: T_contra, *args) -> None:
        """
        Emit an event with arguments to all registered handlers
        
        Args:
            event: The event type/name to emit
            *args: Arguments to pass to the event handlers. If no args provided, 
                  an empty dict will be passed as default data
        """
        if event in self._events:
            # Create copy to avoid modification during iteration
            handlers = self._events[event].copy()
            
            if not args:
                args = ({},)
            
            for handler in handlers:
                try:
                    # Get handler signature
                    sig = inspect.signature(handler)
                    params = sig.parameters.values()
                    
                    # Check if handler accepts variable args
                    has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
                    
                    if has_varargs:
                        # Pass all args if handler accepts them
                        handler(*args)
                    else:
                        # Only pass the number of args the handler accepts
                        positional_params = [
                            p for p in params 
                            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                        ]
                        num_params = len(positional_params)
                        handler_args = args[:num_params]
                        handler(*handler_args)
                        
                except Exception as e:
                    logger.error(f"Error in event handler for {event}: {e}")

    def on(self, event: T_contra, callback: Callable | None = None) -> Callable:
        """
        Register an event handler.
        Can be used as a decorator or regular method.

        Args:
            event: Event type/name to listen for
            callback: Handler function to call when event occurs
        
        Returns:
            The registered callback function
        """
        def register(handler: Callable) -> Callable:
            # Don't allow async handlers
            if asyncio.iscoroutinefunction(handler):
                raise ValueError(
                    "Async event handlers are not supported. Use asyncio.create_task in a sync wrapper instead."
                )
            
            if event not in self._events:
                self._events[event] = set()
            self._events[event].add(handler)
            return handler

        # Used as decorator
        if callback is None:
            return register
        
        # Used as regular method
        return register(callback)

    def once(self, event: T_contra, callback: Callable | None = None) -> Callable:
        """
        Register a one-time event handler that removes itself after execution
        
        Args:
            event: Event type/name to listen for
            callback: Handler function to call when event occurs
            
        Returns:
            The registered callback function
        """
        def create_once_handler(handler: Callable) -> Callable:
            def once_handler(*args: Any) -> Any:
                self.off(event, once_handler)
                return handler(*args)
            return once_handler

        if callback is None:
            # Used as decorator
            def decorator(handler: Callable) -> Callable:
                wrapped = create_once_handler(handler)
                self.on(event, wrapped)
                return handler
            return decorator
        
        # Used as regular method
        wrapped = create_once_handler(callback)
        return self.on(event, wrapped)

    def off(self, event: T_contra, callback: Callable) -> None:
        """
        Remove an event handler
        
        Args:
            event: Event type/name to remove handler from
            callback: Handler function to remove
        """
        if event in self._events:
            self._events[event].discard(callback)
            # Clean up empty event sets
            if not self._events[event]:
                del self._events[event]