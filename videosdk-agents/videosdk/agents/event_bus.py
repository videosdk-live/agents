from typing import Dict, Set, Callable, TypeVar, Generic, Literal
from .event_emitter import EventEmitter

EventTypes = Literal[
    "instructions_updated",
    "tools_updated",
]

T = TypeVar('T')

class EventBus(EventEmitter[EventTypes]):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            super().__init__()
            self._initialized = True

    @classmethod
    def get_instance(cls) -> 'EventBus':
        return cls()
    
global_event_emitter = EventBus()
