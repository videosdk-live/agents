import uuid
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class EvalTurn:
    stt: Optional[Any] = None
    llm: Optional[Any] = None
    tts: Optional[Any] = None
    judge: Optional[Any] = None
    id: Optional[str] = None
    name: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
