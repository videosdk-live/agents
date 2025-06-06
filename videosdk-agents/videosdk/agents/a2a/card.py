from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import uuid

@dataclass
class AgentCard:
    """Represents an agent's capabilities and identity"""
    id: str
    name: str
    domain: str
    capabilities: List[str]
    description: str
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())