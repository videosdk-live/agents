from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import uuid


@dataclass
class AgentCard:
    """
    Represents an agent's capabilities and identity for agent-to-agent communication.

    Attributes:
        id (str): Unique identifier for the agent. Auto-generated if not provided.
        name (str): Human-readable name of the agent.
        domain (str): The domain or category this agent specializes in.
        capabilities (List[str]): List of capabilities this agent can perform.
        description (str): Detailed description of the agent's purpose and functionality.
        metadata (Optional[Dict[str, Any]]): Additional custom metadata for the agent.
    """
    id: str
    name: str
    domain: str
    capabilities: List[str]
    description: str
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """
        Internal method: Automatically generates a UUID if no ID is provided.
        """
        if not self.id:
            self.id = str(uuid.uuid4())
