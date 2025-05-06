from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, List, Optional, Literal

class AgentState(str, Enum):
    """Enum representing possible states of the agent"""
    SPEAKING = "SPEAKING"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    IDLE = "IDLE"

class Agent(ABC):
    """
    Base agent protocol that defines the interface for creating custom agents.
    Provides hooks for session lifecycle and tool management.
    """
    
    def __init__(
        self,
        instructions: str,
        **kwargs: Any
    ) -> None:
        """
        Initialize the agent with instructions and tools.
        
        Args:
            instructions: System prompt defining the agent's personality
            tools: Optional list of function tools the agent can use
            **kwargs: Additional configuration options for extensibility
        """
        self.instructions = instructions
        self._state = AgentState.IDLE
        self._state_callback: Optional[Callable[[AgentState], None]] = None


    @abstractmethod
    async def on_enter(self) -> None:
        """Hook called when the session starts."""
        pass

    @property
    def system_instructions(self) -> str:
        """Get the agent's system instructions."""
        return self.instructions

    async def update_instructions(self, new_instructions: str) -> None:
        """
        Update the agent's instructions.
        
        Args:
            new_instructions: The new instructions to use
        """
        self.instructions = new_instructions
    
    @property
    def state(self) -> AgentState:
        """Get the current state of the agent."""
        return self._state

    def on_state_change(self, callback: Callable[[AgentState], None]) -> None:
        """
        Register a callback for state changes.
        
        Args:
            callback: Function to call when agent state changes
        """
        self._state_callback = callback

    def _set_state(self, new_state: AgentState) -> None:
        """
        Internal method to update the agent's state and trigger callback.
        
        Args:
            new_state: The new state to set
        """
        if self._state != new_state:
            self._state = new_state
            if self._state_callback:
                self._state_callback(new_state)