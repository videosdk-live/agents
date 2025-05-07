from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, List, Optional, Literal, Union, Dict

from .event_emitter import EventEmitter

class AgentState(str, Enum):
    """Enum representing possible states of the agent"""
    SPEAKING = "SPEAKING"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    IDLE = "IDLE"

AgentEventTypes = Literal[
    "instructions_updated",
    "state_updated",
    "message_sent",
    "message_received"
]

class Agent(EventEmitter[AgentEventTypes], ABC):
    """
    Abstract base class for creating custom agents.
    Inherits from EventEmitter to handle agent events and state updates.
    """
    def __init__(self, instructions: str):
        super().__init__()  # Initialize EventEmitter
        self.instructions = instructions
        self._session: Optional["AgentSession"] = None
        self._state = AgentState.IDLE  # Initialize with IDLE state
        
        if self._session:
            self.on("state_changed", self._handle_state_event)

    @property
    def instructions(self) -> str:
        return self._instructions

    @instructions.setter
    def instructions(self, value: str) -> None:
        self._instructions = value
        self.emit("instructions_updated", {"instructions": value})

    @property
    def state(self) -> AgentState:
        return self._state

    def _handle_state_event(self, data: Dict[str, Any]) -> None:
        """Handle state change events from the session"""
        self.update_state(data)
        
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update state from internal events"""
        old_state = self._state
        self._state = AgentState(updates.get('state', self._state.value))
        self.emit("state_updated", {"state": self._state.value})
        self.on_state_changed(old_state, self._state)

    async def on_state_changed(self, old_state: AgentState, new_state: AgentState) -> None:
        """Callback for state changes"""
        pass

    @property
    def session(self) -> "AgentSession":
        """Get the agent's session"""
        if self._session is None:
            raise RuntimeError("Agent session not initialized. Make sure to call AgentSession.start() first")
        return self._session

    @session.setter
    def session(self, session: "AgentSession") -> None:
        """Set the agent's session"""
        self._session = session
        self.emit("instructions_updated", {"instructions": self._instructions})

    @abstractmethod
    async def on_enter(self) -> None:
        """Called when session starts"""
        pass

    async def send_message(self, message: str) -> None:
        """Send a message"""
        await self.session.say(message)
        self.emit("message_sent", {"message": message})