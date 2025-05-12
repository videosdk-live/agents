from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, List, Optional, Literal, Union, Dict

from .event_emitter import EventEmitter
from .utils import FunctionTool, is_function_tool

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
    def __init__(self, instructions: str, tools: List[FunctionTool] = []):
        super().__init__()
        self.instructions = instructions
        self._state = AgentState.IDLE
        self._tools = [] 
        if tools:
            self.register_tools(tools)

    @property
    def instructions(self) -> str:
        return self._instructions

    @instructions.setter
    def instructions(self, value: str) -> None:
        self._instructions = value
        self.emit("instructions_updated", {"instructions": value})

    @property
    def tools(self) -> List[FunctionTool]:
        return self._tools

    def register_tools(self, tools: List[FunctionTool]) -> None:
        """Register new tools to the agent"""
        for tool in tools:
            if not is_function_tool(tool):
                raise ValueError(f"Tool {tool.__name__ if hasattr(tool, '__name__') else tool} is not a valid FunctionTool")
        
        self._tools.extend(tools)
        self.emit("tools_updated", {"tools": self._tools})

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

    @abstractmethod
    async def on_enter(self) -> None:
        """Called when session starts"""
        pass