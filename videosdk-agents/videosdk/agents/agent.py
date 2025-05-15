from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Literal

from .event_emitter import EventEmitter
from .utils import FunctionTool, is_function_tool

AgentEventTypes = Literal[
    "instructions_updated",
    "tools_updated",
]

class Agent(EventEmitter[AgentEventTypes], ABC):
    """
    Abstract base class for creating custom agents.
    Inherits from EventEmitter to handle agent events and state updates.
    """
    def __init__(self, instructions: str, tools: List[FunctionTool] = []):
        super().__init__()
        self.instructions = instructions
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
        """Register function tools for the agent"""
        for tool in tools:
            if not is_function_tool(tool):
                raise ValueError(f"Tool {tool.__name__ if hasattr(tool, '__name__') else tool} is not a valid FunctionTool")
        
        self._tools.extend(tools)
        self.emit("tools_updated", {"tools": self._tools})

    @abstractmethod
    async def on_enter(self) -> None:
        """Called when session starts"""
        pass
    
    @abstractmethod
    async def on_exit(self) -> None:
        """Called when session ends"""
        pass
