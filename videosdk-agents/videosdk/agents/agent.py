from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Literal
import inspect
from .event_bus import global_event_emitter, EventTypes
from .event_emitter import EventEmitter
from .utils import FunctionTool, is_function_tool
from .a2a.protocol import A2AProtocol
from .a2a.card import AgentCard

AgentEventTypes = Literal[
    "instructions_updated",
    "tools_updated",
]

class Agent(EventEmitter[AgentEventTypes], ABC):
    """
    Abstract base class for creating custom agents.
    Inherits from EventEmitter to handle agent events and state updates.
    """
    def __init__(self, instructions: str, tools: List[FunctionTool] = [],agent_id: str = None):
        super().__init__()
        self.instructions = instructions
        self._tools = tools
        self._register_class_tools()
        self.register_tools()
        self.a2a = A2AProtocol(self)  # Initialize A2A protocol
        self._agent_card = None # Store the agent card
        self.id = agent_id or str(uuid.uuid4())

    def _register_class_tools(self) -> None:
        """Register all function tools defined in the class"""
        for name, attr in inspect.getmembers(self):
            if is_function_tool(attr):
                self._tools.append(attr)

    @property
    def instructions(self) -> str:
        return self._instructions

    @instructions.setter
    def instructions(self, value: str) -> None:
        self._instructions = value
        global_event_emitter.emit("instructions_updated", {"instructions": value})

    @property
    def tools(self) -> List[FunctionTool]:
        return self._tools

    def register_tools(self) -> None:
        """Register external function tools for the agent"""
        for tool in self._tools:
            if not is_function_tool(tool):
                raise ValueError(f"Tool {tool.__name__ if hasattr(tool, '__name__') else tool} is not a valid FunctionTool")
        
        global_event_emitter.emit("tools_updated", {"tools": self._tools})

    async def on_enter(self) -> None:
        """Called when session starts"""
        if not self.audio_track and hasattr(self.session, 'pipeline'):
            self.audio_track = CustomAudioStreamTrack(loop=self.session.pipeline.loop)
            self.session.pipeline.model.audio_track = self.audio_track
    

    async def register_a2a(self, card: AgentCard) -> None:
        """Register the agent for A2A communication"""
        self._agent_card = card
        await self.a2a.register(card)

    async def unregister_a2a(self) -> None:
        """Unregister the agent from A2A communication"""
        await self.a2a.unregister()
        self._agent_card = None

    def send_a2a_message(self, message: str) -> None:
        """Send a message to the agent"""
        self.a2a.send_message(message)

    @abstractmethod
    async def on_exit(self) -> None:
        """Called when session ends"""
        pass
